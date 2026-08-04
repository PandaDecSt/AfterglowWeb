import { GBuffer, GBUFFER_GEOMETRY_WGSL } from "./gbuffer";
import { LightScene, MAX_LIGHTS } from "../scene/light";
import { mat4, vec3, type Mat4 } from "wgpu-matrix";

export class DeferredLightingPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private uniformBuffer!: GPUBuffer;
  private lightBuffer!: GPUBuffer;
  private lightScene: LightScene;
  private outputFormat: GPUTextureFormat;
  private fallbackAOTexture!: GPUTexture;
  private fallbackAOView!: GPUTextureView;
  private cachedHasShadow: boolean | null = null;
  private bindGroupLayout: GPUBindGroupLayout | null = null;
  private nearestSampler!: GPUSampler;
  private linearClampSampler!: GPUSampler;
  private lastGBuffer: GBuffer | null = null;
  private lastShadowView: GPUTextureView | null = null;
  private lastShadowVPBuffer: GPUBuffer | null = null;
  private lastAOView: GPUTextureView | null = null;
  private lastIBL: { irradiance: GPUTextureView; prefilter: GPUTextureView; brdfLut: GPUTextureView } | null = null;

  envIntensity = 1.0;

  private prevViewProj: Mat4 = mat4.identity(mat4.create());

  constructor(device: GPUDevice, lightScene: LightScene, outputFormat: GPUTextureFormat = "rgba16float") {
    this.device = device;
    this.lightScene = lightScene;
    this.outputFormat = outputFormat;

    this.uniformBuffer = device.createBuffer({
      label: "deferred-lighting-ubo",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const lightBufferSize = (1 + MAX_LIGHTS * 12) * 4;
    this.lightBuffer = device.createBuffer({
      label: "deferred-light-buffer",
      size: lightBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.fallbackAOTexture = device.createTexture({
      label: "deferred-fallback-ao",
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.fallbackAOView = this.fallbackAOTexture.createView();
    device.queue.writeTexture(
      { texture: this.fallbackAOTexture },
      new Uint8Array([255, 0, 0, 0]),
      { bytesPerRow: 4 },
      [1, 1],
    );
    this.nearestSampler = this.device.createSampler({ label: "deferred-nearest", magFilter: "nearest", minFilter: "nearest" });
    this.linearClampSampler = this.device.createSampler({
      label: "deferred-linear-clamp",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
      mipmapFilter: "linear",
    });
  }

  update(
    viewProj: Mat4,
    cameraPos: [number, number, number],
    invViewProj: Mat4,
    screenWidth: number,
    screenHeight: number,
    near: number,
    far: number,
  ): void {
    const ubo = new Float32Array(64);
    ubo[0] = cameraPos[0];
    ubo[1] = cameraPos[1];
    ubo[2] = cameraPos[2];
    ubo[3] = 0;
    ubo[4] = screenWidth;
    ubo[5] = screenHeight;
    ubo[6] = near;
    ubo[7] = far;
    ubo.set(invViewProj as unknown as ArrayLike<number>, 8);
    ubo.set(this.prevViewProj as unknown as ArrayLike<number>, 24);
    ubo[40] = this.lightScene.ambientColor[0] * this.lightScene.ambientIntensity;
    ubo[41] = this.lightScene.ambientColor[1] * this.lightScene.ambientIntensity;
    ubo[42] = this.lightScene.ambientColor[2] * this.lightScene.ambientIntensity;
    ubo[43] = 0;
    ubo[44] = this.envIntensity;
    ubo[45] = 0;
    ubo[46] = 0;
    ubo[47] = 0;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);

    const lightData = this.lightScene.buildLightBuffer();
    this.device.queue.writeBuffer(this.lightBuffer, 0, lightData as unknown as GPUAllowSharedBufferSource);

    this.prevViewProj = viewProj;
  }

  execute(
    encoder: GPUCommandEncoder,
    gbuffer: GBuffer,
    outputView: GPUTextureView,
    shadowView?: GPUTextureView,
    shadowSampler?: GPUSampler,
    shadowVPBuffer?: GPUBuffer,
    aoView?: GPUTextureView,
    ibl?: {
      irradiance: GPUTextureView;
      prefilter: GPUTextureView;
      brdfLut: GPUTextureView;
    },
  ): void {
    const needsShadow = shadowView !== undefined;
    if (!this.pipeline || this.cachedHasShadow !== needsShadow) {
      this.createPipeline(needsShadow);
      this.cachedHasShadow = needsShadow;
    }

    if (
      this.bindGroup === null ||
      this.lastGBuffer !== gbuffer ||
      this.lastShadowView !== shadowView ||
      this.lastShadowVPBuffer !== shadowVPBuffer ||
      this.lastAOView !== aoView ||
      this.lastIBL !== ibl
    ) {
      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.lightBuffer } },
        { binding: 2, resource: gbuffer.albedoView },
        { binding: 3, resource: gbuffer.normalView },
        { binding: 4, resource: gbuffer.materialView },
        { binding: 5, resource: gbuffer.depthCopyView },
        { binding: 6, resource: this.nearestSampler },
        { binding: 10, resource: aoView ?? this.fallbackAOView },
      ];

      if (shadowView && shadowSampler && shadowVPBuffer) {
        entries.push(
          { binding: 7, resource: shadowView },
          { binding: 8, resource: shadowSampler },
          { binding: 9, resource: { buffer: shadowVPBuffer } },
        );
      }

      if (ibl) {
        entries.push(
          { binding: 11, resource: ibl.irradiance },
          { binding: 12, resource: ibl.prefilter },
          { binding: 13, resource: ibl.brdfLut },
          { binding: 14, resource: this.linearClampSampler },
        );
      }

      this.bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout!,
        entries,
      });
      this.lastGBuffer = gbuffer;
      this.lastShadowView = shadowView ?? null;
      this.lastShadowVPBuffer = shadowVPBuffer ?? null;
      this.lastAOView = aoView ?? null;
      this.lastIBL = ibl ?? null;
    }

    const pass = encoder.beginRenderPass({
      label: "deferred-lighting",
      colorAttachments: [{
        view: outputView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.draw(3);
    pass.end();
  }

  private createPipeline(hasShadow: boolean): void {
    const code = this.buildShaderCode(hasShadow);
    const module = this.device.createShaderModule({ label: "deferred-lighting", code });

this.bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 11, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
        { binding: 12, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
        { binding: 13, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 14, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        ...(hasShadow
        ? [
            { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } } as GPUBindGroupLayoutEntry,
            { binding: 8, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } } as GPUBindGroupLayoutEntry,
            { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: {} } as GPUBindGroupLayoutEntry,
          ]
          : []),
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      label: "deferred-lighting-pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout!],
      }),
      vertex: {
        module,
        entryPoint: "vs_main",
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: this.outputFormat }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private buildShaderCode(hasShadow: boolean): string {
    const shadowBindings = hasShadow ? `
@group(0) @binding(7) var shadowTex: texture_depth_2d;
@group(0) @binding(8) var shadowSampler: sampler_comparison;
@group(0) @binding(9) var<uniform> shadowVP: mat4x4<f32>;
` : "";

    const shadowSampleFn = hasShadow ? `
fn sampleShadowPCF(worldPos: vec3<f32>, normal: vec3<f32>, lightDir: vec3<f32>) -> f32 {
  if (dot(normal, lightDir) <= 0.0) { return 1.0; }
  let biasedPos = worldPos + normal * 0.08;
  let lclip = shadowVP * vec4<f32>(biasedPos, 1.0);
  let ndc = lclip.xyz / max(lclip.w, 1e-6);
  let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let cmpZ = ndc.z - 0.001;
  let ts = 1.0 / f32(textureDimensions(shadowTex).x);
  let s00 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(shadowTex, shadowSampler, suv, cmpZ);
  let s21 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(shadowTex, shadowSampler, suv + vec2f( ts,  ts), cmpZ);
  var vis = (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);
  let inZ = select(0.0, 1.0, ndc.z > 0.0 && ndc.z < 1.0);
  let frustum = (1.0 - smoothstep(0.88, 0.96, abs(ndc.x)))
              * (1.0 - smoothstep(0.88, 0.96, abs(ndc.y))) * inZ;
  return mix(1.0, vis, frustum);
}
` : "";


    return `
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 1.0e-10;

struct Uniforms {
  cameraPos: vec4<f32>,
  screenSize: vec4<f32>,
  invViewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  ambient: vec4<f32>,
  envIntensity: vec4<f32>,
};

struct LightData {
  typeAndParams: vec4<f32>,
  posOrDir: vec4<f32>,
  colorOrDir2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> lights: array<f32, ${(1 + MAX_LIGHTS * 12)}>;
@group(0) @binding(2) var albedoTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;
@group(0) @binding(4) var materialTex: texture_2d<f32>;
@group(0) @binding(5) var depthTex: texture_2d<f32>;
@group(0) @binding(6) var pointSampler: sampler;
@group(0) @binding(10) var aoTex: texture_2d<f32>;
@group(0) @binding(11) var irradianceTex: texture_cube<f32>;
@group(0) @binding(12) var prefilterTex: texture_cube<f32>;
@group(0) @binding(13) var brdfLutTex: texture_2d<f32>;
@group(0) @binding(14) var envSampler: sampler;
${shadowBindings}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3)
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

fn reconstructWorldPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
  let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
  let world = u.invViewProj * ndc;
  return world.xyz / world.w;
}

${shadowSampleFn}

fn distributionGGX(a2: f32, noh: f32) -> f32 {
  let d = (noh * a2 - noh) * noh + 1.0;
  return a2 / (PI * d * d + EPSILON);
}

fn visibilitySmithJointApprox(a2: f32, nov: f32, nol: f32) -> f32 {
  let a = sqrt(a2);
  let smithV = nol * (nov * (1.0 - a) + a);
  let smithL = nov * (nol * (1.0 - a) + a);
  return 0.5 / (smithV + smithL + EPSILON);
}

fn fresnelSchlick(f0: vec3<f32>, voh: f32) -> vec3<f32> {
  let fc = pow(1.0 - voh, 5.0);
  return saturate(50.0 * f0.g) * fc + (1.0 - fc) * f0;
}

fn brdfLutSample(nv: f32, roughness: f32) -> vec2<f32> {
  let LUT_SIZE: f32 = 64.0;
  var uv = vec2<f32>(clamp(roughness, 0.0, 1.0), sqrt(clamp(1.0 - nv, 0.0, 1.0)));
  uv = uv * ((LUT_SIZE - 1.0) / LUT_SIZE) + 0.5 / LUT_SIZE;
  return textureSampleLevel(brdfLutTex, envSampler, uv, 0.0).xy;
}

fn brdfMultiScatter(f0: vec3<f32>, f90: vec3<f32>, lut: vec2<f32>) -> vec3<f32> {
  let FssEss = lut.y * f90 + lut.x * f0;
  let Ess = lut.x + lut.y;
  let Ems = 1.0 - Ess;
  let Favg = f0 + (1.0 - f0) / 21.0;
  let Fms = FssEss * Favg / (1.0 - (1.0 - Ess) * Favg);
  return FssEss + Fms * Ems;
}

fn cookTorrance(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
  let H = normalize(V + L);
  let a2 = pow(roughness, 4.0);
  let noh = max(dot(N, H), 0.0);
  let nov = max(dot(N, V), 1e-6);
  let nol = max(dot(N, L), 0.0);
  let voh = max(dot(V, H), 0.0);
  let D = distributionGGX(a2, noh);
  let G = visibilitySmithJointApprox(a2, nov, nol);
  let F = fresnelSchlick(f0, voh);
  return D * G * F;
}

fn evalLight(
  lightType: f32, lightParams: vec4<f32>, posOrDir: vec4<f32>, colorOrDir2: vec4<f32>,
  worldPos: vec3<f32>, N: vec3<f32>, V: vec3<f32>,
  baseColor: vec3<f32>, metallic: f32, roughness: f32
) -> vec3<f32> {
  var L: vec3<f32>;
  var attenuation = 1.0;
  var lightColor: vec3<f32>;
  let lt = i32(lightType);

  if (lt == 0) {
    L = normalize(-posOrDir.xyz);
    lightColor = colorOrDir2.rgb * lightParams.y;
  } else if (lt == 1) {
    let toLight = posOrDir.xyz - worldPos;
    let dist = length(toLight);
    L = toLight / max(dist, EPSILON);
    let range = lightParams.z;
    let falloff = lightParams.w;
    let distRatio = dist / range;
    attenuation = saturate(1.0 - pow(distRatio, falloff));
    attenuation = attenuation * attenuation;
    lightColor = colorOrDir2.rgb * lightParams.y * attenuation;
  } else if (lt == 2) {
    let toLight = posOrDir.xyz - worldPos;
    let dist = length(toLight);
    L = toLight / max(dist, EPSILON);
    let range = lightParams.z;
    let distRatio = dist / range;
    attenuation = saturate(1.0 - pow(distRatio, 2.0));
    attenuation = attenuation * attenuation;
    let spotDir = normalize(colorOrDir2.xyz);
    let cosAngle = dot(-L, spotDir);
    let outerCone = lightParams.w;
    let innerCone = posOrDir.w;
    let spotAtten = saturate((cosAngle - outerCone) / max(innerCone - outerCone, EPSILON));
    attenuation *= spotAtten * spotAtten;
    lightColor = vec3<f32>(1.0, 1.0, 1.0) * lightParams.y * attenuation;
  } else {
    return vec3<f32>(0.0);
  }

  let nol = max(dot(N, L), 0.0);
  if (nol <= 0.0) { return vec3<f32>(0.0); }

  let f0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let specular = cookTorrance(N, V, L, f0, roughness) * nol;

  let kS = fresnelSchlick(f0, max(dot(N, V), 0.0));
  let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
  let diffuse = kD * baseColor * INV_PI * nol;

  return (diffuse + specular) * lightColor;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / u.screenSize.xy;

  let albedo = textureSample(albedoTex, pointSampler, uv);
  let normalSample = textureSample(normalTex, pointSampler, uv);
  let material = textureSample(materialTex, pointSampler, uv);
  let depth = textureSample(depthTex, pointSampler, uv).r;
  let ao = textureSample(aoTex, pointSampler, uv).r;

  if (depth >= 1.0) {
    return vec4<f32>(0.02, 0.02, 0.04, 1.0);
  }

  let baseColor = albedo.rgb;
  let N = normalize(normalSample.rgb);

  let metallic = material.r;
  let roughness = max(material.g, 0.04);
  let emissive = material.b;

  let worldPos = reconstructWorldPos(uv, depth);
  let V = normalize(u.cameraPos.xyz - worldPos);

  var color = u.ambient.rgb * baseColor * ao * (1.0 - min(u.envIntensity.x, 1.0));

  let numLights = i32(lights[0]);
  for (var i = 0; i < numLights; i++) {
    let base = 1 + i * 12;
    let lightType = lights[base + 0];
    let lightParams = vec4<f32>(lights[base + 0], lights[base + 1], lights[base + 2], lights[base + 3]);
    let posOrDir = vec4<f32>(lights[base + 4], lights[base + 5], lights[base + 6], lights[base + 7]);
    let colorOrDir2 = vec4<f32>(lights[base + 8], lights[base + 9], lights[base + 10], lights[base + 11]);

    var shadowVis = 1.0;
    if (lightType == 0.0) {
      let L = normalize(-posOrDir.xyz);
      ${hasShadow ? "shadowVis = sampleShadowPCF(worldPos, N, L);" : "let _l = L;"}
    }

    color += evalLight(lightType, lightParams, posOrDir, colorOrDir2, worldPos, N, V, baseColor, metallic, roughness) * shadowVis;
  }

  // IBL: split-sum integration (irradiance + prefiltered specular, multi-scatter)
  let NdotV = max(dot(N, V), 0.0);
  let f0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let f90 = vec3<f32>(saturate(f0.g * 50.0));
  let lut = brdfLutSample(NdotV, roughness);
  let Fms = brdfMultiScatter(f0, f90, lut);

  let kD = (vec3<f32>(1.0) - Fms) * (1.0 - metallic);
  let irradiance = textureSampleLevel(irradianceTex, envSampler, N, 0.0).rgb;
  let diffuseIBL = irradiance * baseColor * kD;

  let R = reflect(-V, N);
  let mip = roughness * f32(textureNumLevels(prefilterTex) - 1u);
  let prefiltered = textureSampleLevel(prefilterTex, envSampler, R, mip).rgb;
  let specularIBL = prefiltered * Fms;

  color += (diffuseIBL + specularIBL) * u.envIntensity.x * ao;

  color += baseColor * emissive;

  return vec4<f32>(color, 1.0);
}
`;
  }

  destroy(): void {
    this.uniformBuffer?.destroy();
    this.lightBuffer?.destroy();
    this.fallbackAOTexture?.destroy();
  }
}