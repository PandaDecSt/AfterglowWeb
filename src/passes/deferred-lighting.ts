import { GBuffer } from "./gbuffer";
import { LightScene, MAX_LIGHTS } from "../scene/light";
import { ShadingModel } from "../core/shading-model";
import { TOON_BODY, TOON_FACE, TOON_HAIR, TOON_EYE, TOON_EYELASH, NORMAL_DEBUG, UNLIT } from "../core/shading-registry";
import { mat4, type Mat4 } from "wgpu-matrix";

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
  // IBL is optional: a demo that has no environment map still needs *some*
  // resource bound for bindings 11-14 or bind-group creation fails. These 1x1
  // black stand-ins let any demo use the deferred pass without an env probe.
  private fallbackCubeTexture!: GPUTexture;
  private fallbackCubeView!: GPUTextureView;
  private fallbackLutTexture!: GPUTexture;
  private fallbackLutView!: GPUTextureView;
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
    // Textures are zero-initialized by WebGPU, so these read as black.
    this.fallbackCubeTexture = device.createTexture({
      label: "deferred-fallback-env-cube",
      size: [1, 1, 6],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.fallbackCubeView = this.fallbackCubeTexture.createView({ dimension: "cube" });

    this.fallbackLutTexture = device.createTexture({
      label: "deferred-fallback-brdf-lut",
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.fallbackLutView = this.fallbackLutTexture.createView();

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
        // Bindings 11-14 are always present in the layout; without an env probe
        // they point at 1x1 black stand-ins so the IBL terms evaluate to zero.
        { binding: 11, resource: ibl?.irradiance ?? this.fallbackCubeView },
        { binding: 12, resource: ibl?.prefilter ?? this.fallbackCubeView },
        { binding: 13, resource: ibl?.brdfLut ?? this.fallbackLutView },
        { binding: 14, resource: this.linearClampSampler },
      ];

      if (shadowView && shadowSampler && shadowVPBuffer) {
        entries.push(
          { binding: 7, resource: shadowView },
          { binding: 8, resource: shadowSampler },
          { binding: 9, resource: { buffer: shadowVPBuffer } },
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

    // Only directional lights get a shadow map; emit nothing at all when the
    // caller passed no shadow atlas (an empty `if` body reads as a mistake).
    const shadowBlock = hasShadow ? `
    if (lightParams.x == 0.0) {
      shadowVis = sampleShadowPCF(worldPos, N, rl.L);
    }` : "";

    return `
const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 1.0e-10;

// ShadingModelIDs, mirrored from src/core/shading-model.ts + shading-registry.ts.
const SM_STANDARD: f32 = ${ShadingModel.STANDARD}.0;
const SM_TOON: f32 = ${ShadingModel.TOON}.0;
const SM_SKIN: f32 = ${ShadingModel.SKIN}.0;
const SM_HAIR: f32 = ${ShadingModel.HAIR}.0;
const SM_EYE: f32 = ${ShadingModel.EYE}.0;
const SM_TOON_BODY: f32 = ${TOON_BODY}.0;
const SM_TOON_FACE: f32 = ${TOON_FACE}.0;
const SM_TOON_HAIR: f32 = ${TOON_HAIR}.0;
const SM_TOON_EYE: f32 = ${TOON_EYE}.0;
const SM_TOON_EYELASH: f32 = ${TOON_EYELASH}.0;
const SM_NORMAL_DEBUG: f32 = ${NORMAL_DEBUG}.0;
const SM_UNLIT: f32 = ${UNLIT}.0;

struct Uniforms {
  cameraPos: vec4<f32>,
  screenSize: vec4<f32>,
  invViewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  ambient: vec4<f32>,
  envIntensity: vec4<f32>,
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

// ===========================================================================
// Light resolution — one place that turns the packed light array into an
// (incoming direction, radiance) pair. Every BxDF below consumes that pair, so
// adding a shading model never means re-implementing attenuation again.
// ===========================================================================
struct ResolvedLight {
  L: vec3<f32>,
  color: vec3<f32>,
  valid: bool,
};

fn resolveLight(
  lightParams: vec4<f32>, posOrDir: vec4<f32>, colorOrDir2: vec4<f32>, worldPos: vec3<f32>
) -> ResolvedLight {
  var out: ResolvedLight;
  out.L = vec3<f32>(0.0, 1.0, 0.0);
  out.color = vec3<f32>(0.0);
  out.valid = false;

  let lt = i32(lightParams.x);
  if (lt == 0) {
    // Directional.
    out.L = normalize(-posOrDir.xyz);
    out.color = colorOrDir2.rgb * lightParams.y;
    out.valid = true;
  } else if (lt == 1) {
    // Point.
    let toLight = posOrDir.xyz - worldPos;
    let dist = length(toLight);
    out.L = toLight / max(dist, EPSILON);
    let distRatio = dist / lightParams.z;
    var attenuation = saturate(1.0 - pow(distRatio, lightParams.w));
    attenuation = attenuation * attenuation;
    out.color = colorOrDir2.rgb * lightParams.y * attenuation;
    out.valid = true;
  } else if (lt == 2) {
    // Spot.
    let toLight = posOrDir.xyz - worldPos;
    let dist = length(toLight);
    out.L = toLight / max(dist, EPSILON);
    let distRatio = dist / lightParams.z;
    var attenuation = saturate(1.0 - pow(distRatio, 2.0));
    attenuation = attenuation * attenuation;
    let spotDir = normalize(colorOrDir2.xyz);
    let cosAngle = dot(-out.L, spotDir);
    let outerCone = lightParams.w;
    let innerCone = posOrDir.w;
    let spotAtten = saturate((cosAngle - outerCone) / max(innerCone - outerCone, EPSILON));
    attenuation = attenuation * spotAtten * spotAtten;
    out.color = vec3<f32>(1.0) * lightParams.y * attenuation;
    out.valid = true;
  }
  return out;
}

// ===========================================================================
// Shared BRDF building blocks.
// ===========================================================================
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

// Endfield-style rim term, ported verbatim from the old glb-viewer forward
// shader. Kept as a shared helper because all four toon variants use it.
fn endfieldRim(nov: f32, vol: f32, nol: f32, width: f32) -> f32 {
  let rimStepMin = clamp(0.9 - width, 0.0, 0.99);
  let rimStepMax = clamp(1.0 - width, 0.01, 1.0);
  var rim = smoothstep(rimStepMin, rimStepMax, 1.0 - nov);
  rim = rim * max(vol, 0.0) * max(nol + 0.5, 0.0) * 2.0;
  return rim;
}

// ===========================================================================
// BxDFs — one function per ShadingModelID. Each takes the same signature:
// surface (N, V, baseColor) + light (L, lightColor) + the model's two packed
// params from GBuffer material.rg. Adding a look = adding one of these plus a
// dispatch line in shadeOne().
// ===========================================================================

// STANDARD (0): Cook-Torrance GGX metallic/roughness.
fn bxdfStandard(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, metallic: f32, roughness: f32
) -> vec3<f32> {
  let nol = max(dot(N, L), 0.0);
  let f0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let specular = cookTorrance(N, V, L, f0, roughness) * nol;
  let kS = fresnelSchlick(f0, max(dot(N, V), 0.0));
  let kD = (vec3<f32>(1.0) - kS) * (1.0 - metallic);
  let diffuse = kD * baseColor * INV_PI * nol;
  return (diffuse + specular) * lightColor;
}

// TOON (1): generic cel shading — quantized diffuse, stepped specular.
fn quantizeToon(x: f32, bands: f32) -> f32 {
  let b = max(bands, 1.0);
  let scaled = clamp(x, 0.0, 1.0) * b;
  let lo = floor(scaled);
  let frac = scaled - lo;
  let edge = smoothstep(0.9, 1.0, frac);
  return (lo + edge) / max(b - 1.0, 1.0);
}

fn bxdfToon(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, bands: f32
) -> vec3<f32> {
  let nol = max(dot(N, L), 0.0);
  let diffuseTone = quantizeToon(nol, bands);
  var col = baseColor * diffuseTone * lightColor;
  let H = normalize(V + L);
  let spec = smoothstep(0.86, 0.94, max(dot(N, H), 0.0));
  col = col + lightColor * spec * 0.7;
  return col;
}

// SKIN (2): wrap diffuse + reddish back-scatter, a cheap SSS stand-in.
fn bxdfSkin(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, sssStrength: f32, roughness: f32
) -> vec3<f32> {
  let nol = max(dot(N, L), 0.0);
  let wrap = 0.35;
  let wrapNL = max(0.0, (nol + wrap) / (1.0 + wrap));
  let backWrap = max(0.0, (dot(-N, L) + wrap * 2.0) / (1.0 + wrap * 2.0));
  let subsurfaceColor = vec3<f32>(1.0, 0.5, 0.42);
  let diffuse = baseColor * wrapNL + baseColor * subsurfaceColor * backWrap * sssStrength * 0.6;

  let f0 = vec3<f32>(0.03);
  let specular = cookTorrance(N, V, L, f0, roughness) * nol;
  let kS = fresnelSchlick(f0, max(dot(N, V), 0.0));
  let kD = vec3<f32>(1.0) - kS;
  return (kD * diffuse * INV_PI * nol + specular) * lightColor;
}

// HAIR (3): Kajiya-Kay dual-lobe anisotropic specular.
fn bxdfHair(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, roughness: f32, aniso: f32
) -> vec3<f32> {
  let nol = max(dot(N, L), 0.0);
  let up = vec3<f32>(0.0, 1.0, 0.0);
  var T = cross(up, N);
  if (length(T) < 0.001) { T = cross(vec3<f32>(1.0, 0.0, 0.0), N); }
  T = normalize(T);
  let H = normalize(V + L);

  let tH1 = dot(T, H) + aniso * 0.12;
  let tH2 = dot(T, H) - aniso * 0.12;
  let sinTH1 = sqrt(max(1.0 - tH1 * tH1, 0.0));
  let sinTH2 = sqrt(max(1.0 - tH2 * tH2, 0.0));
  let spec1 = pow(sinTH1, mix(90.0, 12.0, roughness)) * 0.85;
  let spec2 = pow(sinTH2, mix(45.0, 6.0, roughness)) * 0.5;

  let diffuse = baseColor * nol * INV_PI;
  let specular = (spec1 + spec2) * vec3<f32>(1.0, 0.95, 0.85);
  return (diffuse + specular) * lightColor;
}

// EYE (4): tight cornea highlight + darkened iris disc.
fn bxdfEye(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, cornea: f32, irisDark: f32
) -> vec3<f32> {
  let nol = max(dot(N, L), 0.0);
  let H = normalize(V + L);
  let spec = pow(max(dot(N, H), 0.0), mix(500.0, 120.0, 1.0 - cornea)) * cornea * 2.0;
  let fresnel = fresnelSchlick(vec3<f32>(0.02), max(dot(N, V), 0.0)).r;
  let iris = baseColor * irisDark * nol;
  return (iris + baseColor * spec + baseColor * fresnel * 0.3) * lightColor;
}

// TOON_BODY (5): Endfield toon ramp + rim, ported from glb-viewer's forward
// mode 1. The warm/cool ramp fakes bounce light on the shadow side.
fn bxdfToonBody(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, rimWidth: f32
) -> vec3<f32> {
  let ndl = dot(N, L);
  let nov = dot(N, V);
  let vol = dot(-V, L);

  let viewFactor = max(mix(1.0, max(nov, 0.0), 0.5), 0.6);
  let radiance = max(ndl, 0.0) * viewFactor;
  let fadedSSS = 0.4 * (0.5 + viewFactor * 0.5);

  let rampOffset = 0.2;
  let rampCoord = clamp((1.0 - rampOffset) - radiance * (0.5 - rampOffset * 0.5), 0.1, 0.9);
  let rampWarm = vec3<f32>(1.0, 0.6, 0.4);
  let rampCool = vec3<f32>(0.7, 0.75, 0.9);
  let rampColor = mix(rampCool, rampWarm, smoothstep(0.3, 0.7, rampCoord));
  let rampAlpha = smoothstep(0.2, 0.8, rampCoord) * 0.6;

  let finalToon = baseColor * (radiance + (rampColor * rampAlpha) * min(vec3<f32>(fadedSSS), baseColor));
  let rim = endfieldRim(nov, vol, ndl, rimWidth);
  return (finalToon + rim * baseColor * 0.25) * lightColor;
}

// TOON_FACE (6): hard SDF-like terminator, no soft ramp — anime faces must not
// pick up nose/brow shadows from the geometric normal.
fn bxdfToonFace(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, rimWidth: f32
) -> vec3<f32> {
  let ndl = dot(N, L);
  let nov = dot(N, V);
  let vol = dot(-V, L);

  let faceRadiance = smoothstep(0.0, 0.1, ndl);
  let viewFactor = mix(1.0, max(nov, 0.0), 0.3);
  let finalFace = baseColor * (0.3 + faceRadiance * 0.7) * viewFactor;
  let shadowTint = baseColor * vec3<f32>(0.9, 0.7, 0.65);
  let faceResult = mix(shadowTint * 0.5, finalFace, faceRadiance);
  let rim = endfieldRim(nov, vol, ndl, rimWidth);
  return (faceResult + rim * 0.2) * lightColor;
}

// TOON_HAIR (7): hard diffuse step + a solid anisotropic streak.
fn bxdfToonHair(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, rimWidth: f32
) -> vec3<f32> {
  let ndl = dot(N, L);
  let nov = dot(N, V);
  let vol = dot(-V, L);
  let H = normalize(V + L);

  let viewFactor = mix(1.0, max(nov, 0.0), 0.4);
  let tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), N) + vec3<f32>(1e-6));
  let tdh = dot(tangent, H);
  let anisoSpec = pow(sqrt(max(1.0 - tdh * tdh, 0.0)), 16.0);
  let specSolid = smoothstep(0.3, 0.5, anisoSpec) * 0.6;

  let hairDiffuse = smoothstep(0.0, 0.15, max(ndl, 0.0));
  let hairColor = baseColor * (0.25 + hairDiffuse * 0.75) * viewFactor;
  let rim = endfieldRim(nov, vol, ndl, rimWidth);
  return (hairColor + specSolid + rim * 0.25) * lightColor;
}

// TOON_EYE (8): high contrast iris + one sharp square highlight.
fn bxdfToonEye(
  N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>
) -> vec3<f32> {
  let H = normalize(V + L);
  let eyeDiffuse = smoothstep(0.0, 0.05, dot(N, L));
  let eyeColor = baseColor * (0.4 + eyeDiffuse * 0.6);
  let eyeSpec = smoothstep(0.85, 0.9, max(dot(N, H), 0.0)) * 0.8;
  return (eyeColor + eyeSpec) * lightColor;
}

// TOON_EYELASH (9): almost flat — lashes must stay a solid graphic shape.
fn bxdfToonEyelash(
  N: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>, baseColor: vec3<f32>
) -> vec3<f32> {
  let lash = 0.5 + max(dot(N, L), 0.0) * 0.5;
  return baseColor * lash * lightColor;
}

// Dispatch one light against one surface. This is the whole "unified pipeline":
// a realistic character and a toon character differ by this single switch.
fn shadeOne(
  matID: f32, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, lightColor: vec3<f32>,
  baseColor: vec3<f32>, pr: f32, pg: f32
) -> vec3<f32> {
  if (matID < 0.5) {
    return bxdfStandard(N, V, L, lightColor, baseColor, pr, max(pg, 0.04));
  } else if (matID < 1.5) {
    return bxdfToon(N, V, L, lightColor, baseColor, max(pg, 1.0));
  } else if (matID < 2.5) {
    return bxdfSkin(N, V, L, lightColor, baseColor, pr, max(pg, 0.04));
  } else if (matID < 3.5) {
    return bxdfHair(N, V, L, lightColor, baseColor, max(pr, 0.04), pg);
  } else if (matID < 4.5) {
    return bxdfEye(N, V, L, lightColor, baseColor, pr, max(pg, 0.1));
  } else if (matID < 5.5) {
    return bxdfToonBody(N, V, L, lightColor, baseColor, pr);
  } else if (matID < 6.5) {
    return bxdfToonFace(N, V, L, lightColor, baseColor, pr);
  } else if (matID < 7.5) {
    return bxdfToonHair(N, V, L, lightColor, baseColor, pr);
  } else if (matID < 8.5) {
    return bxdfToonEye(N, V, L, lightColor, baseColor);
  } else if (matID < 9.5) {
    return bxdfToonEyelash(N, L, lightColor, baseColor);
  }
  // NORMAL_DEBUG / UNLIT never reach here (they return before the light loop).
  return vec3<f32>(0.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / u.screenSize.xy;

  let albedo = textureSample(albedoTex, pointSampler, uv);
  let normalSample = textureSample(normalTex, pointSampler, uv);
  let material = textureSample(materialTex, pointSampler, uv);
  let depth = textureSample(depthTex, pointSampler, uv).r;
  let ssao = textureSample(aoTex, pointSampler, uv).r;

  if (depth >= 1.0) {
    return vec4<f32>(0.02, 0.02, 0.04, 1.0);
  }

  let baseColor = albedo.rgb;
  // albedo.a carries baked occlusion from the GBuffer writer; the AO texture
  // carries screen-space occlusion. Both apply.
  let ao = ssao * albedo.a;
  let N = normalize(normalSample.rgb);
  let matID = material.a;
  // material.rg meaning is per-model (see shading-registry packMeaning).
  let pr = material.r;
  let pg = material.g;
  let emissive = material.b;

  // --- Unlit models: bail out before touching a single light ---------------
  if (matID > SM_NORMAL_DEBUG - 0.5 && matID < SM_NORMAL_DEBUG + 0.5) {
    return vec4<f32>(N * 0.5 + 0.5, 1.0);
  }
  if (matID > SM_UNLIT - 0.5) {
    return vec4<f32>(baseColor, 1.0);
  }

  let worldPos = reconstructWorldPos(uv, depth);
  let V = normalize(u.cameraPos.xyz - worldPos);

  let isStandard = matID < 0.5;
  // The Endfield toon variants bake their own shadow floor into the ramp, so
  // piling scene ambient on top of them just washes the flat colors out.
  let isEndfieldToon = matID > SM_TOON_BODY - 0.5 && matID < SM_TOON_EYELASH + 0.5;
  var ambientScale = 1.0;
  if (isStandard) { ambientScale = 1.0 - min(u.envIntensity.x, 1.0); }
  if (isEndfieldToon) { ambientScale = 0.0; }

  var color = u.ambient.rgb * baseColor * ao * ambientScale;

  // --- One loop, every shading model ---------------------------------------
  let numLights = i32(lights[0]);
  for (var i = 0; i < numLights; i++) {
    let base = 1 + i * 12;
    let lightParams = vec4<f32>(lights[base + 0], lights[base + 1], lights[base + 2], lights[base + 3]);
    let posOrDir = vec4<f32>(lights[base + 4], lights[base + 5], lights[base + 6], lights[base + 7]);
    let colorOrDir2 = vec4<f32>(lights[base + 8], lights[base + 9], lights[base + 10], lights[base + 11]);

    let rl = resolveLight(lightParams, posOrDir, colorOrDir2, worldPos);
    if (!rl.valid) { continue; }

    var shadowVis = 1.0;${shadowBlock}

    color += shadeOne(matID, N, V, rl.L, rl.color, baseColor, pr, pg) * shadowVis;
  }

  // --- Image-based lighting (STANDARD only) --------------------------------
  // Without an env probe bindings 11-13 are 1x1 black, so these terms vanish.
  if (isStandard) {
    let metallic = pr;
    let roughness = max(pg, 0.04);
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
  }

  color += baseColor * emissive;

  return vec4<f32>(color, 1.0);
}
`;
  }

  destroy(): void {
    this.uniformBuffer?.destroy();
    this.lightBuffer?.destroy();
    this.fallbackAOTexture?.destroy();
    this.fallbackCubeTexture?.destroy();
    this.fallbackLutTexture?.destroy();
  }
}
