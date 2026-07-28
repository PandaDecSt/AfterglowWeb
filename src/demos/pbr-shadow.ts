import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import { mat4, vec3, type Mat4 } from "wgpu-matrix";
import type { EngineContext } from "../core/engine";

const SHADOW_SIZE = 2048;

const shadowShader = `
struct ShadowUniforms {
  lightViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: ShadowUniforms;

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> @builtin(position) vec4<f32> {
  return u.lightViewProj * u.model * vec4<f32>(pos, 1.0);
}
`;

const pbrShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  lightViewProj: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  albedoColor: vec4<f32>,
  params: vec4<f32>,  // metallic, roughness, ao, time
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) lightSpacePos: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  out.position = u.viewProj * worldPos;
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  out.lightSpacePos = u.lightViewProj * worldPos;
  return out;
}

const PI: f32 = 3.14159265359;

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  let denom = NdotH2 * (a2 - 1.0) + 1.0;
  return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
         geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn calcShadow(lightSpacePos: vec4<f32>) -> f32 {
  let projCoords = lightSpacePos.xyz / lightSpacePos.w;
  let uv = vec2<f32>(projCoords.x * 0.5 + 0.5, 1.0 - (projCoords.y * 0.5 + 0.5));

  // Mask: 0 if outside shadow map, 1 if inside
  let inRange = select(0.0, 1.0,
    uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && projCoords.z <= 1.0);

  let clampedUV = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let texelSize = 1.0 / vec2<f32>(textureDimensions(shadowMap));
  var shadow = 0.0;
  let bias = 0.002;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      let sampleUV = clampedUV + vec2<f32>(f32(x), f32(y)) * texelSize;
      shadow += textureSampleCompare(shadowMap, shadowSampler, sampleUV, projCoords.z - bias);
    }
  }
  return (1.0 - shadow / 9.0) * inRange;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);
  let L = normalize(-u.lightDir.xyz);
  let H = normalize(V + L);

  let metallic = u.params.x;
  let roughness = u.params.y;
  let ao = u.params.z;

  let baseColor = u.albedoColor.rgb;
  let F0 = mix(vec3<f32>(0.04), baseColor, metallic);

  // Cook-Torrance BRDF
  let NDF = distributionGGX(N, H, roughness);
  let G = geometrySmith(N, V, L, roughness);
  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);

  let numerator = NDF * G * F;
  let denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
  let specular = numerator / denominator;

  let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
  let NdotL = max(dot(N, L), 0.0);

  // Shadow
  let shadow = calcShadow(in.lightSpacePos);

  // Directional light
  let Lo = (kD * baseColor / PI + specular) * u.lightColor.rgb * NdotL * (1.0 - shadow);

  // Ambient (hemisphere approximation)
  let skyColor = vec3<f32>(0.4, 0.5, 0.7);
  let groundColor = vec3<f32>(0.15, 0.1, 0.08);
  let hemi = mix(groundColor, skyColor, N.y * 0.5 + 0.5);
  let ambient = hemi * baseColor * ao * 0.3;

  var color = ambient + Lo;

  // ACES tonemapping
  color = acesTonemap(color * 1.2);

  // Gamma
  color = pow(color, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(color, 1.0);
}
`;

const groundShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  lightViewProj: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  albedoColor: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var shadowMap: texture_depth_2d;
@group(0) @binding(2) var shadowSampler: sampler_comparison;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) lightSpacePos: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  out.position = u.viewProj * worldPos;
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  out.lightSpacePos = u.lightViewProj * worldPos;
  return out;
}

fn calcShadow(lightSpacePos: vec4<f32>) -> f32 {
  let projCoords = lightSpacePos.xyz / lightSpacePos.w;
  let uv = vec2<f32>(projCoords.x * 0.5 + 0.5, 1.0 - (projCoords.y * 0.5 + 0.5));
  let inRange = select(0.0, 1.0,
    uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && projCoords.z <= 1.0);
  let clampedUV = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
  let texelSize = 1.0 / vec2<f32>(textureDimensions(shadowMap));
  var shadow = 0.0;
  let bias = 0.003;
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      let sampleUV = clampedUV + vec2<f32>(f32(x), f32(y)) * texelSize;
      shadow += textureSampleCompare(shadowMap, shadowSampler, sampleUV, projCoords.z - bias);
    }
  }
  return (1.0 - shadow / 9.0) * inRange;
}

fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let L = normalize(-u.lightDir.xyz);
  let NdotL = max(dot(N, L), 0.0);

  let shadow = calcShadow(in.lightSpacePos);

  // Checkerboard pattern
  let grid = floor(in.uv * 10.0);
  let checker = fract((grid.x + grid.y) * 0.5) < 0.25;
  let baseColor = mix(vec3<f32>(0.35, 0.35, 0.38), vec3<f32>(0.2, 0.2, 0.22), select(0.0, 1.0, checker));

  let diffuse = baseColor * NdotL * (1.0 - shadow) * u.lightColor.rgb;
  let ambient = baseColor * 0.15;
  var color = ambient + diffuse;

  color = acesTonemap(color * 1.2);
  color = pow(color, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(color, 1.0);
}
`;

export class PBRShadowDemo implements Demo {
  label = "PBR + Shadow";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private engine: EngineContext | null = null;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private shadowPipeline!: GPURenderPipeline;
  private pbrPipeline!: GPURenderPipeline;
  private groundPipeline!: GPURenderPipeline;

  private shadowTexture!: GPUTexture;
  private shadowView!: GPUTextureView;
  private shadowSampler!: GPUSampler;

  private sphereVB!: GPUBuffer;
  private sphereIB!: GPUBuffer;
  private sphereIndexCount = 0;
  private groundVB!: GPUBuffer;
  private groundIB!: GPUBuffer;
  private groundIndexCount = 0;

  private sphereUBO!: GPUBuffer;
  private groundUBO!: GPUBuffer;
  private shadowUBO!: GPUBuffer;
  private shadowGroundUBO!: GPUBuffer;

  private pbrBindGroup!: GPUBindGroup;
  private groundBindGroup!: GPUBindGroup;
  private shadowBindGroup!: GPUBindGroup;
  private shadowGroundBindGroup!: GPUBindGroup;

  private depthTexture: GPUTexture | null = null;
  private cachedDepthView: GPUTextureView | null = null;

  private sphereData = new Float32Array(84);
  private groundData = new Float32Array(84);
  private shadowData = new Float32Array(32);
  private shadowGroundData = new Float32Array(32);

  private shadowCode = shadowShader;
  private pbrCode = pbrShader;
  private groundCode = groundShader;

  metallic = 0.9;
  roughness = 0.25;

  init(ctx: GPUContext, camera: Camera, engine?: EngineContext) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;
    this.engine = engine ?? null;

    this.shadowTexture = this.device.createTexture({
      label: "shadow-depth",
      size: [SHADOW_SIZE, SHADOW_SIZE],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this.shadowTexture.createView();
    this.shadowSampler = this.device.createSampler({
      compare: "less",
      magFilter: "linear",
      minFilter: "linear",
    });

    const { vertices: sv, indices: si } = this.createSphere(24, 16);
    this.sphereIndexCount = si.length;
    this.sphereVB = this.createBuffer("sphere-vb", sv, GPUBufferUsage.VERTEX);
    this.sphereIB = this.createBuffer("sphere-ib", si, GPUBufferUsage.INDEX);

    const { vertices: gv, indices: gi } = this.createGround();
    this.groundIndexCount = gi.length;
    this.groundVB = this.createBuffer("ground-vb", gv, GPUBufferUsage.VERTEX);
    this.groundIB = this.createBuffer("ground-ib", gi, GPUBufferUsage.INDEX);

    this.sphereUBO = this.createUniform("sphere-ubo", 336);
    this.groundUBO = this.createUniform("ground-ubo", 336);
    this.shadowUBO = this.createUniform("shadow-sphere-ubo", 128);
    this.shadowGroundUBO = this.createUniform("shadow-ground-ubo", 128);

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    // Shadow pipeline
    const shadowModule = this.device.createShaderModule({ code: shadowShader });
    const shadowBGL = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });
    this.shadowPipeline = this.device.createRenderPipeline({
      label: "shadow-pass",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [shadowBGL] }),
      vertex: { module: shadowModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });

    // PBR pipeline
    const pbrModule = this.device.createShaderModule({ code: pbrShader });
    const pbrBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
      ],
    });
    const pbrLayout = this.device.createPipelineLayout({ bindGroupLayouts: [pbrBGL] });
    this.pbrPipeline = this.device.createRenderPipeline({
      label: "pbr-pass",
      layout: pbrLayout,
      vertex: { module: pbrModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: pbrModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    // Ground pipeline (same layout)
    const groundModule = this.device.createShaderModule({ code: groundShader });
    this.groundPipeline = this.device.createRenderPipeline({
      label: "ground-pass",
      layout: pbrLayout,
      vertex: { module: groundModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: groundModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    // Bind groups
    this.shadowBindGroup = this.device.createBindGroup({
      layout: shadowBGL,
      entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
    });
    this.shadowGroundBindGroup = this.device.createBindGroup({
      layout: shadowBGL,
      entries: [{ binding: 0, resource: { buffer: this.shadowGroundUBO } }],
    });
    this.pbrBindGroup = this.device.createBindGroup({
      layout: pbrBGL,
      entries: [
        { binding: 0, resource: { buffer: this.sphereUBO } },
        { binding: 1, resource: this.shadowView },
        { binding: 2, resource: this.shadowSampler },
      ],
    });
    this.groundBindGroup = this.device.createBindGroup({
      layout: pbrBGL,
      entries: [
        { binding: 0, resource: { buffer: this.groundUBO } },
        { binding: 1, resource: this.shadowView },
        { binding: 2, resource: this.shadowSampler },
      ],
    });
  }

  private createBuffer(label: string, data: Float32Array | Uint16Array, usage: GPUBufferUsageFlags): GPUBuffer {
    const buf = this.device.createBuffer({
      label,
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    if (data instanceof Float32Array) {
      new Float32Array(buf.getMappedRange()).set(data);
    } else {
      new Uint16Array(buf.getMappedRange()).set(data);
    }
    buf.unmap();
    return buf;
  }

  private createUniform(label: string, size: number): GPUBuffer {
    return this.device.createBuffer({
      label,
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  private createSphere(segments: number, rings: number) {
    const verts: number[] = [];
    const inds: number[] = [];
    for (let y = 0; y <= rings; y++) {
      const phi = (y / rings) * Math.PI;
      for (let x = 0; x <= segments; x++) {
        const theta = (x / segments) * Math.PI * 2;
        const nx = Math.sin(phi) * Math.cos(theta);
        const ny = Math.cos(phi);
        const nz = Math.sin(phi) * Math.sin(theta);
        verts.push(nx, ny, nz, nx, ny, nz, x / segments, y / rings);
      }
    }
    for (let y = 0; y < rings; y++) {
      for (let x = 0; x < segments; x++) {
        const a = y * (segments + 1) + x;
        const b = a + segments + 1;
        inds.push(a, a + 1, b, a + 1, b + 1, b);
      }
    }
    return { vertices: new Float32Array(verts), indices: new Uint16Array(inds) };
  }

  private createGround() {
    const s = 8;
    const verts = new Float32Array([
      -s, 0, -s,  0, 1, 0,  0, 0,
       s, 0, -s,  0, 1, 0,  1, 0,
       s, 0,  s,  0, 1, 0,  1, 1,
      -s, 0, -s,  0, 1, 0,  0, 0,
       s, 0,  s,  0, 1, 0,  1, 1,
      -s, 0,  s,  0, 1, 0,  0, 1,
    ]);
    const inds = new Uint16Array([0, 2, 1, 3, 5, 4]);
    return { vertices: verts, indices: inds };
  }

  private ensureDepth() {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (this.depthTexture && this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.cachedDepthView = this.depthTexture.createView();
  }

  private getLightViewProj(): Mat4 {
    const lightDir = vec3.normalize(vec3.create(-0.4, -1.0, -0.3));
    const lightPos = vec3.scale(lightDir, -10);
    const lightView = mat4.lookAt(lightPos, [0, 0, 0], [0, 1, 0]);
    const lightProj = mat4.ortho(-6, 6, -6, 6, 0.1, 30);
    return mat4.mul(lightProj, lightView);
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const lightViewProj = this.getLightViewProj();

    const sphereModel = mat4.mul(
      mat4.translation([0, 1.2, 0]),
      mat4.mul(mat4.rotationY(time * 0.3), mat4.scaling([1.0, 1.0, 1.0]))
    );
    const sphereInvTrans = mat4.transpose(mat4.inverse(sphereModel));

    const groundModel = mat4.identity();
    const groundInvTrans = mat4.identity();

    // Sphere PBR UBO (80 floats = 320 bytes)
    const s = this.sphereData;
    s.set(viewProj as unknown as ArrayLike<number>, 0);
    s.set(sphereModel as unknown as ArrayLike<number>, 16);
    s.set(sphereInvTrans as unknown as ArrayLike<number>, 32);
    s.set(lightViewProj as unknown as ArrayLike<number>, 48);
    s[64] = this.camera.position[0]; s[65] = this.camera.position[1]; s[66] = this.camera.position[2]; s[67] = 1;
    s[68] = -0.4; s[69] = -1.0; s[70] = -0.3; s[71] = 0;
    s[72] = 3.0; s[73] = 3.0; s[74] = 3.0; s[75] = 1;
    s[76] = 0.8; s[77] = 0.3; s[78] = 0.2; s[79] = 1; // albedo
    s[80] = this.metallic; s[81] = this.roughness; s[82] = 1.0; s[83] = time; // params
    this.device.queue.writeBuffer(this.sphereUBO, 0, s as unknown as GPUAllowSharedBufferSource);

    // Ground UBO
    const g = this.groundData;
    g.set(viewProj as unknown as ArrayLike<number>, 0);
    g.set(groundModel as unknown as ArrayLike<number>, 16);
    g.set(groundInvTrans as unknown as ArrayLike<number>, 32);
    g.set(lightViewProj as unknown as ArrayLike<number>, 48);
    g[64] = this.camera.position[0]; g[65] = this.camera.position[1]; g[66] = this.camera.position[2]; g[67] = 1;
    g[68] = -0.4; g[69] = -1.0; g[70] = -0.3; g[71] = 0;
    g[72] = 3.0; g[73] = 3.0; g[74] = 3.0; g[75] = 1;
    g[76] = 0.35; g[77] = 0.35; g[78] = 0.38; g[79] = 1;
    g[80] = 0.0; g[81] = 0.9; g[82] = 1.0; g[83] = time;
    this.device.queue.writeBuffer(this.groundUBO, 0, g as unknown as GPUAllowSharedBufferSource);

    // Shadow UBOs (lightVP + model = 32 floats)
    const sh = this.shadowData;
    sh.set(lightViewProj as unknown as ArrayLike<number>, 0);
    sh.set(sphereModel as unknown as ArrayLike<number>, 16);
    this.device.queue.writeBuffer(this.shadowUBO, 0, sh as unknown as GPUAllowSharedBufferSource);

    const shg = this.shadowGroundData;
    shg.set(lightViewProj as unknown as ArrayLike<number>, 0);
    shg.set(groundModel as unknown as ArrayLike<number>, 16);
    this.device.queue.writeBuffer(this.shadowGroundUBO, 0, shg as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    this.ensureDepth();

    // Pass 1: Shadow map
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.shadowPipeline);
      pass.setVertexBuffer(0, this.sphereVB);
      pass.setIndexBuffer(this.sphereIB, "uint16");
      pass.setBindGroup(0, this.shadowBindGroup);
      pass.drawIndexed(this.sphereIndexCount);

      pass.setVertexBuffer(0, this.groundVB);
      pass.setIndexBuffer(this.groundIB, "uint16");
      pass.setBindGroup(0, this.shadowGroundBindGroup);
      pass.drawIndexed(this.groundIndexCount);
      pass.end();
    }

    // Pass 2: PBR render
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view,
          clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
        depthStencilAttachment: {
          view: this.cachedDepthView!,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });

      pass.setPipeline(this.groundPipeline);
      pass.setBindGroup(0, this.groundBindGroup);
      pass.setVertexBuffer(0, this.groundVB);
      pass.setIndexBuffer(this.groundIB, "uint16");
      pass.drawIndexed(this.groundIndexCount);

      pass.setPipeline(this.pbrPipeline);
      pass.setBindGroup(0, this.pbrBindGroup);
      pass.setVertexBuffer(0, this.sphereVB);
      pass.setIndexBuffer(this.sphereIB, "uint16");
      pass.drawIndexed(this.sphereIndexCount);

      pass.end();
    }
  }

  stats() {
    return {
      drawCalls: 4,
      triangles: this.sphereIndexCount / 3 + this.groundIndexCount / 3,
      custom: {
        "Shadow Map": `${SHADOW_SIZE}x${SHADOW_SIZE}`,
        "PCF": "3x3",
        "BRDF": "Cook-Torrance GGX",
        "Tonemap": "ACES",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "metallic", 0, 1, 0.01).name("Metallic");
    gui.add(this, "roughness", 0.01, 1, 0.01).name("Roughness");
  }

  getShaderStages(): ShaderStageDesc[] {
    return [
      { label: "PBR / Shadow Pass", type: "vertex", code: this.shadowCode },
      { label: "PBR / Sphere", type: "fragment", code: this.pbrCode },
      { label: "PBR / Ground", type: "fragment", code: this.groundCode },
    ];
  }

  onShaderReload(stageLabel: string, code: string): boolean {
    if (stageLabel === "PBR / Shadow Pass") this.shadowCode = code;
    else if (stageLabel === "PBR / Sphere") this.pbrCode = code;
    else if (stageLabel === "PBR / Ground") this.groundCode = code;
    return this.rebuildPipelines();
  }

  private rebuildPipelines(): boolean {
    try {
      const vertexLayout: GPUVertexBufferLayout = {
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
          { shaderLocation: 2, offset: 24, format: "float32x2" },
        ],
      };

      const compile = (label: string, code: string) =>
        this.engine
          ? this.engine.modules.resolveAndCompile(this.device, label, code)
          : this.device.createShaderModule({ label, code });

      const shadowModule = compile("pbr-shadow-pass", this.shadowCode);
      const shadowBGL = this.device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
      });
      this.shadowPipeline = this.device.createRenderPipeline({
        label: "shadow-pass",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [shadowBGL] }),
        vertex: { module: shadowModule, entryPoint: "vs_main", buffers: [vertexLayout] },
        primitive: { topology: "triangle-list", cullMode: "front" },
        depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
      });

      const pbrModule = compile("pbr-sphere", this.pbrCode);
      const pbrBGL = this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        ],
      });
      const pbrLayout = this.device.createPipelineLayout({ bindGroupLayouts: [pbrBGL] });
      this.pbrPipeline = this.device.createRenderPipeline({
        label: "pbr-pass",
        layout: pbrLayout,
        vertex: { module: pbrModule, entryPoint: "vs_main", buffers: [vertexLayout] },
        fragment: { module: pbrModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list", cullMode: "back" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      });

      const groundModule = compile("pbr-ground", this.groundCode);
      this.groundPipeline = this.device.createRenderPipeline({
        label: "ground-pass",
        layout: pbrLayout,
        vertex: { module: groundModule, entryPoint: "vs_main", buffers: [vertexLayout] },
        fragment: { module: groundModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
        primitive: { topology: "triangle-list", cullMode: "back" },
        depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      });

      this.shadowBindGroup = this.device.createBindGroup({
        layout: shadowBGL,
        entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
      });
      this.shadowGroundBindGroup = this.device.createBindGroup({
        layout: shadowBGL,
        entries: [{ binding: 0, resource: { buffer: this.shadowGroundUBO } }],
      });
      this.pbrBindGroup = this.device.createBindGroup({
        layout: pbrBGL,
        entries: [
          { binding: 0, resource: { buffer: this.sphereUBO } },
          { binding: 1, resource: this.shadowView },
          { binding: 2, resource: this.shadowSampler },
        ],
      });
      this.groundBindGroup = this.device.createBindGroup({
        layout: pbrBGL,
        entries: [
          { binding: 0, resource: { buffer: this.groundUBO } },
          { binding: 1, resource: this.shadowView },
          { binding: 2, resource: this.shadowSampler },
        ],
      });

      return true;
    } catch (e) {
      console.error("[PBRShadow] Pipeline rebuild failed:", e);
      return false;
    }
  }

  destroy() {
    this.shadowTexture.destroy();
    this.sphereVB.destroy();
    this.sphereIB.destroy();
    this.groundVB.destroy();
    this.groundIB.destroy();
    this.sphereUBO.destroy();
    this.groundUBO.destroy();
    this.shadowUBO.destroy();
    this.shadowGroundUBO.destroy();
    this.depthTexture?.destroy();
  }
}
