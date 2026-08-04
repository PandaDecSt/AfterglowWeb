// IBL environment map: procedural sky radiance cubemap +
// diffuse (irradiance) and specular (split-sum prefiltered) bakes,
// ready to be sampled by the deferred lighting pass.

export const ENV_BASE_SIZE = 256;
export const ENV_IRRADIANCE_SIZE = 64;
export const ENV_PREFILTER_SIZE = 128;

const PI = Math.PI;

const envBakeWGSL = /* wgsl */ `
struct BakeUniforms {
  faceIndexSamples: vec4<f32>,   // face, sampleCount, unused, unused
  roughness: f32,
  pad: vec3<f32>,
  res: vec2<f32>,
  resPad: vec2<f32>,
  sunDir: vec4<f32>,
  sunColor: vec4<f32>,
  horizonColor: vec4<f32>,
  zenithColor: vec4<f32>,
  skyParams: vec4<f32>,          // groundColor.xyz, sunAngularSoftness
};

@group(0) @binding(0) var env: texture_cube<f32>;
@group(0) @binding(1) var envSampler: sampler;
@group(0) @binding(2) var<uniform> u: BakeUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3)
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

fn dirFromFace(face: u32, uv: vec2<f32>) -> vec3<f32> {
  let u = uv.x * 2.0 - 1.0;
  let v = uv.y * 2.0 - 1.0;
  if (face == 0u) { return normalize(vec3<f32>( 1.0, -v, -u)); }
  if (face == 1u) { return normalize(vec3<f32>(-1.0, -v,  u)); }
  if (face == 2u) { return normalize(vec3<f32>( u,  1.0,  v)); }
  if (face == 3u) { return normalize(vec3<f32>( u, -1.0, -v)); }
  if (face == 4u) { return normalize(vec3<f32>( u, -v,  1.0)); }
  return normalize(vec3<f32>(-u, -v, -1.0));
}

fn skyRadiance(dir: vec3f) -> vec3f {
  let up = dir.y;
  var col: vec3f;
  if (up >= 0.0) {
    col = mix(u.horizonColor.xyz, u.zenithColor.xyz, pow(up, 0.75));
  } else {
    col = u.skyParams.xyz * 0.15;
  }
  let sunCone = cos(u.skyParams.w);
  let sunDirN = normalize(u.sunDir.xyz);
  let cosAlpha = max(dot(dir, sunDirN), 0.0);
  let sun = u.sunColor.w * smoothstep(sunCone, 1.0, cosAlpha) * u.sunColor.xyz;
  let halo = u.sunColor.w * pow(cosAlpha, 8.0) * 0.35 * u.sunColor.xyz;
  return col + sun + halo;
}

@fragment
fn fsSky(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / u.res;
  let face = u32(u.faceIndexSamples.x);
  return vec4<f32>(skyRadiance(dirFromFace(face, uv)), 1.0);
}

fn radicalInverseVDC(bits: u32) -> f32 {
  var b = bits;
  b = (b << 16u) | (b >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return f32(b) * 2.3283064365386963e-10;
}

fn hammersley(i: u32, n: u32) -> vec2f {
  return vec2f(f32(i) / f32(n), radicalInverseVDC(i));
}

fn orthoBasis(N: vec3f) -> mat2x3f {
  let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.y) < 0.999);
  let T = normalize(cross(helper, N));
  let B = cross(N, T);
  return mat2x3f(T, B);
}

fn cosineSampleHemisphere(xi: vec2f, N: vec3f, basis: mat2x3f) -> vec3f {
  let r = sqrt(max(xi.y, 1e-5));
  let phi = 2.0 * 3.14159265359 * xi.x;
  let local = vec3f(r * cos(phi), r * sin(phi), sqrt(max(1.0 - xi.y, 0.0)));
  return normalize(basis[0] * local.x + basis[1] * local.y + N * local.z);
}

fn importanceSampleGGX(xi: vec2f, N: vec3f, basis: mat2x3f, rough: f32) -> vec3f {
  let a = rough * rough;
  let a2 = a * a;
  let phi = 2.0 * 3.14159265359 * xi.x;
  let cosTheta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
  let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
  let local = vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
  return normalize(basis[0] * local.x + basis[1] * local.y + N * local.z);
}

@fragment
fn fsIrradiance(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / u.res;
  let face = u32(u.faceIndexSamples.x);
  let N = dirFromFace(face, uv);
  let basis = orthoBasis(N);
  let n = u32(u.faceIndexSamples.y);
  var irr = vec3f(0.0);
  for (var i = 0u; i < n; i++) {
    let xi = hammersley(i, n);
    let d = cosineSampleHemisphere(xi, N, basis);
    irr += textureSampleLevel(env, envSampler, d, 0.0).rgb * d.z;
  }
  irr *= 3.14159265359 / f32(n);
  return vec4<f32>(irr, 1.0);
}

@fragment
fn fsPrefilter(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / u.res;
  let face = u32(u.faceIndexSamples.x);
  let N = dirFromFace(face, uv);
  let V = N;
  let basis = orthoBasis(N);
  let n = u32(u.faceIndexSamples.y);
  var pre = vec3f(0.0);
  var totalWeight = 0.0;
  for (var i = 0u; i < n; i++) {
    let xi = hammersley(i, n);
    let H = importanceSampleGGX(xi, N, basis, u.roughness);
    let L = normalize(2.0 * dot(V, H) * H - V);
    let nol = max(dot(N, L), 0.0);
    if (nol > 0.0) {
      pre += textureSampleLevel(env, envSampler, L, 0.0).rgb * nol;
      totalWeight += nol;
    }
  }
  pre = pre / max(totalWeight, 1e-3);
  return vec4<f32>(pre, 1.0);
}
`;

export interface EnvSkyParams {
  sunDir: [number, number, number];
  sunIntensity: number;
  sunColor: [number, number, number];
  sunAngularSoftness: number;
  horizonColor: [number, number, number];
  zenithColor: [number, number, number];
  groundColor: [number, number, number];
}

export const DEFAULT_SKY: EnvSkyParams = {
  sunDir: [-0.5, -0.65, 0.35],
  sunIntensity: 9.0,
  sunColor: [1.0, 0.96, 0.88],
  sunAngularSoftness: 0.15,
  horizonColor: [0.25, 0.3, 0.38],
  zenithColor: [0.05, 0.12, 0.3],
  groundColor: [0.03, 0.03, 0.035],
};

export class EnvironmentMap {
  irradianceView!: GPUTextureView;
  prefilterView!: GPUTextureView;
  prefilterTexture!: GPUTexture;
  prefilterMipCount: number;
  private baseEnvView!: GPUTextureView;
  private textures: GPUTexture[] = [];
  private uniformBuffer: GPUBuffer | null = null;

  private skyParams: EnvSkyParams;

  constructor(skyParams: EnvSkyParams = DEFAULT_SKY) {
    this.skyParams = { ...DEFAULT_SKY, ...skyParams };
    this.prefilterMipCount = Math.log2(ENV_PREFILTER_SIZE) + 1;
  }

  bake(device: GPUDevice): void {
    const d = this.skyParams;
    const sunDir = d.sunDir;

    const baseEnv = device.createTexture({
      label: "env-base-sun",
      size: [ENV_BASE_SIZE, ENV_BASE_SIZE, 6],
      dimension: "2d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const irradianceTex = device.createTexture({
      label: "env-irradiance",
      size: [ENV_IRRADIANCE_SIZE, ENV_IRRADIANCE_SIZE, 6],
      dimension: "2d",
      format: "rgba16float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    const prefilterTex = device.createTexture({
      label: "env-prefilter",
      size: [ENV_PREFILTER_SIZE, ENV_PREFILTER_SIZE, 6],
      dimension: "2d",
      format: "rgba16float",
      mipLevelCount: this.prefilterMipCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.baseEnvView = baseEnv.createView({ dimension: "cube" });
    this.irradianceView = irradianceTex.createView({ dimension: "cube" });
    this.prefilterTexture = prefilterTex;
    this.prefilterView = prefilterTex.createView({ dimension: "cube", mipLevelCount: this.prefilterMipCount });

    const sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    const uniformBuffer = device.createBuffer({
      label: "env-bake-uniforms",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uniformBuffer = uniformBuffer;
    this.textures.push(baseEnv, irradianceTex, prefilterTex);

    const module = device.createShaderModule({ label: "env-bake", code: envBakeWGSL });
    const bgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "cube" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

    const makePipeline = (entryPoint: string, size: number) =>
      device.createRenderPipeline({
        label: `env-bake-${entryPoint}`,
        layout: pipelineLayout,
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint, targets: [{ format: "rgba16float" }] },
        primitive: { topology: "triangle-list" },
      });

    const skyPipeline = makePipeline("fsSky", ENV_BASE_SIZE);
    const irrPipeline = makePipeline("fsIrradiance", ENV_IRRADIANCE_SIZE);
    const prePipeline = makePipeline("fsPrefilter", ENV_PREFILTER_SIZE);

    const data = new Float32Array(64);
    const writeUniform = (face: number, sampleCount: number, rough: number, resX: number, resY: number) => {
      data[0] = face;
      data[1] = sampleCount;
      data[2] = 0;
      data[3] = 0;
      data[4] = rough;
      data[5] = 0;
      data[6] = 0;
      data[7] = 0;
      data[8] = resX;
      data[9] = resY;
      data[10] = 0;
      data[11] = 0;
      data[12] = sunDir[0];
      data[13] = sunDir[1];
      data[14] = sunDir[2];
      data[15] = 0;
      data[16] = d.sunColor[0];
      data[17] = d.sunColor[1];
      data[18] = d.sunColor[2];
      data[19] = d.sunIntensity;
      data[20] = d.horizonColor[0];
      data[21] = d.horizonColor[1];
      data[22] = d.horizonColor[2];
      data[23] = 0;
      data[24] = d.zenithColor[0];
      data[25] = d.zenithColor[1];
      data[26] = d.zenithColor[2];
      data[27] = 0;
      data[28] = d.groundColor[0];
      data[29] = d.groundColor[1];
      data[30] = d.groundColor[2];
      data[31] = d.sunAngularSoftness;
      device.queue.writeBuffer(uniformBuffer, 0, data as unknown as GPUAllowSharedBufferSource);
    };

    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: this.baseEnvView },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    // The sky pass renders INTO baseEnv, so it must not also sample it
    // (would be a read+write conflict in the same synchronization scope).
    // fsSky never samples binding 0, so bind a placeholder that is not
    // written in this stage.
    const skyBindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: irradianceTex.createView({ dimension: "cube" }) },
        { binding: 1, resource: sampler },
        { binding: 2, resource: { buffer: uniformBuffer } },
      ],
    });

    const enc = device.createCommandEncoder({ label: "env-bake" });

    const bakeFace = (
      pipeline: GPURenderPipeline,
      texture: GPUTexture,
      face: number,
      res: number,
      clear: boolean,
      mip?: number,
      bg?: GPUBindGroup,
    ) => {
      const view = texture.createView({
        dimension: "2d",
        baseArrayLayer: face,
        arrayLayerCount: 1,
        baseMipLevel: mip ?? 0,
        mipLevelCount: 1,
      });
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        }],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg ?? bindGroup);
      pass.draw(3);
      pass.end();
    };

    // 1. procedural sky into base cube
    writeUniform(0, 0, 0, ENV_BASE_SIZE, ENV_BASE_SIZE);
    for (let face = 0; face < 6; face++) {
      data[0] = face;
      device.queue.writeBuffer(uniformBuffer, 0, data as unknown as GPUAllowSharedBufferSource);
      bakeFace(skyPipeline, baseEnv, face, ENV_BASE_SIZE, true, undefined, skyBindGroup);
    }

    // 2. irradiance bake (diffuse)
    const IRR_SAMPLES = 128;
    data[1] = IRR_SAMPLES;
    for (let face = 0; face < 6; face++) {
      data[0] = face;
      device.queue.writeBuffer(uniformBuffer, 0, data as unknown as GPUAllowSharedBufferSource);
      bakeFace(irrPipeline, irradianceTex, face, ENV_IRRADIANCE_SIZE, true);
    }

    // 3. prefilter bake (specular, one mip per roughness step)
    const PREF_SAMPLES = 128;
    data[1] = PREF_SAMPLES;
    for (let mip = 0; mip < this.prefilterMipCount; mip++) {
      const rough = mip / (this.prefilterMipCount - 1);
      const size = ENV_PREFILTER_SIZE >> mip;
      data[4] = rough;
      data[8] = size;
      data[9] = size;
      for (let face = 0; face < 6; face++) {
        data[0] = face;
        device.queue.writeBuffer(uniformBuffer, 0, data as unknown as GPUAllowSharedBufferSource);
        bakeFace(prePipeline, prefilterTex, face, size, true, mip);
      }
    }

    device.queue.submit([enc.finish()]);
  }

  destroy(): void {
    for (const tex of this.textures) tex?.destroy();
    this.textures.length = 0;
    this.prefilterTexture = null as unknown as GPUTexture;
    this.uniformBuffer?.destroy();
    this.uniformBuffer = null;
  }
}