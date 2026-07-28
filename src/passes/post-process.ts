const postProcessShader = `
struct PostParams {
  resolution: vec2<f32>,
  time: f32,
  chromaticStrength: f32,
  fogDensity: f32,
  vignetteStrength: f32,
  exposure: f32,
  saturation: f32,
  fogColor: vec3<f32>,
  pad: f32,
};

@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var texSampler: sampler;
@group(0) @binding(3) var depthSampler: sampler;
@group(0) @binding(4) var<uniform> pp: PostParams;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(pos[vi], 0.0, 1.0);
  out.uv = pos[vi] * 0.5 + 0.5;
  return out;
}

fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSRGB(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  let s = step(vec3<f32>(0.0031308), c);
  return mix(lo, hi, s);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  var uv = in.uv;

  // Chromatic aberration
  let dir = uv - 0.5;
  let dist = length(dir);
  let offset = dir * dist * pp.chromaticStrength;
  let r = textureSample(sceneTex, texSampler, uv + offset).r;
  let g = textureSample(sceneTex, texSampler, uv).g;
  let b = textureSample(sceneTex, texSampler, uv - offset).b;
  var color = vec3<f32>(r, g, b);

  // Exposure
  color *= pp.exposure;

  // Height fog (screen-space approximation using depth)
  let depth = textureSampleLevel(depthTex, depthSampler, uv, 0);
  let fogFactor = 1.0 - exp(-depth * pp.fogDensity * 20.0);
  color = mix(color, pp.fogColor, clamp(fogFactor, 0.0, 1.0));

  // ACES Tonemapping
  color = acesTonemap(color);

  // Saturation
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  color = mix(vec3<f32>(luma), color, pp.saturation);

  // Vignette
  let vignette = 1.0 - pp.vignetteStrength * dist * dist * 2.0;
  color *= vignette;

  // Gamma
  color = linearToSRGB(color);

  // Subtle film grain
  let grain = fract(sin(dot(uv * pp.resolution + pp.time * 100.0, vec2<f32>(12.9898, 78.233))) * 43758.5453);
  color += (grain - 0.5) * 0.015;

  return vec4<f32>(color, 1.0);
}
`;

export interface PostProcessParams {
  chromaticStrength: number;
  fogDensity: number;
  fogColor: [number, number, number];
  vignetteStrength: number;
  exposure: number;
  saturation: number;
}

export class PostProcessPass {
  private device: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private paramBuffer!: GPUBuffer;
  private sampler!: GPUSampler;
  private depthSampler!: GPUSampler;
  private paramData = new Float32Array(12);
  private cachedSceneTex: GPUTexture | null = null;
  private cachedDepthTex: GPUTexture | null = null;
  private cachedSceneView: GPUTextureView | null = null;
  private cachedDepthView: GPUTextureView | null = null;
  private cachedBindGroup: GPUBindGroup | null = null;
  params: PostProcessParams = {
    chromaticStrength: 0.003,
    fogDensity: 0.02,
    fogColor: [0.4, 0.5, 0.6],
    vignetteStrength: 0.4,
    exposure: 1.2,
    saturation: 1.1,
  };

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.depthSampler = device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });
    this.paramBuffer = device.createBuffer({
      label: "postprocess-params",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const module = device.createShaderModule({ code: postProcessShader });
    this.pipeline = device.createRenderPipeline({
      label: "postprocess",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureBindGroup(sceneTexture: GPUTexture, depthTexture: GPUTexture) {
    if (this.cachedSceneTex === sceneTexture && this.cachedDepthTex === depthTexture && this.cachedBindGroup) {
      return;
    }
    this.cachedSceneTex = sceneTexture;
    this.cachedDepthTex = depthTexture;
    this.cachedSceneView = sceneTexture.createView();
    this.cachedDepthView = depthTexture.createView();
    this.cachedBindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.cachedSceneView },
        { binding: 1, resource: this.cachedDepthView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.depthSampler },
        { binding: 4, resource: { buffer: this.paramBuffer } },
      ],
    });
  }

  execute(
    encoder: GPUCommandEncoder,
    sceneTexture: GPUTexture,
    depthTexture: GPUTexture,
    outputView: GPUTextureView,
    resolution: [number, number],
    time: number
  ) {
    const p = this.params;
    const d = this.paramData;
    d[0] = resolution[0];
    d[1] = resolution[1];
    d[2] = time;
    d[3] = p.chromaticStrength;
    d[4] = p.fogDensity;
    d[5] = p.vignetteStrength;
    d[6] = p.exposure;
    d[7] = p.saturation;
    d[8] = p.fogColor[0];
    d[9] = p.fogColor[1];
    d[10] = p.fogColor[2];
    d[11] = 0;
    this.device.queue.writeBuffer(this.paramBuffer, 0, d as unknown as GPUAllowSharedBufferSource);

    this.ensureBindGroup(sceneTexture, depthTexture);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: outputView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.cachedBindGroup!);
    pass.draw(3);
    pass.end();
  }

  destroy() {
    this.paramBuffer.destroy();
  }
}
