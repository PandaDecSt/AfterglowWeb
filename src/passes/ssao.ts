import { GBuffer } from "./gbuffer";

export class GTAOPass {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private aoTexture!: GPUTexture;
  private aoView!: GPUTextureView;
  private uniformBuffer!: GPUBuffer;

  private width = 0;
  private height = 0;

  intensity = 1.0;
  radius = 0.5;
  falloff = 0.5;
  sampleCount = 8;
  stepCount = 4;

  constructor(device: GPUDevice) {
    this.device = device;
    this.uniformBuffer = device.createBuffer({
      label: "gtao-ubo",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.aoTexture?.destroy();
    this.width = width;
    this.height = height;

    this.aoTexture = this.device.createTexture({
      label: "gtao-result",
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aoView = this.aoTexture.createView();
  }

  get texture(): GPUTexture { return this.aoTexture; }
  get view(): GPUTextureView { return this.aoView; }
  get w(): number { return this.width; }
  get h(): number { return this.height; }

  execute(
    encoder: GPUCommandEncoder,
    gbuffer: GBuffer,
    projMatrix: Float32Array | number[],
  ): void {
    this.resize(gbuffer.w, gbuffer.h);

    if (!this.pipeline) {
      this.createPipeline();
    }

    const ubo = new Float32Array(8);
    ubo[0] = this.width;
    ubo[1] = this.height;
    ubo[2] = this.intensity;
    ubo[3] = this.radius;
    ubo[4] = this.falloff;
    ubo[5] = this.sampleCount;
    ubo[6] = this.stepCount;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: gbuffer.depthCopyView },
        { binding: 2, resource: gbuffer.normalView },
        { binding: 3, resource: this.aoView },
      ],
    });

    const pass = encoder.beginComputePass({ label: "gtao" });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.dispatchWorkgroups(
      Math.ceil(this.width / 8),
      Math.ceil(this.height / 8),
    );
    pass.end();
  }

  private createPipeline(): void {
    const code = `
struct Uniforms {
  screenSize: vec2<f32>,
  intensity: f32,
  radius: f32,
  falloff: f32,
  sampleCount: f32,
  stepCount: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var depthTex: texture_2d<f32>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
@group(0) @binding(3) var aoTex: texture_storage_2d<rgba8unorm, write>;

const PI: f32 = 3.14159265359;

fn reconstructViewPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
  let ndcX = uv.x * 2.0 - 1.0;
  let ndcY = 1.0 - uv.y * 2.0;
  let ndcZ = depth * 2.0 - 1.0;
  return vec3<f32>(ndcX, ndcY, ndcZ);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.xy;
  if (pixel.x >= u32(u.screenSize.x) || pixel.y >= u32(u.screenSize.y)) { return; }

  let uv = (vec2<f32>(pixel) + 0.5) / u.screenSize;
  let depth = textureLoad(depthTex, pixel, 0).r;

  if (depth >= 1.0) {
    textureStore(aoTex, pixel, vec4<f32>(1.0, 0.0, 0.0, 0.0));
    return;
  }

  let normalSample = textureLoad(normalTex, pixel, 0);
  let N = normalize(normalSample.rgb);

  let screenRadius = u.radius * u.screenSize.y * 0.1;
  let numSteps = i32(u.stepCount);
  let numDirections = i32(u.sampleCount);

  var ao = 0.0;

  for (var dirIdx = 0; dirIdx < numDirections; dirIdx++) {
    let angle = f32(dirIdx) * PI / f32(numDirections);
    let dir = vec2<f32>(cos(angle), sin(angle));
    var horizons = vec2<f32>(-1.0, -1.0);

    for (var stepIdx = 1; stepIdx <= numSteps; stepIdx++) {
      let t = f32(stepIdx) / f32(numSteps) * screenRadius;
      let offset = dir * t;

      let s1UV = uv + offset / u.screenSize;
      let s1Pixel = vec2<u32>(clamp(vec2<i32>(s1UV * u.screenSize), vec2<i32>(0), vec2<i32>(i32(u.screenSize.x) - 1, i32(u.screenSize.y) - 1)));
      let s1Depth = textureLoad(depthTex, s1Pixel, 0).r;
      let s1ViewZ = reconstructViewPos(s1UV, s1Depth).z;
      let centerViewZ = reconstructViewPos(uv, depth).z;
      let s1Horizon = atan2(s1Depth - depth, t / u.screenSize.y);
      horizons.x = max(horizons.x, s1Horizon);

      let s2UV = uv - offset / u.screenSize;
      let s2Pixel = vec2<u32>(clamp(vec2<i32>(s2UV * u.screenSize), vec2<i32>(0), vec2<i32>(i32(u.screenSize.x) - 1, i32(u.screenSize.y) - 1)));
      let s2Depth = textureLoad(depthTex, s2Pixel, 0).r;
      let s2Horizon = atan2(s2Depth - depth, t / u.screenSize.y);
      horizons.y = max(horizons.y, s2Horizon);
    }

    let nDotDir = N.x * dir.x + N.y * dir.y;
    let nPerpDir = -N.x * dir.y + N.y * dir.x;
    let projN = nDotDir;
    let perpN = N.z;

    let h1 = horizons.x - asin(clamp(projN, -1.0, 1.0));
    let h2 = horizons.y + asin(clamp(projN, -1.0, 1.0));

    var arc = 0.0;
    if (projN > 0.0) {
      arc = 2.0 * (h2 - h1) * projN + perpN * (cos(h1) - cos(h2) + h2 * sin(h1) - h1 * sin(h2));
    } else {
      arc = 2.0 * (h2 - h1) * projN + perpN * (cos(h2) - cos(h1) + h1 * sin(h2) - h2 * sin(h1));
    }
    ao += arc;
  }

  ao = ao / (2.0 * PI * f32(numDirections));
  ao = clamp(1.0 - ao * u.intensity, 0.0, 1.0);

  textureStore(aoTex, pixel, vec4<f32>(ao, 0.0, 0.0, 0.0));
}
`;
    const module = this.device.createShaderModule({ label: "gtao", code });

    this.pipeline = this.device.createComputePipeline({
      label: "gtao-pipeline",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  destroy(): void {
    this.aoTexture?.destroy();
    this.uniformBuffer?.destroy();
  }
}

export class SSAOPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private aoTexture!: GPUTexture;
  private aoView!: GPUTextureView;
  private uniformBuffer!: GPUBuffer;
  private noiseTexture!: GPUTexture;
  private noiseView!: GPUTextureView;
  private kernelBuffer!: GPUBuffer;

  private width = 0;
  private height = 0;

  kernelSize = 32;
  radius = 0.5;
  bias = 0.025;
  intensity = 1.0;

  constructor(device: GPUDevice) {
    this.device = device;
    this.uniformBuffer = device.createBuffer({
      label: "ssao-ubo",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.kernelBuffer = device.createBuffer({
      label: "ssao-kernel",
      size: this.kernelSize * 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.generateNoise();
    this.generateKernel();
  }

  private generateKernel(): void {
    const kernel = new Float32Array(this.kernelSize * 4);
    for (let i = 0; i < this.kernelSize; i++) {
      let x = Math.random() * 2 - 1;
      let y = Math.random() * 2 - 1;
      let z = Math.random();
      const len = Math.sqrt(x * x + y * y + z * z);
      x /= len; y /= len; z /= len;
      let scale = i / this.kernelSize;
      scale = 0.1 + scale * scale * 0.9;
      kernel[i * 4 + 0] = x * scale;
      kernel[i * 4 + 1] = y * scale;
      kernel[i * 4 + 2] = z * scale;
      kernel[i * 4 + 3] = 0;
    }
    this.device.queue.writeBuffer(this.kernelBuffer, 0, kernel as unknown as GPUAllowSharedBufferSource);
  }

  private generateNoise(): void {
    const noiseSize = 4;
    const noiseData = new Float32Array(noiseSize * noiseSize * 4);
    for (let i = 0; i < noiseSize * noiseSize; i++) {
      const theta = Math.random() * Math.PI * 2;
      noiseData[i * 4 + 0] = Math.cos(theta);
      noiseData[i * 4 + 1] = Math.sin(theta);
      noiseData[i * 4 + 2] = 0;
      noiseData[i * 4 + 3] = 0;
    }

    this.noiseTexture = this.device.createTexture({
      label: "ssao-noise",
      size: [noiseSize, noiseSize],
      format: "rgba8snorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.noiseView = this.noiseTexture.createView();

    this.device.queue.writeTexture(
      { texture: this.noiseTexture },
      noiseData as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: noiseSize * 4 },
      [noiseSize, noiseSize],
    );
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.aoTexture?.destroy();
    this.width = width;
    this.height = height;

    this.aoTexture = this.device.createTexture({
      label: "ssao-result",
      size: [width, height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.aoView = this.aoTexture.createView();
  }

  get texture(): GPUTexture { return this.aoTexture; }
  get view(): GPUTextureView { return this.aoView; }

  execute(
    encoder: GPUCommandEncoder,
    gbuffer: GBuffer,
    projMatrix: Float32Array | number[],
    outputView: GPUTextureView,
  ): void {
    this.resize(gbuffer.w, gbuffer.h);

    const ubo = new Float32Array(8);
    ubo[0] = this.width;
    ubo[1] = this.height;
    ubo[2] = this.radius;
    ubo[3] = this.bias;
    ubo[4] = this.intensity;
    ubo[5] = this.kernelSize;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);

    if (!this.pipeline) {
      this.createPipeline();
    }

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.kernelBuffer } },
        { binding: 2, resource: gbuffer.depthCopyView },
        { binding: 3, resource: gbuffer.normalView },
        { binding: 4, resource: this.noiseView },
        { binding: 5, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" }) },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: "ssao",
      colorAttachments: [{
        view: this.aoView,
        clearValue: { r: 1, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline!);
    pass.setBindGroup(0, this.bindGroup!);
    pass.draw(3);
    pass.end();
  }

  private createPipeline(): void {
    const code = `
struct Uniforms {
  screenSize: vec2<f32>,
  radius: f32,
  bias: f32,
  intensity: f32,
  kernelSize: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<uniform> kernel: array<vec4<f32>, 32>;
@group(0) @binding(2) var depthTex: texture_2d<f32>;
@group(0) @binding(3) var normalTex: texture_2d<f32>;
@group(0) @binding(4) var noiseTex: texture_2d<f32>;
@group(0) @binding(5) var noiseSampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3)
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / u.screenSize;
  let depth = textureSample(depthTex, noiseSampler, uv).r;

  if (depth >= 1.0) {
    return vec4<f32>(1.0, 0.0, 0.0, 0.0);
  }

  let normalSample = textureSample(normalTex, noiseSampler, uv);
  let N = normalize(normalSample.rgb);

  let noiseSize = 4.0;
  let noiseUV = uv * u.screenSize / noiseSize;
  let randomVec = textureSample(noiseTex, noiseSampler, noiseUV).xyz;

  let tangent = normalize(randomVec - N * dot(randomVec, N));
  let bitangent = cross(N, tangent);
  let TBN = mat3x3<f32>(tangent, bitangent, N);

  var occlusion = 0.0;
  let kernelSize = i32(u.kernelSize);

  for (var i = 0; i < kernelSize; i++) {
    let sampleDir = TBN * kernel[i].xyz;
    let sampleUV = uv + sampleDir.xy * u.radius / u.screenSize;
    let sampleDepth = textureSample(depthTex, noiseSampler, sampleUV).r;

    let rangeCheck = select(0.0, 1.0, abs(depth - sampleDepth) < u.radius);
    occlusion += select(0.0, 1.0, sampleDepth >= depth + u.bias) * rangeCheck;
  }

  let ao = 1.0 - (occlusion / f32(kernelSize)) * u.intensity;
  return vec4<f32>(clamp(ao, 0.0, 1.0), 0.0, 0.0, 0.0);
}
`;
    const module = this.device.createShaderModule({ label: "ssao", code });

    this.pipeline = this.device.createRenderPipeline({
      label: "ssao-pipeline",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: "rgba8unorm" }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  destroy(): void {
    this.aoTexture?.destroy();
    this.noiseTexture?.destroy();
    this.uniformBuffer?.destroy();
    this.kernelBuffer?.destroy();
  }
}