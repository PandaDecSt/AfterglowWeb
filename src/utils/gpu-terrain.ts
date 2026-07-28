const terrainComputeShader = `
struct TerrainUniforms {
  size: f32,
  heightScale: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> u: TerrainUniforms;
@group(0) @binding(1) var heightTex: texture_storage_2d<rgba16float, write>;

fn hash2d(x: f32, y: f32) -> f32 {
  return fract(sin(dot(vec2<f32>(x, y), vec2<f32>(127.1, 311.7))) * 43758.5453123);
}

fn noise2d(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let a = hash2d(i.x, i.y);
  let b = hash2d(i.x + 1.0, i.y);
  let c = hash2d(i.x, i.y + 1.0);
  let d = hash2d(i.x + 1.0, i.y + 1.0);

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: vec2<f32>) -> f32 {
  var value = 0.0;
  var amplitude = 0.5;
  var freq = 1.0;
  var pos = p;
  for (var i = 0; i < 6; i++) {
    value += amplitude * noise2d(pos * freq);
    freq *= 2.0;
    amplitude *= 0.5;
  }
  return value;
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = u32(u.size);
  if (gid.x >= size || gid.y >= size) {
    return;
  }

  let uv = vec2<f32>(f32(gid.x) / u.size, f32(gid.y) / u.size);
  var height = fbm(uv * 4.0);
  height = pow(height, 1.5);
  height *= u.heightScale;

  textureStore(heightTex, gid.xy, vec4<f32>(height, 0.0, 0.0, 0.0));
}
`;

export class GPUTerrain {
  private device: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private heightTexture!: GPUTexture;
  private heightView!: GPUTextureView;
  private sampler!: GPUSampler;
  private dispatched = false;
  readonly size: number;

  constructor(device: GPUDevice, size = 256, heightScale = 10.0) {
    this.device = device;
    this.size = size;
    this.init(heightScale);
  }

  private init(heightScale: number) {
    this.heightTexture = this.device.createTexture({
      label: "gpu-terrain-height",
      size: [this.size, this.size],
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.heightView = this.heightTexture.createView();

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "terrain-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.uniformBuffer.getMappedRange()).set([this.size, heightScale, 0, 0]);
    this.uniformBuffer.unmap();

    const module = this.device.createShaderModule({ code: terrainComputeShader });
    this.pipeline = this.device.createComputePipeline({
      label: "terrain-compute",
      layout: "auto",
      compute: { module, entryPoint: "cs_main" },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.heightView },
      ],
    });
  }

  dispatchOnce(encoder: GPUCommandEncoder) {
    if (this.dispatched) return;
    this.dispatched = true;
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.size / 8), Math.ceil(this.size / 8));
    pass.end();
  }

  get texture() {
    return this.heightTexture;
  }

  get view() {
    return this.heightView;
  }

  get heightSampler() {
    return this.sampler;
  }

  destroy() {
    this.heightTexture.destroy();
    this.uniformBuffer.destroy();
  }
}
