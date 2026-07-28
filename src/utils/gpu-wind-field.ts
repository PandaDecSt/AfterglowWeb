const windComputeShader = `
struct WindUniforms {
  time: f32,
  size: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> u: WindUniforms;
@group(0) @binding(1) var windTex: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = u32(u.size);
  if (gid.x >= size || gid.y >= size) {
    return;
  }

  let nx = f32(gid.x) / u.size;
  let ny = f32(gid.y) / u.size;
  let t = u.time * 0.1;

  let angle =
    sin(nx * 6.28318 * 2.0 + ny * 3.14159 + t) * 1.5 +
    cos(ny * 6.28318 * 3.0 - nx * 2.0 + t * 0.7) * 0.8 +
    sin((nx + ny) * 6.28318 + t * 1.3) * 0.5;

  let strength = 0.3 + 0.7 * (0.5 + 0.5 * sin(nx * 4.0 + ny * 3.0 + t * 0.5));

  let wind = vec2<f32>(cos(angle), sin(angle)) * strength;
  textureStore(windTex, gid.xy, vec4<f32>(wind, 0.0, 0.0));
}
`;

export class GPUWindField {
  private device: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private windTexture!: GPUTexture;
  private uniformData = new Float32Array(4);
  readonly size: number;

  constructor(device: GPUDevice, size = 64) {
    this.device = device;
    this.size = size;
    this.init();
  }

  private init() {
    this.windTexture = this.device.createTexture({
      label: "gpu-wind-field",
      size: [this.size, this.size],
      format: "rg32float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "wind-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const module = this.device.createShaderModule({ code: windComputeShader });
    this.pipeline = this.device.createComputePipeline({
      label: "wind-compute",
      layout: "auto",
      compute: { module, entryPoint: "cs_main" },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.windTexture.createView() },
      ],
    });
  }

  dispatch(encoder: GPUCommandEncoder, time: number) {
    this.uniformData[0] = time;
    this.uniformData[1] = this.size;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as unknown as GPUAllowSharedBufferSource);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.size / 8), Math.ceil(this.size / 8));
    pass.end();
  }

  get texture() {
    return this.windTexture;
  }

  get view() {
    return this.windTexture.createView();
  }

  createSampler(): GPUSampler {
    return this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
  }

  destroy() {
    this.windTexture.destroy();
    this.uniformBuffer.destroy();
  }
}
