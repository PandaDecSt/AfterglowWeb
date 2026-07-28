export class WindFieldSystem {
  private device: GPUDevice;
  private windTexture!: GPUTexture;
  private windView!: GPUTextureView;
  sampler!: GPUSampler;
  readonly size: number;

  constructor(device: GPUDevice, size = 64) {
    this.device = device;
    this.size = size;
    this.init();
  }

  private init() {
    const data = this.generateWindField(this.size);

    this.windTexture = this.device.createTexture({
      label: "wind-field",
      size: [this.size, this.size],
      format: "rg32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.windTexture },
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: this.size * 8, rowsPerImage: this.size },
      [this.size, this.size]
    );

    this.windView = this.windTexture.createView();
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
  }

  private generateWindField(size: number): Float32Array {
    const data = new Float32Array(size * size * 2);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = x / size;
        const ny = y / size;

        const angle =
          Math.sin(nx * 6.28 * 2 + ny * 3.14) * 1.5 +
          Math.cos(ny * 6.28 * 3 - nx * 2.0) * 0.8 +
          Math.sin((nx + ny) * 6.28) * 0.5;

        const strength = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(nx * 4.0 + ny * 3.0));

        const idx = (y * size + x) * 2;
        data[idx + 0] = Math.cos(angle) * strength;
        data[idx + 1] = Math.sin(angle) * strength;
      }
    }
    return data;
  }

  update(time: number) {
    const data = new Float32Array(this.size * this.size * 2);
    const t = time * 0.1;

    for (let y = 0; y < this.size; y++) {
      for (let x = 0; x < this.size; x++) {
        const nx = x / this.size;
        const ny = y / this.size;

        const angle =
          Math.sin(nx * 6.28 * 2 + ny * 3.14 + t) * 1.5 +
          Math.cos(ny * 6.28 * 3 - nx * 2.0 + t * 0.7) * 0.8 +
          Math.sin((nx + ny) * 6.28 + t * 1.3) * 0.5;

        const strength = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(nx * 4.0 + ny * 3.0 + t * 0.5));

        const idx = (y * this.size + x) * 2;
        data[idx + 0] = Math.cos(angle) * strength;
        data[idx + 1] = Math.sin(angle) * strength;
      }
    }

    this.device.queue.writeTexture(
      { texture: this.windTexture },
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: this.size * 8, rowsPerImage: this.size },
      [this.size, this.size]
    );
  }

  get texture() {
    return this.windTexture;
  }

  get view() {
    return this.windView;
  }

  getSampleWGSL(): string {
    return `
fn sampleWindField(windTex: texture_2d<f32>, windSampler: sampler, worldXY: vec2<f32>, fieldSize: f32) -> vec2<f32> {
  let uv = worldXY / fieldSize;
  return textureSample(windTex, windSampler, uv).rg;
}
`;
  }

  destroy() {
    this.windTexture.destroy();
  }
}
