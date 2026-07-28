export class TerrainSystem {
  private device: GPUDevice;
  private heightTexture!: GPUTexture;
  private heightView!: GPUTextureView;
  sampler!: GPUSampler;
  readonly size: number;

  constructor(device: GPUDevice, size = 256) {
    this.device = device;
    this.size = size;
    this.init();
  }

  private init() {
    const data = this.generateHeightmap(this.size);

    this.heightTexture = this.device.createTexture({
      label: "terrain-height",
      size: [this.size, this.size],
      format: "r32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture: this.heightTexture },
      data as unknown as GPUAllowSharedBufferSource,
      { bytesPerRow: this.size * 4, rowsPerImage: this.size },
      [this.size, this.size]
    );

    this.heightView = this.heightTexture.createView();
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
  }

  private generateHeightmap(size: number): Float32Array {
    const data = new Float32Array(size * size);
    const octaves = 6;
    const persistence = 0.5;
    const lacunarity = 2.0;

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let amplitude = 1.0;
        let frequency = 1.0;
        let value = 0;
        let maxVal = 0;

        for (let o = 0; o < octaves; o++) {
          const nx = (x / size) * frequency * 4.0;
          const ny = (y / size) * frequency * 4.0;
          value += this.simplex2d(nx, ny) * amplitude;
          maxVal += amplitude;
          amplitude *= persistence;
          frequency *= lacunarity;
        }

        value = (value / maxVal) * 0.5 + 0.5;
        value = Math.pow(value, 1.5);
        data[y * size + x] = value * 10.0;
      }
    }
    return data;
  }

  private simplex2d(x: number, y: number): number {
    const F2 = 0.5 * (Math.sqrt(3) - 1);
    const G2 = (3 - Math.sqrt(3)) / 6;
    const s = (x + y) * F2;
    const i = Math.floor(x + s);
    const j = Math.floor(y + s);
    const t = (i + j) * G2;
    const x0 = x - (i - t);
    const y0 = y - (j - t);

    const i1 = x0 > y0 ? 1 : 0;
    const j1 = x0 > y0 ? 0 : 1;

    const x1 = x0 - i1 + G2;
    const y1 = y0 - j1 + G2;
    const x2 = x0 - 1 + 2 * G2;
    const y2 = y0 - 1 + 2 * G2;

    const grad = (hash: number, gx: number, gy: number): number => {
      const h = hash & 7;
      const u = h < 4 ? gx : gy;
      const v = h < 4 ? gy : gx;
      return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    };

    const hash2d = (ix: number, iy: number): number => {
      let h = ix * 374761393 + iy * 668265263;
      h = (h ^ (h >> 13)) * 1274126177;
      return (h ^ (h >> 16)) & 0xff;
    };

    let n0 = 0, n1 = 0, n2 = 0;
    let t0 = 0.5 - x0 * x0 - y0 * y0;
    if (t0 > 0) { t0 *= t0; n0 = t0 * t0 * grad(hash2d(i, j), x0, y0); }
    let t1 = 0.5 - x1 * x1 - y1 * y1;
    if (t1 > 0) { t1 *= t1; n1 = t1 * t1 * grad(hash2d(i + i1, j + j1), x1, y1); }
    let t2 = 0.5 - x2 * x2 - y2 * y2;
    if (t2 > 0) { t2 *= t2; n2 = t2 * t2 * grad(hash2d(i + 1, j + 1), x2, y2); }

    return 70.0 * (n0 + n1 + n2);
  }

  get texture() {
    return this.heightTexture;
  }

  get view() {
    return this.heightView;
  }

  getHeightWGSL(): string {
    return `
fn loadTerrainHeight(heightTex: texture_2d<f32>, heightSampler: sampler, worldXY: vec2<f32>, terrainSize: f32) -> f32 {
  let uv = worldXY / terrainSize + 0.5;
  return textureSample(heightTex, heightSampler, uv).r;
}
`;
  }

  destroy() {
    this.heightTexture.destroy();
  }
}
