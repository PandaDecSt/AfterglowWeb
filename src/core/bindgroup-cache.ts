export class BindGroupCache {
  private device: GPUDevice;
  private cache = new Map<string, GPUBindGroup>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  get(
    key: string,
    layout: GPUBindGroupLayout,
    entries: GPUBindGroupEntry[]
  ): GPUBindGroup {
    const existing = this.cache.get(key);
    if (existing) return existing;

    const bg = this.device.createBindGroup({ layout, entries });
    this.cache.set(key, bg);
    return bg;
  }

  invalidate(key: string) {
    this.cache.delete(key);
  }

  invalidateAll() {
    this.cache.clear();
  }

  get size() {
    return this.cache.size;
  }
}
