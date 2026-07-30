export class ResourceManager {
  private device: GPUDevice;
  private buffers = new Map<string, GPUBuffer>();
  private textures = new Map<string, GPUTexture>();
  private samplers = new Map<string, GPUSampler>();
  private namespace: string;

  constructor(device: GPUDevice, namespace = "") {
    this.device = device;
    this.namespace = namespace;
  }

  private ns(label: string): string {
    return this.namespace ? `${this.namespace}:${label}` : label;
  }

  withNamespace(ns: string): ResourceManager {
    const child = new ResourceManager(this.device, ns);
    child.buffers = this.buffers;
    child.textures = this.textures;
    child.samplers = this.samplers;
    return child;
  }

  createBuffer(
    label: string,
    size: number,
    usage: GPUBufferUsageFlags,
    data?: ArrayBufferView
  ): GPUBuffer {
    const key = this.ns(label);
    const buffer = this.device.createBuffer({
      label: key,
      size: Math.max(size, 4),
      usage: usage | GPUBufferUsage.COPY_DST,
      mappedAtCreation: !!data,
    });
    if (data) {
      const dst = new Uint8Array(buffer.getMappedRange());
      dst.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buffer.unmap();
    }
    this.buffers.set(key, buffer);
    return buffer;
  }

  createUniformBuffer(label: string, data: ArrayBufferView): GPUBuffer {
    return this.createBuffer(
      label,
      data.byteLength,
      GPUBufferUsage.UNIFORM,
      data
    );
  }

  updateBuffer(label: string, data: ArrayBufferView, offset = 0) {
    const buffer = this.buffers.get(this.ns(label));
    if (buffer) {
      this.device.queue.writeBuffer(buffer, offset, data as GPUAllowSharedBufferSource);
    }
  }

  async createTextureFromImage(
    label: string,
    url: string
  ): Promise<GPUTexture> {
    const key = this.ns(label);
    const response = await fetch(url);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);

    const texture = this.device.createTexture({
      label: key,
      size: [bitmap.width, bitmap.height],
      format: "rgba8unorm",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture },
      [bitmap.width, bitmap.height]
    );

    this.textures.set(key, texture);
    bitmap.close();
    return texture;
  }

  createSampler(label: string, desc?: GPUSamplerDescriptor): GPUSampler {
    const key = this.ns(label);
    const sampler = this.device.createSampler(desc);
    this.samplers.set(key, sampler);
    return sampler;
  }

  getBuffer(label: string) {
    return this.buffers.get(this.ns(label));
  }

  getTexture(label: string) {
    return this.textures.get(this.ns(label));
  }

  getSampler(label: string) {
    return this.samplers.get(this.ns(label));
  }

  destroy() {
    for (const b of this.buffers.values()) b.destroy();
    for (const t of this.textures.values()) t.destroy();
    this.buffers.clear();
    this.textures.clear();
    this.samplers.clear();
  }
}
