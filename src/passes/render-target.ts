export class RenderTarget {
  texture: GPUTexture;
  view: GPUTextureView;
  width: number;
  height: number;
  format: GPUTextureFormat;
  usage: GPUTextureUsageFlags;
  scale: number;
  label: string;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    format: GPUTextureFormat,
    label: string,
    usage?: GPUTextureUsageFlags,
    scale = 1.0,
  ) {
    this.width = width;
    this.height = height;
    this.format = format;
    this.label = label;
    this.usage = usage ?? GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    this.scale = scale;
    this.texture = device.createTexture({
      label,
      size: [width, height],
      format,
      usage: this.usage,
    });
    this.view = this.texture.createView();
  }

  resize(device: GPUDevice, width: number, height: number): RenderTarget {
    this.texture.destroy();
    this.width = width;
    this.height = height;
    this.texture = device.createTexture({
      label: this.label,
      size: [width, height],
      format: this.format,
      usage: this.usage,
    });
    this.view = this.texture.createView();
    return this;
  }

  destroy() {
    this.texture.destroy();
  }
}


export class DepthTarget {
  texture: GPUTexture;
  view: GPUTextureView;
  width: number;
  height: number;

  constructor(device: GPUDevice, width: number, height: number, label: string, format: GPUTextureFormat = "depth24plus") {
    this.width = width;
    this.height = height;
    this.texture = device.createTexture({
      label,
      size: [width, height],
      format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView();
  }

  destroy() {
    this.texture.destroy();
  }
}

export class PassManager {
  private device: GPUDevice;
  private targets = new Map<string, RenderTarget>();
  private depthTargets = new Map<string, DepthTarget>();
  private currentWidth = 0;
  private currentHeight = 0;
  format: GPUTextureFormat;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
  }

  resize(width: number, height: number) {
    if (width === this.currentWidth && height === this.currentHeight) return;
    this.currentWidth = width;
    this.currentHeight = height;

    for (const [key, rt] of this.targets) {
      const w = Math.max(1, Math.floor(width * rt.scale));
      const h = Math.max(1, Math.floor(height * rt.scale));
      rt.resize(this.device, w, h);
    }
    for (const [key, dt] of this.depthTargets) {
      dt.destroy();
      this.depthTargets.set(key, new DepthTarget(this.device, width, height, key));
    }
  }

  getOrCreateTarget(
    label: string,
    format?: GPUTextureFormat,
    scale = 1.0
  ): RenderTarget {
    const w = Math.max(1, Math.floor(this.currentWidth * scale));
    const h = Math.max(1, Math.floor(this.currentHeight * scale));
    const existing = this.targets.get(label);
    if (existing && existing.width === w && existing.height === h) {
      return existing;
    }
    existing?.destroy();
    const rt = new RenderTarget(
      this.device,
      w,
      h,
      format ?? this.format,
      label,
      undefined,
      scale,
    );
    this.targets.set(label, rt);
    return rt;
  }

  getOrCreateDepth(label: string): DepthTarget {
    const existing = this.depthTargets.get(label);
    if (
      existing &&
      existing.width === this.currentWidth &&
      existing.height === this.currentHeight
    ) {
      return existing;
    }
    existing?.destroy();
    const dt = new DepthTarget(
      this.device,
      this.currentWidth,
      this.currentHeight,
      label
    );
    this.depthTargets.set(label, dt);
    return dt;
  }

  get width() {
    return this.currentWidth;
  }

  get height() {
    return this.currentHeight;
  }

  destroy() {
    for (const rt of this.targets.values()) rt.destroy();
    for (const dt of this.depthTargets.values()) dt.destroy();
    this.targets.clear();
    this.depthTargets.clear();
  }
}
