export class RenderTarget {
  texture: GPUTexture;
  view: GPUTextureView;
  width: number;
  height: number;
  format: GPUTextureFormat;

  constructor(
    device: GPUDevice,
    width: number,
    height: number,
    format: GPUTextureFormat,
    label: string,
    usage?: GPUTextureUsageFlags
  ) {
    this.width = width;
    this.height = height;
    this.format = format;
    this.texture = device.createTexture({
      label,
      size: [width, height],
      format,
      usage:
        usage ??
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.view = this.texture.createView();
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

  constructor(device: GPUDevice, width: number, height: number, label: string) {
    this.width = width;
    this.height = height;
    this.texture = device.createTexture({
      label,
      size: [width, height],
      format: "depth24plus",
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
      rt.destroy();
      this.targets.set(
        key,
        new RenderTarget(this.device, width, height, rt.format, key)
      );
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
      label
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
