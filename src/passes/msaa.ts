import { RenderTarget } from "./render-target";

export class MSAARenderTarget {
  private device: GPUDevice;
  sampleCount: number;
  format: GPUTextureFormat;

  private width = 0;
  private height = 0;
  private msaaTexture!: GPUTexture;
  private msaaView!: GPUTextureView;
  private depthTexture!: GPUTexture;
  private depthView!: GPUTextureView;
  resolveTarget!: RenderTarget;

  constructor(
    device: GPUDevice,
    sampleCount = 4,
    format: GPUTextureFormat = "rgba16float"
  ) {
    this.device = device;
    this.sampleCount = sampleCount;
    this.format = format;
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;

    this.msaaTexture?.destroy();
    this.depthTexture?.destroy();
    this.resolveTarget?.destroy();

    this.msaaTexture = this.device.createTexture({
      label: "msaa-color",
      size: [width, height],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: this.sampleCount,
    });
    this.msaaView = this.msaaTexture.createView();

    this.depthTexture = this.device.createTexture({
      label: "msaa-depth",
      size: [width, height],
      format: "depth24plus-stencil8",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
      sampleCount: this.sampleCount,
    });
    this.depthView = this.depthTexture.createView();

    this.resolveTarget = new RenderTarget(
      this.device, width, height, this.format, "msaa-resolve",
      GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    );
  }

  get w(): number { return this.width; }
  get h(): number { return this.height; }

  beginRenderPass(encoder: GPUCommandEncoder, clear = true): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      colorAttachments: [{
        view: this.msaaView,
        resolveTarget: this.resolveTarget.view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: clear ? "clear" : "load",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: clear ? "clear" : "load",
        depthStoreOp: "store",
      },
    });
  }

  get resolvedTexture(): GPUTexture {
    return this.resolveTarget.texture;
  }

  get resolvedView(): GPUTextureView {
    return this.resolveTarget.view;
  }

  destroy(): void {
    this.msaaTexture?.destroy();
    this.depthTexture?.destroy();
    this.resolveTarget?.destroy();
  }
}