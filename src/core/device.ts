export class GPUContext {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  format!: GPUTextureFormat;
  canvas!: HTMLCanvasElement;
  supportsRG11B10 = false;

  static async create(canvas: HTMLCanvasElement): Promise<GPUContext> {
    const ctx = new GPUContext();
    ctx.canvas = canvas;

    if (!navigator.gpu) {
      throw new Error("WebGPU is not supported in this browser.");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("Failed to get GPU adapter.");
    }

    const features: GPUFeatureName[] = [];
    if (adapter.features.has("timestamp-query")) features.push("timestamp-query");
    if (adapter.features.has("rg11b10ufloat-renderable")) {
      features.push("rg11b10ufloat-renderable");
      ctx.supportsRG11B10 = true;
    }

    ctx.device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxColorAttachmentBytesPerSample: Math.min(128, adapter.limits.maxColorAttachmentBytesPerSample),
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      } as any,
    });

    ctx.format = navigator.gpu.getPreferredCanvasFormat();
    ctx.context = canvas.getContext("webgpu")!;
    ctx.context.configure({
      device: ctx.device,
      format: ctx.format,
      alphaMode: "premultiplied",
    });

    ctx.device.lost.then((info) => {
      if (info.reason !== "destroyed") {
        console.warn("GPU device lost:", info.message);
      }
    });

    return ctx;
  }

  resize(width: number, height: number) {
    const dpr = Math.min(window.devicePixelRatio, 2);
    this.canvas.width = Math.floor(width * dpr);
    this.canvas.height = Math.floor(height * dpr);
  }

  get width() {
    return this.canvas.width;
  }

  get height() {
    return this.canvas.height;
  }
}
