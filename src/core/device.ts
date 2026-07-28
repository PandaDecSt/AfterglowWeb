export class GPUContext {
  device!: GPUDevice;
  context!: GPUCanvasContext;
  format!: GPUTextureFormat;
  canvas!: HTMLCanvasElement;

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

    ctx.device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has("timestamp-query")
        ? ["timestamp-query"]
        : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
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
