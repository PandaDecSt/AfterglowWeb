import { GPUContext } from "./device";

export interface RenderContext {
  device: GPUDevice;
  time: number;
  deltaTime: number;
  frameIndex: number;
  width: number;
  height: number;
}

export interface RenderPass {
  label: string;
  execute(encoder: GPUCommandEncoder, screenView: GPUTextureView, ctx: RenderContext): void;
}

export class Renderer {
  private gpu: GPUContext;
  private passes: RenderPass[] = [];
  private frameIndex = 0;
  private startTime = performance.now();
  private lastTime = this.startTime;
  private animId = 0;

  onUpdate?: (ctx: RenderContext) => void;
  onPostSubmit?: (ctx: RenderContext) => void;

  constructor(gpu: GPUContext) {
    this.gpu = gpu;
  }

  addPass(pass: RenderPass) {
    this.passes.push(pass);
  }

  removePass(label: string) {
    this.passes = this.passes.filter((p) => p.label !== label);
  }

  clearPasses() {
    this.passes = [];
  }

  getPassCount() {
    return this.passes.length;
  }

  private frame = () => {
    const now = performance.now();
    const time = (now - this.startTime) / 1000;
    const deltaTime = (now - this.lastTime) / 1000;
    this.lastTime = now;
    this.frameIndex++;

    const ctx: RenderContext = {
      device: this.gpu.device,
      time,
      deltaTime,
      frameIndex: this.frameIndex,
      width: this.gpu.canvas.width,
      height: this.gpu.canvas.height,
    };

    this.onUpdate?.(ctx);

    if (this.passes.length > 0) {
      const encoder = this.gpu.device.createCommandEncoder();
      const view = this.gpu.context.getCurrentTexture().createView();

      for (const pass of this.passes) {
        pass.execute(encoder, view, ctx);
      }

      this.gpu.device.queue.submit([encoder.finish()]);
    }

    this.onPostSubmit?.(ctx);
    this.animId = requestAnimationFrame(this.frame);
  };

  start() {
    this.startTime = performance.now();
    this.lastTime = this.startTime;
    this.frameIndex = 0;
    this.animId = requestAnimationFrame(this.frame);
  }

  stop() {
    cancelAnimationFrame(this.animId);
  }

  resetTime() {
    this.startTime = performance.now();
    this.lastTime = this.startTime;
    this.frameIndex = 0;
  }
}
