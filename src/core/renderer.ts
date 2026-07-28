import { GPUContext } from "./device";
import { PipelineManager } from "./pipeline";
import { ResourceManager } from "./resource";

export interface RenderPass {
  label: string;
  pipeline: GPURenderPipeline;
  draw: (pass: GPURenderPassEncoder, ctx: RenderContext) => void;
}

export interface RenderContext {
  device: GPUDevice;
  resources: ResourceManager;
  pipelines: PipelineManager;
  time: number;
  deltaTime: number;
  frameIndex: number;
}

export class Renderer {
  private ctx: GPUContext;
  private passes: RenderPass[] = [];
  private depthTexture: GPUTexture | null = null;
  private frameIndex = 0;
  private startTime = performance.now();
  private lastTime = this.startTime;
  private animId = 0;

  pipelines: PipelineManager;
  resources: ResourceManager;

  onUpdate?: (ctx: RenderContext) => void;

  constructor(ctx: GPUContext) {
    this.ctx = ctx;
    this.pipelines = new PipelineManager(ctx.device);
    this.resources = new ResourceManager(ctx.device);
  }

  addPass(pass: RenderPass) {
    this.passes.push(pass);
  }

  removePass(label: string) {
    this.passes = this.passes.filter((p) => p.label !== label);
  }

  private ensureDepthTexture() {
    const { width, height } = this.ctx;
    if (
      this.depthTexture &&
      this.depthTexture.width === width &&
      this.depthTexture.height === height
    ) {
      return;
    }
    this.depthTexture?.destroy();
    this.depthTexture = this.ctx.device.createTexture({
      label: "depth",
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private frame = () => {
    const now = performance.now();
    const time = (now - this.startTime) / 1000;
    const deltaTime = (now - this.lastTime) / 1000;
    this.lastTime = now;
    this.frameIndex++;

    const renderCtx: RenderContext = {
      device: this.ctx.device,
      resources: this.resources,
      pipelines: this.pipelines,
      time,
      deltaTime,
      frameIndex: this.frameIndex,
    };

    this.onUpdate?.(renderCtx);
    this.ensureDepthTexture();

    const encoder = this.ctx.device.createCommandEncoder();
    const view = this.ctx.context.getCurrentTexture().createView();

    for (const pass of this.passes) {
      const renderPass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view,
            clearValue: { r: 0.05, g: 0.05, b: 0.1, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: this.depthTexture
          ? {
              view: this.depthTexture.createView(),
              depthClearValue: 1.0,
              depthLoadOp: "clear",
              depthStoreOp: "store",
            }
          : undefined,
      });
      renderPass.setPipeline(pass.pipeline);
      pass.draw(renderPass, renderCtx);
      renderPass.end();
    }

    this.ctx.device.queue.submit([encoder.finish()]);
    this.animId = requestAnimationFrame(this.frame);
  };

  start() {
    this.animId = requestAnimationFrame(this.frame);
  }

  stop() {
    cancelAnimationFrame(this.animId);
  }

  destroy() {
    this.stop();
    this.resources.destroy();
    this.depthTexture?.destroy();
  }
}
