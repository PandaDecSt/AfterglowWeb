import { RenderContext } from "./renderer";

export interface GraphResourceDesc {
  label: string;
  type: "texture" | "buffer";
  format?: GPUTextureFormat;
  width?: number;
  height?: number;
  usage?: number;
  size?: number;
  bufferUsage?: number;
}

export interface GraphPass {
  label: string;
  reads: string[];
  writes: string[];
  execute(
    encoder: GPUCommandEncoder,
    resolver: ResourceResolver,
    screenView: GPUTextureView,
    ctx: RenderContext
  ): void;
}

export interface ResourceResolver {
  getTexture(label: string): GPUTexture;
  getTextureView(label: string): GPUTextureView;
  getBuffer(label: string): GPUBuffer;
}

export class RenderGraph {
  private device: GPUDevice;
  private passes: GraphPass[] = [];
  private declaredResources = new Map<string, GraphResourceDesc>();
  private allocatedTextures = new Map<string, GPUTexture>();
  private allocatedBuffers = new Map<string, GPUBuffer>();
  private lastWidth = 0;
  private lastHeight = 0;

  constructor(device: GPUDevice) {
    this.device = device;
  }

  declareResource(desc: GraphResourceDesc): void {
    this.declaredResources.set(desc.label, desc);
  }

  addPass(pass: GraphPass): void {
    this.passes.push(pass);
  }

  clearPasses(): void {
    this.passes = [];
  }

  clear(): void {
    this.passes = [];
    this.declaredResources.clear();
    this.destroyAllocated();
  }

  getPassCount(): number {
    return this.passes.length;
  }

  private sortPasses(): GraphPass[] {
    if (this.passes.length <= 1) return this.passes;

    const labelToIndex = new Map<string, number>();
    for (let i = 0; i < this.passes.length; i++) {
      labelToIndex.set(this.passes[i].label, i);
    }

    const resourceWriters = new Map<string, Set<number>>();
    for (let i = 0; i < this.passes.length; i++) {
      for (const w of this.passes[i].writes) {
        if (!resourceWriters.has(w)) resourceWriters.set(w, new Set());
        resourceWriters.get(w)!.add(i);
      }
    }

    const adj = new Map<number, Set<number>>();
    for (let i = 0; i < this.passes.length; i++) adj.set(i, new Set());

    for (let i = 0; i < this.passes.length; i++) {
      for (const r of this.passes[i].reads) {
        const writers = resourceWriters.get(r);
        if (writers) {
          for (const w of writers) {
            if (w !== i) adj.get(w)!.add(i);
          }
        }
      }
    }

    const inDegree = new Array(this.passes.length).fill(0);
    for (const [, targets] of adj) {
      for (const t of targets) inDegree[t]++;
    }

    const queue: number[] = [];
    for (let i = 0; i < this.passes.length; i++) {
      if (inDegree[i] === 0) queue.push(i);
    }

    const sorted: number[] = [];
    while (queue.length > 0) {
      const node = queue.shift()!;
      sorted.push(node);
      for (const target of adj.get(node)!) {
        inDegree[target]--;
        if (inDegree[target] === 0) queue.push(target);
      }
    }

    if (sorted.length !== this.passes.length) {
      console.warn("[RenderGraph] Cycle detected, falling back to insertion order");
      return this.passes;
    }

    return sorted.map((i) => this.passes[i]);
  }

  private ensureResources(screenWidth: number, screenHeight: number): ResourceResolver {
    const usedResources = new Set<string>();
    for (const pass of this.passes) {
      for (const r of pass.reads) usedResources.add(r);
      for (const r of pass.writes) usedResources.add(r);
    }

    const needsResize = screenWidth !== this.lastWidth || screenHeight !== this.lastHeight;

    for (const label of usedResources) {
      const desc = this.declaredResources.get(label);
      if (!desc) continue;

      if (desc.type === "texture") {
        const existing = this.allocatedTextures.get(label);
        const w = desc.width || screenWidth;
        const h = desc.height || screenHeight;

        if (existing && !needsResize && existing.width === w && existing.height === h) {
          continue;
        }

        existing?.destroy();
        this.allocatedTextures.set(label, this.device.createTexture({
          label,
          size: [w, h],
          format: desc.format || "rgba8unorm",
          usage: desc.usage || (GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING),
        }));
      } else if (desc.type === "buffer") {
        if (!this.allocatedBuffers.has(label) && desc.size) {
          this.allocatedBuffers.set(label, this.device.createBuffer({
            label,
            size: desc.size,
            usage: desc.bufferUsage || GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          }));
        }
      }
    }

    this.lastWidth = screenWidth;
    this.lastHeight = screenHeight;

    return {
      getTexture: (l: string) => {
        const t = this.allocatedTextures.get(l);
        if (!t) throw new Error(`[RenderGraph] Texture "${l}" not allocated`);
        return t;
      },
      getTextureView: (l: string) => {
        const t = this.allocatedTextures.get(l);
        if (!t) throw new Error(`[RenderGraph] Texture "${l}" not allocated`);
        return t.createView();
      },
      getBuffer: (l: string) => {
        const b = this.allocatedBuffers.get(l);
        if (!b) throw new Error(`[RenderGraph] Buffer "${l}" not allocated`);
        return b;
      },
    };
  }

  execute(encoder: GPUCommandEncoder, screenView: GPUTextureView, ctx: RenderContext): void {
    if (this.passes.length === 0) return;

    const sorted = this.sortPasses();
    const resolver = this.ensureResources(ctx.width, ctx.height);

    for (const pass of sorted) {
      pass.execute(encoder, resolver, screenView, ctx);
    }
  }

  private destroyAllocated(): void {
    for (const t of this.allocatedTextures.values()) t.destroy();
    for (const b of this.allocatedBuffers.values()) b.destroy();
    this.allocatedTextures.clear();
    this.allocatedBuffers.clear();
  }

  destroy(): void {
    this.destroyAllocated();
    this.declaredResources.clear();
    this.passes = [];
  }
}