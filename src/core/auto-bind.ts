import { ResourceManager } from "./resource";

export type BindingResourceType =
  | "uniform"
  | "storage"
  | "texture"
  | "sampler"
  | "storage-texture";

export interface BindingDecl {
  group: number;
  binding: number;
  type: BindingResourceType;
  resource: string;
  offset?: number;
  size?: number;
}

export interface BindGroupDecl {
  group: number;
  entries: BindingDecl[];
}

export class AutoBinder {
  private device: GPUDevice;
  private resources: ResourceManager;
  private cache = new Map<string, GPUBindGroup>();
  private layoutCache = new Map<string, GPUBindGroupLayout>();

  constructor(device: GPUDevice, resources: ResourceManager) {
    this.device = device;
    this.resources = resources;
  }

  private resolveEntry(decl: BindingDecl): GPUBindGroupEntry | null {
    const base = { binding: decl.binding };

    switch (decl.type) {
      case "uniform":
      case "storage": {
        const buffer = this.resources.getBuffer(decl.resource);
        if (!buffer) return null;
        return {
          ...base,
          resource: {
            buffer,
            offset: decl.offset ?? 0,
            size: decl.size,
          },
        };
      }
      case "texture": {
        const texture = this.resources.getTexture(decl.resource);
        if (!texture) return null;
        return { ...base, resource: texture.createView() };
      }
      case "storage-texture": {
        const texture = this.resources.getTexture(decl.resource);
        if (!texture) return null;
        return { ...base, resource: texture.createView() };
      }
      case "sampler": {
        const sampler = this.resources.getSampler(decl.resource);
        if (!sampler) return null;
        return { ...base, resource: sampler };
      }
    }
  }

  private cacheKey(decls: BindingDecl[]): string {
    return decls
      .map((d) => `${d.group}:${d.binding}:${d.type}:${d.resource}:${d.offset ?? 0}`)
      .join("|");
  }

  getBindGroup(
    pipeline: GPURenderPipeline | GPUComputePipeline,
    decls: BindingDecl[]
  ): GPUBindGroup | null {
    const key = this.cacheKey(decls);
    const cached = this.cache.get(key);
    if (cached) return cached;

    const group = decls[0]?.group ?? 0;
    const layout = pipeline.getBindGroupLayout(group);

    const entries: GPUBindGroupEntry[] = [];
    for (const decl of decls) {
      const entry = this.resolveEntry(decl);
      if (!entry) {
        console.warn(`[AutoBinder] Resource not found: ${decl.resource}`);
        return null;
      }
      entries.push(entry);
    }

    const bg = this.device.createBindGroup({ layout, entries });
    this.cache.set(key, bg);
    return bg;
  }

  getBindGroups(
    pipeline: GPURenderPipeline | GPUComputePipeline,
    groups: BindGroupDecl[]
  ): (GPUBindGroup | null)[] {
    return groups.map((g) => this.getBindGroup(pipeline, g.entries));
  }

  invalidate(key?: string) {
    if (key) {
      this.cache.delete(key);
    } else {
      this.cache.clear();
    }
  }

  invalidateForResource(resourceName: string) {
    for (const [key] of this.cache) {
      if (key.includes(resourceName)) {
        this.cache.delete(key);
      }
    }
  }
}
