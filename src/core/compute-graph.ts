export interface ComputeNodeDesc {
  label: string;
  code: string;
  entryPoint: string;
  workgroups: [number, number, number];
  bindings: {
    binding: number;
    type: "uniform" | "storage" | "storage-read" | "texture" | "storage-texture";
    resource: string;
  }[];
}

export interface ComputeNode {
  desc: ComputeNodeDesc;
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
}

export class ComputeGraph {
  private device: GPUDevice;
  private nodes: ComputeNode[] = [];
  private buffers = new Map<string, GPUBuffer>();
  private textures = new Map<string, GPUTexture>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  registerBuffer(name: string, buffer: GPUBuffer) {
    this.buffers.set(name, buffer);
  }

  registerTexture(name: string, texture: GPUTexture) {
    this.textures.set(name, texture);
  }

  getBuffer(name: string) {
    return this.buffers.get(name);
  }

  getTexture(name: string) {
    return this.textures.get(name);
  }

  addNode(desc: ComputeNodeDesc): boolean {
    try {
      const module = this.device.createShaderModule({
        label: `${desc.label}-module`,
        code: desc.code,
      });

      const pipeline = this.device.createComputePipeline({
        label: desc.label,
        layout: "auto",
        compute: { module, entryPoint: desc.entryPoint },
      });

      const entries: GPUBindGroupEntry[] = desc.bindings.map((b) => {
        const buffer = this.buffers.get(b.resource);
        const texture = this.textures.get(b.resource);
        if (buffer) {
          return { binding: b.binding, resource: { buffer } };
        }
        if (texture) {
          return { binding: b.binding, resource: texture.createView() };
        }
        throw new Error(`Resource not found: ${b.resource}`);
      });

      const bindGroup = this.device.createBindGroup({
        label: `${desc.label}-bg`,
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });

      this.nodes.push({ desc, pipeline, bindGroup });
      return true;
    } catch (e) {
      console.error(`[ComputeGraph] Failed to add node "${desc.label}":`, e);
      return false;
    }
  }

  removeNode(label: string) {
    this.nodes = this.nodes.filter((n) => n.desc.label !== label);
  }

  replaceNodeCode(label: string, code: string): boolean {
    const idx = this.nodes.findIndex((n) => n.desc.label === label);
    if (idx === -1) return false;

    const old = this.nodes[idx];
    const newDesc = { ...old.desc, code };
    this.nodes.splice(idx, 1);
    return this.addNode(newDesc);
  }

  execute(encoder: GPUCommandEncoder, filter?: (node: ComputeNode) => boolean) {
    const pass = encoder.beginComputePass();
    for (const node of this.nodes) {
      if (filter && !filter(node)) continue;
      pass.setPipeline(node.pipeline);
      pass.setBindGroup(0, node.bindGroup);
      pass.dispatchWorkgroups(...node.desc.workgroups);
    }
    pass.end();
  }

  get nodeCount() {
    return this.nodes.length;
  }

  getNodes(): ReadonlyArray<ComputeNode> {
    return this.nodes;
  }

  destroy() {
    this.nodes = [];
    this.buffers.clear();
    this.textures.clear();
  }
}
