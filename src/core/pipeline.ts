export interface PipelineDescriptor {
  label: string;
  shader: string;
  topology?: GPUPrimitiveTopology;
  cullMode?: GPUCullMode;
  depthWrite?: boolean;
  depthCompare?: GPUCompareFunction;
  blend?: GPUBlendState;
  vertexBuffers?: GPUVertexBufferLayout[];
  bindGroupLayouts?: GPUBindGroupLayoutEntry[][];
}

export class PipelineManager {
  private device: GPUDevice;
  private pipelines = new Map<string, GPURenderPipeline>();
  private computePipelines = new Map<string, GPUComputePipeline>();
  private bindGroupLayouts = new Map<string, GPUBindGroupLayout[]>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  createRenderPipeline(desc: PipelineDescriptor): GPURenderPipeline {
    const layouts = (desc.bindGroupLayouts ?? []).map((entries, i) =>
      this.device.createBindGroupLayout({
        label: `${desc.label}-bgl-${i}`,
        entries,
      })
    );
    this.bindGroupLayouts.set(desc.label, layouts);

    const module = this.device.createShaderModule({
      label: `${desc.label}-shader`,
      code: desc.shader,
    });

    const pipeline = this.device.createRenderPipeline({
      label: desc.label,
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: layouts,
      }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: desc.vertexBuffers ?? [],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [
          {
            format: navigator.gpu.getPreferredCanvasFormat(),
            blend: desc.blend ?? undefined,
          },
        ],
      },
      primitive: {
        topology: desc.topology ?? "triangle-list",
        cullMode: desc.cullMode ?? "back",
      },
      depthStencil: desc.depthWrite !== undefined
        ? {
            format: "depth24plus",
            depthWriteEnabled: desc.depthWrite,
            depthCompare: desc.depthCompare ?? "less",
          }
        : undefined,
    });

    this.pipelines.set(desc.label, pipeline);
    return pipeline;
  }

  createComputePipeline(label: string, code: string): GPUComputePipeline {
    const pipeline = this.device.createComputePipeline({
      label,
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code }),
        entryPoint: "cs_main",
      },
    });
    this.computePipelines.set(label, pipeline);
    return pipeline;
  }

  getRenderPipeline(label: string): GPURenderPipeline | undefined {
    return this.pipelines.get(label);
  }

  getComputePipeline(label: string): GPUComputePipeline | undefined {
    return this.computePipelines.get(label);
  }

  getBindGroupLayouts(label: string): GPUBindGroupLayout[] | undefined {
    return this.bindGroupLayouts.get(label);
  }

  hotReloadRenderPipeline(label: string, desc: PipelineDescriptor): boolean {
    try {
      const pipeline = this.createRenderPipeline(desc);
      this.pipelines.set(label, pipeline);
      return true;
    } catch (e) {
      console.error(`Pipeline hot-reload failed [${label}]:`, e);
      return false;
    }
  }
}
