import type { IKChain } from "../scene/ik-solver";

export class IKDebugRenderer {
  private device: GPUDevice;
  private screenFormat: GPUTextureFormat;
  private pointPipeline: GPURenderPipeline | null = null;
  private linePipeline: GPURenderPipeline | null = null;
  private pointBuf: GPUBuffer | null = null;
  private lineBuf: GPUBuffer | null = null;
  private ubo: GPUBuffer | null = null;

  constructor(device: GPUDevice, screenFormat: GPUTextureFormat) {
    this.device = device;
    this.screenFormat = screenFormat;
  }

  draw(
    encoder: GPUCommandEncoder,
    screenView: GPUTextureView,
    depthView: GPUTextureView,
    viewProj: Float32Array,
    worldMatrices: Float32Array,
    parentIndices: Int16Array | Int32Array,
    boneCount: number,
    ikChains: IKChain[],
    screenW: number,
    screenH: number,
  ): void {
    if (!this.pointPipeline) {
      const code = `
struct U { viewProj: mat4x4<f32>, screenSize: vec2<f32>, pad: vec2<f32> };
@group(0) @binding(0) var<uniform> u: U;
struct V { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, @location(2) size: f32 };
struct O { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> };
@vertex fn vs(v: V) -> O {
  let clip = u.viewProj * vec4<f32>(v.pos, 1.0);
  var o: O;
  o.position = clip;
  o.color = v.color;
  return o;
}
@fragment fn fs(o: O) -> @location(0) vec4<f32> {
  return vec4<f32>(o.color, 1.0);
}`;
      const mod = this.device.createShaderModule({ code });
      const vbLayout: GPUVertexBufferLayout = { arrayStride: 28, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32" },
      ]};
      const dsState: GPUDepthStencilState = { format: "depth24plus-stencil8", depthWriteEnabled: false, depthCompare: "less" };
      const sharedBGL = this.device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ]});
      const sharedLayout = this.device.createPipelineLayout({ bindGroupLayouts: [sharedBGL] });
      this.pointPipeline = this.device.createRenderPipeline({
        label: "ik-debug-points",
        layout: sharedLayout,
        vertex: { module: mod, entryPoint: "vs", buffers: [vbLayout] },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: this.screenFormat }] },
        primitive: { topology: "point-list" },
        depthStencil: dsState,
      });
      this.linePipeline = this.device.createRenderPipeline({
        label: "ik-debug-lines",
        layout: sharedLayout,
        vertex: { module: mod, entryPoint: "vs", buffers: [vbLayout] },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: this.screenFormat }] },
        primitive: { topology: "line-list" },
        depthStencil: dsState,
      });
    }

    const wm = worldMatrices;

    const lines: number[] = [];
    for (let i = 0; i < boneCount; i++) {
      const pi = parentIndices[i];
      if (pi < 0) continue;
      const cOff = i * 16, pOff = pi * 16;
      lines.push(wm[pOff+12], wm[pOff+13], -wm[pOff+14], 0, 0.8, 0.8, 1);
      lines.push(wm[cOff+12], wm[cOff+13], -wm[cOff+14], 0, 0.8, 0.8, 1);
    }

    const pts: number[] = [];
    for (const chain of ikChains) {
      const tOff = chain.targetIndex * 16;
      pts.push(wm[tOff+12], wm[tOff+13], -wm[tOff+14], 1, 0, 0, 8);
      const eOff = chain.effectorIndex * 16;
      pts.push(wm[eOff+12], wm[eOff+13], -wm[eOff+14], 0, 1, 0, 8);
      for (const link of chain.links) {
        const lOff = link.index * 16;
        pts.push(wm[lOff+12], wm[lOff+13], -wm[lOff+14], 1, 1, 0, 5);
      }
    }

    const lineVertCount = lines.length / 7;
    const lineByteSize = lineVertCount * 28;
    if (!this.lineBuf || this.lineBuf.size < lineByteSize) {
      this.lineBuf?.destroy();
      this.lineBuf = this.device.createBuffer({ label: "bone-line-vb", size: Math.max(lineByteSize, 1024), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.lineBuf, 0, new Float32Array(lines) as unknown as GPUAllowSharedBufferSource);

    const ptVertCount = pts.length / 7;
    const ptByteSize = ptVertCount * 28;
    if (!this.pointBuf || this.pointBuf.size < ptByteSize) {
      this.pointBuf?.destroy();
      this.pointBuf = this.device.createBuffer({ label: "ik-debug-vb", size: Math.max(ptByteSize, 256), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.pointBuf, 0, new Float32Array(pts) as unknown as GPUAllowSharedBufferSource);

    const uboData = new Float32Array(20);
    uboData.set(viewProj as unknown as ArrayLike<number>, 0);
    uboData[16] = screenW; uboData[17] = screenH;

    if (!this.ubo) {
      this.ubo = this.device.createBuffer({ label: "ik-debug-ubo", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this.ubo, 0, uboData as unknown as GPUAllowSharedBufferSource);

    const bg = this.device.createBindGroup({
      layout: this.pointPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.ubo } }],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: screenView, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store", depthClearValue: 1.0, stencilLoadOp: "load", stencilStoreOp: "store", stencilClearValue: 0 },
    });
    pass.setPipeline(this.linePipeline!);
    pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, this.lineBuf);
    pass.draw(lineVertCount);
    pass.setPipeline(this.pointPipeline);
    pass.setVertexBuffer(0, this.pointBuf);
    pass.draw(ptVertCount);
    pass.end();
  }

  destroy(): void {
    this.pointBuf?.destroy();
    this.lineBuf?.destroy();
    this.ubo?.destroy();
  }
}