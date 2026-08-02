export const MORPH_COMPUTE_WGSL = `
struct Params {
  vertexCount: u32,
  morphCount: u32,
  stride: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> baseData: array<u32>;
@group(0) @binding(2) var<storage, read_write> morphedData: array<u32>;
@group(0) @binding(3) var<storage, read> morphWeights: array<f32>;
@group(0) @binding(4) var<storage, read> morphDeltas: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vi = gid.x;
  if (vi >= params.vertexCount) { return; }

  let stride = params.stride;
  let baseOff = vi * stride;
  let morphOff = vi * stride;

  for (var k = 0u; k < stride; k++) {
    morphedData[morphOff + k] = baseData[baseOff + k];
  }

  for (var m = 0u; m < params.morphCount; m++) {
    let w = morphWeights[m];
    if (abs(w) < 1e-6) { continue; }

    let deltaOff = (m * params.vertexCount + vi) * 3u;
    morphedData[morphOff + 0u] = bitcast<u32>(bitcast<f32>(morphedData[morphOff + 0u]) + morphDeltas[deltaOff + 0u] * w);
    morphedData[morphOff + 1u] = bitcast<u32>(bitcast<f32>(morphedData[morphOff + 1u]) + morphDeltas[deltaOff + 1u] * w);
    morphedData[morphOff + 2u] = bitcast<u32>(bitcast<f32>(morphedData[morphOff + 2u]) + morphDeltas[deltaOff + 2u] * w);
  }
}
`;

export class GPUComputeMorph {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private paramsBuffer: GPUBuffer;
  private baseBuffer: GPUBuffer | null = null;
  private morphedBuffer: GPUBuffer | null = null;
  private weightsBuffer: GPUBuffer | null = null;
  private deltasBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private vertexCount = 0;
  private morphCount = 0;
  private stride = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: MORPH_COMPUTE_WGSL });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "cs_main" },
    });

    this.paramsBuffer = device.createBuffer({
      label: "gpu-morph-params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setup(
    vertexCount: number,
    morphCount: number,
    stride: number,
    baseVertexData: Float32Array,
    morphDeltasData: Float32Array,
  ): void {
    this.vertexCount = vertexCount;
    this.morphCount = morphCount;
    this.stride = stride;

    const paramsData = new Uint32Array([vertexCount, morphCount, stride, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    this.baseBuffer?.destroy();
    this.morphedBuffer?.destroy();
    this.weightsBuffer?.destroy();
    this.deltasBuffer?.destroy();

    const vertexSize = vertexCount * stride * 4;
    const weightsSize = morphCount * 4;
    const deltasSize = morphDeltasData.byteLength;

    this.baseBuffer = this.device.createBuffer({
      label: "gpu-morph-base",
      size: vertexSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.morphedBuffer = this.device.createBuffer({
      label: "gpu-morph-output",
      size: vertexSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
    });
    this.weightsBuffer = this.device.createBuffer({
      label: "gpu-morph-weights",
      size: Math.max(weightsSize, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.deltasBuffer = this.device.createBuffer({
      label: "gpu-morph-deltas",
      size: deltasSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(this.baseBuffer, 0, baseVertexData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.deltasBuffer, 0, morphDeltasData as unknown as GPUAllowSharedBufferSource);

    this.rebuildBindGroup();
  }

  private rebuildBindGroup(): void {
    if (!this.baseBuffer || !this.morphedBuffer || !this.weightsBuffer || !this.deltasBuffer) return;
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.baseBuffer } },
        { binding: 2, resource: { buffer: this.morphedBuffer } },
        { binding: 3, resource: { buffer: this.weightsBuffer } },
        { binding: 4, resource: { buffer: this.deltasBuffer } },
      ],
    });
  }

  updateWeights(weights: Float32Array): void {
    if (!this.weightsBuffer) return;
    this.device.queue.writeBuffer(this.weightsBuffer, 0, weights as unknown as GPUAllowSharedBufferSource);
  }

  getMorphedVertexBuffer(): GPUBuffer | null {
    return this.morphedBuffer;
  }

  dispatch(encoder: GPUCommandEncoder): void {
    if (!this.bindGroup || this.vertexCount === 0) return;
    const pass = encoder.beginComputePass({ label: "gpu-morph" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const workgroupCount = Math.ceil(this.vertexCount / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  }

  destroy(): void {
    this.paramsBuffer.destroy();
    this.baseBuffer?.destroy();
    this.morphedBuffer?.destroy();
    this.weightsBuffer?.destroy();
    this.deltasBuffer?.destroy();
    this.baseBuffer = null;
    this.morphedBuffer = null;
    this.weightsBuffer = null;
    this.deltasBuffer = null;
    this.bindGroup = null;
  }
}
