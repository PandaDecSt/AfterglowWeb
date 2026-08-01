export const SKIN_COMPUTE_WGSL = `
struct Params {
  vertexCount: u32,
  boneCount: u32,
  stride: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> srcVertices: array<f32>;
@group(0) @binding(2) var<storage, read_write> dstVertices: array<f32>;
@group(0) @binding(3) var<storage, read> skinMatrices: array<mat4x4<f32>>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vi = gid.x;
  if ((vi >= params.vertexCount)) { return; }

  let stride = params.stride;
  let srcOff = vi * stride;
  let dstOff = vi * stride;

  let px = srcVertices[srcOff + 0u];
  let py = srcVertices[srcOff + 1u];
  let pz = srcVertices[srcOff + 2u];
  let nx = srcVertices[srcOff + 3u];
  let ny = srcVertices[srcOff + 4u];
  let nz = srcVertices[srcOff + 5u];

  let j0 = u32(srcVertices[srcOff + 8u]);
  let j1 = u32(srcVertices[srcOff + 10u]);
  let j2 = u32(srcVertices[srcOff + 12u]);
  let j3 = u32(srcVertices[srcOff + 14u]);
  let w0 = srcVertices[srcOff + 20u];
  let w1 = srcVertices[srcOff + 24u];
  let w2 = srcVertices[srcOff + 28u];
  let w3 = srcVertices[srcOff + 32u];

  let weightSum = w0 + w1 + w2 + w3;
  let invW = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let w = select(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(w0, w1, w2, w3) * invW, weightSum > 0.0001);

  var skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var skinNrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let pos4 = vec4<f32>(px, py, pz, 1.0);
  let nrm4 = vec4<f32>(nx, ny, nz, 0.0);

  let jArr = array<u32, 4>(j0, j1, j2, j3);
  let wArr = array<f32, 4>(w[0], w[1], w[2], w[3]);
  for (var i = 0u; i < 4u; i++) {
    let j = jArr[i];
    let m = skinMatrices[j];
    skinPos += m * pos4 * wArr[i];
    skinNrm += m * nrm4 * wArr[i];
  }

  dstVertices[dstOff + 0u] = skinPos.x;
  dstVertices[dstOff + 1u] = skinPos.y;
  dstVertices[dstOff + 2u] = skinPos.z;
  dstVertices[dstOff + 3u] = skinNrm.x;
  dstVertices[dstOff + 4u] = skinNrm.y;
  dstVertices[dstOff + 5u] = skinNrm.z;

  for (var k = 6u; k < stride; k++) {
    dstVertices[dstOff + k] = srcVertices[srcOff + k];
  }
}
`;

export class GPUComputeSkinning {
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private paramsBuffer: GPUBuffer;
  private srcBuffer: GPUBuffer | null = null;
  private dstBuffer: GPUBuffer | null = null;
  private skinBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private vertexCount = 0;
  private boneCount = 0;
  private stride = 0;
  private srcData: Float32Array | null = null;

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: SKIN_COMPUTE_WGSL });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] });
    this.pipeline = device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint: "cs_main" },
    });

    this.paramsBuffer = device.createBuffer({
      label: "gpu-skin-params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  setup(
    vertexCount: number,
    boneCount: number,
    stride: number,
    srcVertexData: Float32Array,
    skinMatrixData: Float32Array,
  ): void {
    this.vertexCount = vertexCount;
    this.boneCount = boneCount;
    this.stride = stride;
    this.srcData = srcVertexData;

    const paramsData = new Uint32Array([vertexCount, boneCount, stride, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    this.srcBuffer?.destroy();
    this.dstBuffer?.destroy();
    this.skinBuffer?.destroy();

    const srcSize = vertexCount * stride * 4;
    const dstSize = srcSize;
    const skinSize = boneCount * 16 * 4;

    this.srcBuffer = this.device.createBuffer({
      label: "gpu-skin-src",
      size: srcSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.dstBuffer = this.device.createBuffer({
      label: "gpu-skin-dst",
      size: dstSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
    });
    this.skinBuffer = this.device.createBuffer({
      label: "gpu-skin-matrices",
      size: skinSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(this.srcBuffer, 0, srcVertexData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.skinBuffer, 0, skinMatrixData as unknown as GPUAllowSharedBufferSource);

    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: this.srcBuffer } },
        { binding: 2, resource: { buffer: this.dstBuffer } },
        { binding: 3, resource: { buffer: this.skinBuffer } },
      ],
    });
  }

  updateSkinMatrices(skinMatrixData: Float32Array): void {
    if (!this.skinBuffer) return;
    this.device.queue.writeBuffer(this.skinBuffer, 0, skinMatrixData as unknown as GPUAllowSharedBufferSource);
  }

  getSkinnedVertexBuffer(): GPUBuffer | null {
    return this.dstBuffer;
  }

  getStride(): number {
    return this.stride;
  }

  dispatch(encoder: GPUCommandEncoder): void {
    if (!this.bindGroup || this.vertexCount === 0) return;
    const pass = encoder.beginComputePass({ label: "gpu-skinning" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    const workgroupCount = Math.ceil(this.vertexCount / 64);
    pass.dispatchWorkgroups(workgroupCount);
    pass.end();
  }

  destroy(): void {
    this.paramsBuffer.destroy();
    this.srcBuffer?.destroy();
    this.dstBuffer?.destroy();
    this.skinBuffer?.destroy();
    this.srcBuffer = null;
    this.dstBuffer = null;
    this.skinBuffer = null;
    this.bindGroup = null;
  }
}