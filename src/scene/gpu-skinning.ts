export const SKIN_COMPUTE_WGSL = `
struct Params {
  vertexCount: u32,
  boneCount: u32,
  stride: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> srcData: array<u32>;
@group(0) @binding(2) var<storage, read_write> dstData: array<u32>;
@group(0) @binding(3) var<storage, read> skinData: array<u32>;
@group(0) @binding(4) var<storage, read> boneIdxData: array<u32>;
@group(0) @binding(5) var<storage, read> boneWtData: array<u32>;

fn loadF32(off: u32) -> f32 { return bitcast<f32>(srcData[off]); }
fn storeF32(off: u32, v: f32) { dstData[off] = bitcast<u32>(v); }
fn skinF32(off: u32) -> f32 { return bitcast<f32>(skinData[off]); }

fn mat4MulVec4(boneIdx: u32, v: vec4<f32>) -> vec4<f32> {
  let b = boneIdx * 16u;
  let r0 = skinF32(b+0u)*v.x + skinF32(b+4u)*v.y + skinF32(b+8u)*v.z  + skinF32(b+12u)*v.w;
  let r1 = skinF32(b+1u)*v.x + skinF32(b+5u)*v.y + skinF32(b+9u)*v.z  + skinF32(b+13u)*v.w;
  let r2 = skinF32(b+2u)*v.x + skinF32(b+6u)*v.y + skinF32(b+10u)*v.z + skinF32(b+14u)*v.w;
  let r3 = skinF32(b+3u)*v.x + skinF32(b+7u)*v.y + skinF32(b+11u)*v.z + skinF32(b+15u)*v.w;
  return vec4<f32>(r0, r1, r2, r3);
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let vi = gid.x;
  if (vi >= params.vertexCount) { return; }

  let stride = params.stride;
  let srcOff = vi * stride;
  let dstOff = vi * stride;

  let px = loadF32(srcOff + 0u);
  let py = loadF32(srcOff + 1u);
  let pz = loadF32(srcOff + 2u);
  let nx = loadF32(srcOff + 3u);
  let ny = loadF32(srcOff + 4u);
  let nz = loadF32(srcOff + 5u);

  let vi4 = vi * 4u;
  let j0 = boneIdxData[vi4 + 0u];
  let j1 = boneIdxData[vi4 + 1u];
  let j2 = boneIdxData[vi4 + 2u];
  let j3 = boneIdxData[vi4 + 3u];

  let w0 = bitcast<f32>(boneWtData[vi4 + 0u]);
  let w1 = bitcast<f32>(boneWtData[vi4 + 1u]);
  let w2 = bitcast<f32>(boneWtData[vi4 + 2u]);
  let w3 = bitcast<f32>(boneWtData[vi4 + 3u]);
  let weightSum = w0 + w1 + w2 + w3;
  let invW = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let rw0 = select(1.0, w0 * invW, weightSum > 0.0001);
  let rw1 = select(0.0, w1 * invW, weightSum > 0.0001);
  let rw2 = select(0.0, w2 * invW, weightSum > 0.0001);
  let rw3 = select(0.0, w3 * invW, weightSum > 0.0001);

  let pos4 = vec4<f32>(px, py, pz, 1.0);
  let nrm4 = vec4<f32>(nx, ny, nz, 0.0);

  var skinPos = mat4MulVec4(j0, pos4) * rw0
              + mat4MulVec4(j1, pos4) * rw1
              + mat4MulVec4(j2, pos4) * rw2
              + mat4MulVec4(j3, pos4) * rw3;
  var skinNrm = mat4MulVec4(j0, nrm4) * rw0
              + mat4MulVec4(j1, nrm4) * rw1
              + mat4MulVec4(j2, nrm4) * rw2
              + mat4MulVec4(j3, nrm4) * rw3;

  storeF32(dstOff + 0u, skinPos.x);
  storeF32(dstOff + 1u, skinPos.y);
  storeF32(dstOff + 2u, skinPos.z);
  storeF32(dstOff + 3u, skinNrm.x);
  storeF32(dstOff + 4u, skinNrm.y);
  storeF32(dstOff + 5u, skinNrm.z);

  for (var k = 6u; k < stride; k++) {
    dstData[dstOff + k] = srcData[srcOff + k];
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
  private renderVB: GPUBuffer | null = null;
  private skinBuffer: GPUBuffer | null = null;
  private boneIdxBuffer: GPUBuffer | null = null;
  private boneWtBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private currentSrcBuffer: GPUBuffer | null = null;
  private vertexCount = 0;
  private boneCount = 0;
  private stride = 0;

  constructor(device: GPUDevice) {
    this.device = device;
    const module = device.createShaderModule({ code: SKIN_COMPUTE_WGSL });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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
    boneIndicesData: Uint32Array,
    boneWeightsData: Float32Array,
  ): void {
    this.vertexCount = vertexCount;
    this.boneCount = boneCount;
    this.stride = stride;

    const paramsData = new Uint32Array([vertexCount, boneCount, stride, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    this.srcBuffer?.destroy();
    this.dstBuffer?.destroy();
    this.renderVB?.destroy();
    this.skinBuffer?.destroy();
    this.boneIdxBuffer?.destroy();
    this.boneWtBuffer?.destroy();

    const srcSize = vertexCount * stride * 4;
    const skinSize = boneCount * 16 * 4;

    this.srcBuffer = this.device.createBuffer({
      label: "gpu-skin-src",
      size: srcSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.dstBuffer = this.device.createBuffer({
      label: "gpu-skin-dst",
      size: srcSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.renderVB = this.device.createBuffer({
      label: "gpu-skin-render-vb",
      size: srcSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.skinBuffer = this.device.createBuffer({
      label: "gpu-skin-matrices",
      size: skinSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.boneIdxBuffer = this.device.createBuffer({
      label: "gpu-skin-indices",
      size: boneIndicesData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.boneWtBuffer = this.device.createBuffer({
      label: "gpu-skin-weights",
      size: boneWeightsData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.device.queue.writeBuffer(this.srcBuffer, 0, srcVertexData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.skinBuffer, 0, skinMatrixData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.boneIdxBuffer, 0, boneIndicesData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.boneWtBuffer, 0, boneWeightsData as unknown as GPUAllowSharedBufferSource);
    this.currentSrcBuffer = this.srcBuffer;

    this.rebuildBindGroup(this.srcBuffer);
  }

  private rebuildBindGroup(srcBuf: GPUBuffer): void {
    if (!this.dstBuffer || !this.skinBuffer || !this.boneIdxBuffer || !this.boneWtBuffer) return;
    this.bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuffer } },
        { binding: 1, resource: { buffer: srcBuf } },
        { binding: 2, resource: { buffer: this.dstBuffer } },
        { binding: 3, resource: { buffer: this.skinBuffer } },
        { binding: 4, resource: { buffer: this.boneIdxBuffer } },
        { binding: 5, resource: { buffer: this.boneWtBuffer } },
      ],
    });
  }

  updateSkinMatrices(skinMatrixData: Float32Array): void {
    if (!this.skinBuffer) return;
    this.device.queue.writeBuffer(this.skinBuffer, 0, skinMatrixData as unknown as GPUAllowSharedBufferSource);
  }

  setSourceBuffer(buffer: GPUBuffer): void {
    if (buffer === this.currentSrcBuffer) return;
    this.currentSrcBuffer = buffer;
    this.rebuildBindGroup(buffer);
  }

  getSkinnedVertexBuffer(): GPUBuffer | null {
    return this.renderVB;
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
    if (this.dstBuffer && this.renderVB) {
      encoder.copyBufferToBuffer(this.dstBuffer, 0, this.renderVB, 0, this.dstBuffer.size);
    }
  }

  destroy(): void {
    this.paramsBuffer.destroy();
    this.srcBuffer?.destroy();
    this.dstBuffer?.destroy();
    this.renderVB?.destroy();
    this.skinBuffer?.destroy();
    this.boneIdxBuffer?.destroy();
    this.boneWtBuffer?.destroy();
    this.srcBuffer = null;
    this.dstBuffer = null;
    this.renderVB = null;
    this.skinBuffer = null;
    this.boneIdxBuffer = null;
    this.boneWtBuffer = null;
    this.bindGroup = null;
    this.currentSrcBuffer = null;
  }
}
