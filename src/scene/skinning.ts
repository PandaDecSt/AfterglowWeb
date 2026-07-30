export class Skinning {
  boneCount: number;
  joints: Uint16Array;
  weights: Float32Array;
  vertexCount: number;
  weightsPerVertex: number;

  skinMatrixBuffer: GPUBuffer | null = null;
  skinMatrixData: Float32Array;
  modified = true;

  constructor(
    vertexCount: number,
    weightsPerVertex: number,
    joints: Uint16Array,
    weights: Float32Array,
    boneCount: number
  ) {
    this.vertexCount = vertexCount;
    this.weightsPerVertex = weightsPerVertex;
    this.joints = joints;
    this.weights = weights;
    this.boneCount = boneCount;
    this.skinMatrixData = new Float32Array(boneCount * 16);
  }

  createGPUResources(device: GPUDevice, label = "skin-matrices"): GPUBuffer {
    this.skinMatrixBuffer = device.createBuffer({
      label,
      size: this.boneCount * 16 * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    return this.skinMatrixBuffer;
  }

  uploadSkinMatrices(skinMatrices: Float32Array): void {
    this.skinMatrixData.set(skinMatrices);
    this.modified = true;
  }

  flushToDevice(device: GPUDevice): void {
    if (!this.modified || !this.skinMatrixBuffer) return;
    device.queue.writeBuffer(
      this.skinMatrixBuffer,
      0,
      this.skinMatrixData as unknown as GPUAllowSharedBufferSource
    );
    this.modified = false;
  }

  destroy(): void {
    this.skinMatrixBuffer?.destroy();
    this.skinMatrixBuffer = null;
  }
}