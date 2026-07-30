export interface MorphDesc {
  name: string;
  deltaPositions: Float32Array;
  deltaNormals?: Float32Array;
  vertexCount: number;
}

export class MorphTarget {
  morphCount: number;
  morphNames: string[];
  vertexCount: number;
  weights: Float32Array;

  deltaPositions: Float32Array[];
  deltaNormals: Float32Array[];

  baseVertexBuffer: GPUBuffer | null = null;
  morphedVertexBuffer: GPUBuffer | null = null;
  morphComputePipeline: GPUComputePipeline | null = null;
  morphBindGroup: GPUBindGroup | null = null;

  modified = true;

  constructor(descs: MorphDesc[]) {
    this.morphCount = descs.length;
    this.morphNames = descs.map((d) => d.name);
    this.vertexCount = descs.length > 0 ? descs[0].vertexCount : 0;
    this.weights = new Float32Array(this.morphCount);

    this.deltaPositions = descs.map((d) => d.deltaPositions);
    this.deltaNormals = descs.map((d) => d.deltaNormals ?? new Float32Array(0));
  }

  getMorphIndex(name: string): number {
    return this.morphNames.indexOf(name);
  }

  setWeight(index: number, value: number): void {
    if (index >= 0 && index < this.morphCount && this.weights[index] !== value) {
      this.weights[index] = value;
      this.modified = true;
    }
  }

  applyWeights(basePositions: Float32Array, stride: number, out: Float32Array): void {
    out.set(basePositions);

    for (let m = 0; m < this.morphCount; m++) {
      const w = this.weights[m];
      if (Math.abs(w) < 1e-6) continue;

      const deltas = this.deltaPositions[m];
      const vertCount = this.vertexCount;

      for (let v = 0; v < vertCount; v++) {
        const outOffset = v * stride;
        const deltaOffset = v * 3;
        out[outOffset] += deltas[deltaOffset] * w;
        out[outOffset + 1] += deltas[deltaOffset + 1] * w;
        out[outOffset + 2] += deltas[deltaOffset + 2] * w;
      }
    }

    this.modified = false;
  }

  destroy(): void {
    this.baseVertexBuffer?.destroy();
    this.morphedVertexBuffer?.destroy();
    this.baseVertexBuffer = null;
    this.morphedVertexBuffer = null;
    this.morphComputePipeline = null;
    this.morphBindGroup = null;
  }
}