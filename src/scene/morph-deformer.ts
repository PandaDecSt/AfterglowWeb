import type { PMXMorph } from "../utils/pmx-loader";

const VERTEX_STRIDE = 14;

export class MorphDeformer {
  private baseVertices: Float32Array;
  private morphedVertices: Float32Array;
  private morphs: { pmxIndex: number; offsets: { vertexIndex: number; position: Float32Array }[] }[];
  private vertexBuffer: GPUBuffer;
  private device: GPUDevice;

  constructor(device: GPUDevice, vertexBuffer: GPUBuffer, baseVertices: Float32Array, pmxMorphs: PMXMorph[]) {
    this.device = device;
    this.vertexBuffer = vertexBuffer;
    this.baseVertices = new Float32Array(baseVertices);
    this.morphedVertices = new Float32Array(baseVertices.length);
    this.morphedVertices.set(this.baseVertices);
    this.morphs = [];
    for (let i = 0; i < pmxMorphs.length; i++) {
      if (pmxMorphs[i].type === 1) {
        this.morphs.push({ pmxIndex: i, offsets: pmxMorphs[i].offsets });
      }
    }
  }

  get morphCount(): number { return this.morphs.length; }

  apply(weights: Float32Array): void {
    const hasMorph = this.morphs.length > 0 && weights.some(w => w !== 0);
    if (!hasMorph) return;


    this.morphedVertices.set(this.baseVertices);
    for (let mi = 0; mi < this.morphs.length; mi++) {
      const w = weights[this.morphs[mi].pmxIndex];
      if (w === 0) continue;
      const offsets = this.morphs[mi].offsets;
      for (let j = 0; j < offsets.length; j++) {
        const vi = offsets[j].vertexIndex;
        const pos = offsets[j].position;
        const o = vi * VERTEX_STRIDE;
        this.morphedVertices[o] += pos[0] * w;
        this.morphedVertices[o + 1] += pos[1] * w;
        this.morphedVertices[o + 2] += pos[2] * w;
      }
    }
    this.device.queue.writeBuffer(this.vertexBuffer, 0, this.morphedVertices as unknown as GPUAllowSharedBufferSource);
  }

  destroy(): void {
    this.baseVertices = null!;
    this.morphedVertices = null!;
    this.morphs = [];
  }
}