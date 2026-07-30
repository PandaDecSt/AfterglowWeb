import type { Skeleton } from "./skeleton";
import type { Skinning } from "./skinning";
import type { AnimationPlayer } from "./animation-player";
import type { MorphTarget } from "./morph-target";
import type { Component } from "./entity";

export class SkinnedMesh implements Component {
  skeleton: Skeleton;
  skinning: Skinning;
  animation: AnimationPlayer | null = null;
  morph: MorphTarget | null = null;

  vertexBuffer!: GPUBuffer;
  indexBuffer!: GPUBuffer;
  indexCount = 0;
  indexFormat: GPUIndexFormat = "uint16";
  pipeline!: GPURenderPipeline;
  bindGroup!: GPUBindGroup;

  constructor(skeleton: Skeleton, skinning: Skinning) {
    this.skeleton = skeleton;
    this.skinning = skinning;
  }

  update(dt: number, _time: number): void {
    if (this.animation) {
      this.animation.update(dt);

      if (this.morph) {
        const morphWeights = this.animation.getMorphWeights();
        for (let i = 0; i < morphWeights.length; i++) {
          this.morph.setWeight(i, morphWeights[i]);
        }
      }
    }

    this.skeleton.updateWorldMatrices();

    this.skeleton.computeSkinMatrices(this.skinning.skinMatrixData);
    this.skinning.modified = true;
  }

  flushToDevice(device: GPUDevice): void {
    this.skinning.flushToDevice(device);
  }

  destroy(): void {
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.skinning.destroy();
    this.morph?.destroy();
  }
}