import type { Skeleton } from "./skeleton";
import type { VMDData, VMDBoneFrame, VMDMorphFrame } from "../utils/vmd-loader";
import { sampleVMDBone, sampleVMDMorph, vmdDuration } from "../utils/vmd-loader";

const FRAME_RATE = 30.0;

function normalizeBoneName(name: string): string {
  let out = "";
  for (let i = 0; i < name.length; i++) {
    const c = name.charCodeAt(i);
    if (c >= 0xff01 && c <= 0xff5e) {
      out += String.fromCharCode(c - 0xfee0);
    } else if (c === 0x3000) {
      out += " ";
    } else {
      out += name[i];
    }
  }
  return out;
}

export interface VMDAnimationSlot {
  data: VMDData;
  time: number;
  playing: boolean;
  loop: boolean;
  speed: number;
  boneMap: Map<string, number>;
  morphMap: Map<string, number>;
}

export class AnimationPlayer {
  private skeleton: Skeleton;
  private slots: VMDAnimationSlot[] = [];
  private morphWeights: Float32Array;
  private morphCount: number;
  private bindPositions: Float32Array;
  private bindRotations: Float32Array;

  constructor(skeleton: Skeleton, morphCount = 0) {
    this.skeleton = skeleton;
    this.morphCount = morphCount;
    this.morphWeights = new Float32Array(morphCount);
    this.bindPositions = new Float32Array(skeleton.localPositions);
    this.bindRotations = new Float32Array(skeleton.localRotations);
  }

  getMorphWeights(): Float32Array {
    return this.morphWeights;
  }

  get currentTime(): number {
    return this.slots.length > 0 ? this.slots[0].time : 0;
  }

  playVMD(
    data: VMDData,
    boneNames: string[],
    morphNames: string[],
    options?: { loop?: boolean; speed?: number }
  ): VMDAnimationSlot {
    const boneMap = new Map<string, number>();
    for (let i = 0; i < boneNames.length; i++) {
      boneMap.set(normalizeBoneName(boneNames[i]), i);
    }
    const morphMap = new Map<string, number>();
    for (let i = 0; i < morphNames.length; i++) {
      morphMap.set(normalizeBoneName(morphNames[i]), i);
    }

    const slot: VMDAnimationSlot = {
      data,
      time: 0,
      playing: true,
      loop: options?.loop ?? true,
      speed: options?.speed ?? 1.0,
      boneMap,
      morphMap,
    };
    this.slots.push(slot);
    return slot;
  }

  stop(): void {
    this.slots = [];
  }

  isPlaying(): boolean {
    return this.slots.some(s => s.playing);
  }

  update(deltaTime: number): void {
    if (this.slots.length === 0) return;

    this.skeleton.localPositions.set(this.bindPositions);
    this.skeleton.localRotations.set(this.bindRotations);
    this.morphWeights.fill(0);

    for (const slot of this.slots) {
      if (!slot.playing) continue;

      slot.time += deltaTime * slot.speed;

      const duration = vmdDuration(slot.data);
      if (duration <= 0) continue;

      if (slot.loop) {
        slot.time = slot.time % duration;
      } else if (slot.time >= duration) {
        slot.time = duration;
        slot.playing = false;
      }

      const frameNum = slot.time * FRAME_RATE;
      this.applyVMD(slot, frameNum);
    }

    this.slots = this.slots.filter(s => s.playing);
  }

  private _unmatchedLogged = false;

  private applyVMD(slot: VMDAnimationSlot, frameNum: number): void {
    if (!this._unmatchedLogged) {
      this._unmatchedLogged = true;
      const unmatched: string[] = [];
      for (const boneName of slot.data.boneFrames.keys()) {
        if (!slot.boneMap.has(normalizeBoneName(boneName))) unmatched.push(boneName);
      }
      if (unmatched.length > 0) {
        console.warn(`[AnimPlayer] ${unmatched.length} VMD bones unmatched:`, unmatched.slice(0, 20).join(", "));
        console.log("[AnimPlayer] PMX bones:", [...slot.boneMap.keys()].slice(0, 30).join(", "));
      }
    }

    for (const [boneName, frames] of slot.data.boneFrames) {
      const boneIdx = slot.boneMap.get(normalizeBoneName(boneName));
      if (boneIdx === undefined || boneIdx < 0 || boneIdx >= this.skeleton.boneCount) continue;

      const { rotation, translation } = sampleVMDBone(frames as VMDBoneFrame[], frameNum);

      this.skeleton.setLocalRotation(boneIdx, rotation[0], rotation[1], rotation[2], rotation[3]);

      const i3 = boneIdx * 3;
      this.skeleton.localPositions[i3] = this.bindPositions[i3] + translation[0];
      this.skeleton.localPositions[i3 + 1] = this.bindPositions[i3 + 1] + translation[1];
      this.skeleton.localPositions[i3 + 2] = this.bindPositions[i3 + 2] + translation[2];
    }

    for (const [morphName, frames] of slot.data.morphFrames) {
      const morphIdx = slot.morphMap.get(normalizeBoneName(morphName));
      if (morphIdx === undefined || morphIdx < 0 || morphIdx >= this.morphCount) continue;

      const value = sampleVMDMorph(frames as VMDMorphFrame[], frameNum);
      this.morphWeights[morphIdx] += value;
    }
  }
}
