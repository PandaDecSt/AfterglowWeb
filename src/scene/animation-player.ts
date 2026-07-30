import type { Skeleton } from "./skeleton";
import type { AnimationClip, BoneTrack, MorphTrack } from "./animation-clip";
import { sampleTrack } from "./animation-clip";
import { quat } from "wgpu-matrix";

export interface AnimationSlot {
  clip: AnimationClip;
  time: number;
  playing: boolean;
  loop: boolean;
  weight: number;
  speed: number;
  priority: number;
  blendIn: number;
  blendOut: number;
  blendWeight: number;
}

export class AnimationPlayer {
  private skeleton: Skeleton;
  private slots: AnimationSlot[] = [];
  private morphWeights: Float32Array = new Float32Array(0);
  private morphCount = 0;

  constructor(skeleton: Skeleton, morphCount = 0) {
    this.skeleton = skeleton;
    this.morphCount = morphCount;
    this.morphWeights = new Float32Array(morphCount);
  }

  getMorphWeights(): Float32Array {
    return this.morphWeights;
  }

  play(
    clip: AnimationClip,
    options?: {
      loop?: boolean;
      weight?: number;
      speed?: number;
      priority?: number;
      blendIn?: number;
    }
  ): AnimationSlot {
    const slot: AnimationSlot = {
      clip,
      time: 0,
      playing: true,
      loop: options?.loop ?? true,
      weight: options?.weight ?? 1.0,
      speed: options?.speed ?? 1.0,
      priority: options?.priority ?? 0,
      blendIn: options?.blendIn ?? 0,
      blendOut: 0,
      blendWeight: options?.blendIn ? 0 : 1,
    };
    this.slots.push(slot);
    return slot;
  }

  stop(clipName?: string): void {
    if (clipName) {
      this.slots = this.slots.filter((s) => s.clip.name !== clipName);
    } else {
      this.slots = [];
    }
  }

  isPlaying(clipName?: string): boolean {
    if (clipName) return this.slots.some((s) => s.clip.name === clipName && s.playing);
    return this.slots.some((s) => s.playing);
  }

  update(deltaTime: number): void {
    this.morphWeights.fill(0);

    const sortedSlots = this.slots
      .filter((s) => s.playing)
      .sort((a, b) => b.priority - a.priority);

    for (const slot of sortedSlots) {
      slot.time += deltaTime * slot.speed;

      if (slot.loop) {
        slot.time = slot.time % slot.clip.duration;
      } else if (slot.time >= slot.clip.duration) {
        slot.time = slot.clip.duration;
        slot.playing = false;
      }

      if (slot.blendIn > 0 && slot.blendWeight < 1) {
        slot.blendWeight = Math.min(1, slot.blendWeight + deltaTime / slot.blendIn);
      }

      const effectiveWeight = slot.weight * slot.blendWeight;
      if (effectiveWeight <= 0) continue;

      this.applyBoneTracks(slot.clip.boneTracks, slot.time, effectiveWeight);
      this.applyMorphTracks(slot.clip.morphTracks, slot.time, effectiveWeight);
    }

    this.slots = this.slots.filter((s) => s.playing || s.blendWeight > 0);
  }

  private applyBoneTracks(tracks: BoneTrack[], time: number, weight: number): void {
    for (const track of tracks) {
      const boneIdx = track.boneIndex;
      if (boneIdx < 0 || boneIdx >= this.skeleton.boneCount) continue;

      switch (track.type) {
        case "rotation": {
          const rx = sampleTrack(track.keyframes, time);
          const ry = sampleTrack(track.keyframes, time);
          const rz = sampleTrack(track.keyframes, time);
          if (weight >= 0.999) {
            this.skeleton.setLocalRotation(boneIdx, rx, ry, rz, 0);
            this.normalizeBoneRotation(boneIdx);
          } else {
            this.blendRotation(boneIdx, rx, ry, rz, weight);
          }
          break;
        }
        case "position": {
          const x = sampleTrack(track.keyframes, time);
          const y = sampleTrack(track.keyframes, time);
          const z = sampleTrack(track.keyframes, time);
          if (weight >= 0.999) {
            this.skeleton.setLocalPosition(boneIdx, x, y, z);
          } else {
            this.blendPosition(boneIdx, x, y, z, weight);
          }
          break;
        }
        case "scale": {
          break;
        }
      }
    }
  }

  private applyMorphTracks(tracks: MorphTrack[], time: number, weight: number): void {
    for (const track of tracks) {
      const idx = track.morphIndex;
      if (idx < 0 || idx >= this.morphCount) continue;
      const value = sampleTrack(track.keyframes, time);
      this.morphWeights[idx] += value * weight;
    }
  }

  private normalizeBoneRotation(index: number): void {
    const i4 = index * 4;
    const x = this.skeleton.localRotations[i4];
    const y = this.skeleton.localRotations[i4 + 1];
    const z = this.skeleton.localRotations[i4 + 2];
    const w = this.skeleton.localRotations[i4 + 3];
    const len = Math.sqrt(x * x + y * y + z * z + w * w) || 1;
    this.skeleton.localRotations[i4] = x / len;
    this.skeleton.localRotations[i4 + 1] = y / len;
    this.skeleton.localRotations[i4 + 2] = z / len;
    this.skeleton.localRotations[i4 + 3] = w / len;
  }

  private blendPosition(index: number, x: number, y: number, z: number, weight: number): void {
    const i3 = index * 3;
    this.skeleton.localPositions[i3] += (x - this.skeleton.localPositions[i3]) * weight;
    this.skeleton.localPositions[i3 + 1] += (y - this.skeleton.localPositions[i3 + 1]) * weight;
    this.skeleton.localPositions[i3 + 2] += (z - this.skeleton.localPositions[i3 + 2]) * weight;
  }

  private blendRotation(index: number, x: number, y: number, z: number, weight: number): void {
    const i4 = index * 4;
    const curX = this.skeleton.localRotations[i4];
    const curY = this.skeleton.localRotations[i4 + 1];
    const curZ = this.skeleton.localRotations[i4 + 2];
    const curW = this.skeleton.localRotations[i4 + 3];

    const dot = curX * x + curY * y + curZ * z + curW * 0;
    const sign = dot >= 0 ? 1 : -1;

    const result = quat.slerp(
      quat.create(curX, curY, curZ, curW),
      quat.create(x * sign, y * sign, z * sign, 0),
      weight
    );

    this.skeleton.localRotations[i4] = result[0];
    this.skeleton.localRotations[i4 + 1] = result[1];
    this.skeleton.localRotations[i4 + 2] = result[2];
    this.skeleton.localRotations[i4 + 3] = result[3];
  }
}