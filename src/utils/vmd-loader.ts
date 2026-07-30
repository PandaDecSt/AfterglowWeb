import { AnimationClip } from "../scene/animation-clip";
import type { BoneTrack, MorphTrack, Keyframe } from "../scene/animation-clip";

export interface VMDCameraKeyframe {
  time: number;
  position: Float32Array;
  rotation: Float32Array;
  distance: number;
  fov: number;
}

export class VMDData {
  name: string;
  boneTracks: Map<string, Keyframe[]> = new Map();
  morphTracks: Map<string, Keyframe[]> = new Map();
  cameraKeyframes: VMDCameraKeyframe[] = [];

  constructor(name = "") {
    this.name = name;
  }

  toAnimationClip(boneNameToIndex: Map<string, number>, morphNameToIndex: Map<string, number>): AnimationClip {
    const boneTracks: BoneTrack[] = [];
    const morphTracks: MorphTrack[] = [];
    let maxTime = 0;

    for (const [boneName, keyframes] of this.boneTracks) {
      const boneIndex = boneNameToIndex.get(boneName);
      if (boneIndex === undefined) continue;

      const sortedKf = keyframes.slice().sort((a, b) => a.time - b.time);

      const posTrack: Keyframe[] = [];
      const rotTrack: Keyframe[] = [];

      for (const kf of sortedKf) {
        if (kf.time > maxTime) maxTime = kf.time;
        posTrack.push({ time: kf.time, value: kf.value });
        rotTrack.push({ time: kf.time, value: kf.value });
      }

      if (posTrack.length > 0) {
        boneTracks.push({ boneName, boneIndex, type: "position", keyframes: posTrack });
      }
      if (rotTrack.length > 0) {
        boneTracks.push({ boneName, boneIndex, type: "rotation", keyframes: rotTrack });
      }
    }

    for (const [morphName, keyframes] of this.morphTracks) {
      const morphIndex = morphNameToIndex.get(morphName);
      if (morphIndex === undefined) continue;

      const sortedKf = keyframes.slice().sort((a, b) => a.time - b.time);
      for (const kf of sortedKf) {
        if (kf.time > maxTime) maxTime = kf.time;
      }

      if (sortedKf.length > 0) {
        morphTracks.push({ morphName, morphIndex, keyframes: sortedKf });
      }
    }

    return new AnimationClip(this.name, maxTime, boneTracks, morphTracks);
  }
}

export function parseVMD(buffer: ArrayBuffer): VMDData {
  const view = new DataView(buffer);
  let offset = 0;

  function readString(len: number, isUTF16: boolean): string {
    const bytes = new Uint8Array(buffer, offset, len);
    offset += len;
    const decoder = isUTF16 ? new TextDecoder("utf-16le") : new TextDecoder("shift-jis");
    return decoder.decode(bytes).replace(/\0+$/, "");
  }

  function readFloat32(): number {
    const v = view.getFloat32(offset, true);
    offset += 4;
    return v;
  }

  function readUint32(): number {
    const v = view.getUint32(offset, true);
    offset += 4;
    return v;
  }

  const magic = readString(30, false);
  const name = readString(20, true);

  const vmd = new VMDData(name);

  const motionCount = readUint32();
  for (let i = 0; i < motionCount; i++) {
    const boneName = readString(15, true);
    const frameNumber = readUint32();
    const px = readFloat32();
    const py = readFloat32();
    const pz = readFloat32();
    const rx = readFloat32();
    const ry = readFloat32();
    const rz = readFloat32();
    const rw = readFloat32();

    offset += 64; // interpolation bytes

    const time = frameNumber / 30;

    if (!vmd.boneTracks.has(boneName)) {
      vmd.boneTracks.set(boneName, []);
    }
    vmd.boneTracks.get(boneName)!.push({ time, value: px });
  }

  const morphCount = readUint32();
  for (let i = 0; i < morphCount; i++) {
    const morphName = readString(15, true);
    const frameNumber = readUint32();
    const weight = readFloat32();

    const time = frameNumber / 30;

    if (!vmd.morphTracks.has(morphName)) {
      vmd.morphTracks.set(morphName, []);
    }
    vmd.morphTracks.get(morphName)!.push({ time, value: weight });
  }

  if (offset < buffer.byteLength - 4) {
    try {
      const cameraCount = readUint32();
      for (let i = 0; i < cameraCount && offset < buffer.byteLength - 40; i++) {
        const frameNumber = readUint32();
        const distance = readFloat32();
        const cx = readFloat32();
        const cy = readFloat32();
        const cz = readFloat32();
        const crx = readFloat32();
        const cry = readFloat32();
        const crz = readFloat32();
        const fov = readFloat32();
        offset += 24; // interpolation

        vmd.cameraKeyframes.push({
          time: frameNumber / 30,
          position: new Float32Array([cx, cy, cz]),
          rotation: new Float32Array([crx, cry, crz]),
          distance,
          fov,
        });
      }
    } catch { /* camera data optional */ }
  }

  return vmd;
}

export async function loadVMD(url: string): Promise<VMDData> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return parseVMD(buffer);
}