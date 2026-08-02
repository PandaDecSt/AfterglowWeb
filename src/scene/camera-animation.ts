import type { VMDCameraFrame } from "../utils/vmd-loader";
import { evalBezier } from "../utils/vmd-loader";

const FPS = 30.0;

export interface CameraPose {
  target: [number, number, number];
  rotation: [number, number, number];
  distance: number;
  fov: number;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function bez(ip: Uint8Array, channel: number, t: number): number {
  const b = channel * 4;
  return evalBezier(ip[b] / 127.0, ip[b + 1] / 127.0, ip[b + 2] / 127.0, ip[b + 3] / 127.0, t);
}

export class CameraAnimation {
  private frames: VMDCameraFrame[];

  constructor(frames: VMDCameraFrame[]) {
    this.frames = frames;
  }

  get frameCount(): number {
    return this.frames.length;
  }

  sample(timeSec: number): CameraPose | null {
    const frames = this.frames;
    const n = frames.length;
    if (+n === 0) return null;

    const frameNum = timeSec * FPS;
    if (frameNum <= frames[0].frame) return this.pose(frames[0]);
    if (frameNum >= frames[n - 1].frame) return this.pose(frames[n - 1]);

    let lo = 0, hi = n - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].frame <= frameNum) lo = mid; else hi = mid;
    }

    const a = frames[lo], b = frames[hi];
    const span = b.frame - a.frame;
    const t = span > 0 ? (frameNum - a.frame) / span : 0;
    const ip = b.interpolation;

    return {
      target: [
        lerp(a.target[0], b.target[0], bez(ip, 0, t)),
        lerp(a.target[1], b.target[1], bez(ip, 1, t)),
        lerp(a.target[2], b.target[2], bez(ip, 2, t)),
      ],
      rotation: (() => {
        const w = bez(ip, 3, t);
        return [
          lerp(a.rotation[0], b.rotation[0], w),
          lerp(a.rotation[1], b.rotation[1], w),
          lerp(a.rotation[2], b.rotation[2], w),
        ];
      })(),
      distance: lerp(a.distance, b.distance, bez(ip, 4, t)),
      fov: lerp(a.fov, b.fov, bez(ip, 5, t)),
    };
  }

  private pose(f: VMDCameraFrame): CameraPose {
    return {
      target: [f.target[0], f.target[1], f.target[2]],
      rotation: [f.rotation[0], f.rotation[1], f.rotation[2]],
      distance: f.distance,
      fov: f.fov,
    };
  }
}