export type TrackType = "rotation" | "position" | "scale" | "morph";

export interface Keyframe {
  time: number;
  value: number | Float32Array;
  controlPointIn?: number;
  controlPointOut?: number;
}

export interface BoneTrack {
  boneName: string;
  boneIndex: number;
  type: TrackType;
  keyframes: Keyframe[];
}

export interface MorphTrack {
  morphName: string;
  morphIndex: number;
  keyframes: Keyframe[];
}

export class AnimationClip {
  name: string;
  duration: number;
  boneTracks: BoneTrack[];
  morphTracks: MorphTrack[];

  constructor(
    name: string,
    duration: number,
    boneTracks: BoneTrack[],
    morphTracks: MorphTrack[] = []
  ) {
    this.name = name;
    this.duration = duration;
    this.boneTracks = boneTracks;
    this.morphTracks = morphTracks;
  }

  static fromTracks(name: string, tracks: BoneTrack[], morphTracks?: MorphTrack[]): AnimationClip {
    let maxTime = 0;
    for (const t of tracks) {
      for (const kf of t.keyframes) {
        if (kf.time > maxTime) maxTime = kf.time;
      }
    }
    if (morphTracks) {
      for (const t of morphTracks) {
        for (const kf of t.keyframes) {
          if (kf.time > maxTime) maxTime = kf.time;
        }
      }
    }
    return new AnimationClip(name, maxTime, tracks, morphTracks);
  }
}

export function sampleTrack(track: Keyframe[], time: number): number {
  if (track.length === 0) return 0;
  if (track.length === 1) return track[0].value as number;
  if (time <= track[0].time) return track[0].value as number;
  if (time >= track[track.length - 1].time) return track[track.length - 1].value as number;

  let lo = 0;
  let hi = track.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (track[mid].time <= time) lo = mid;
    else hi = mid;
  }

  const kf0 = track[lo];
  const kf1 = track[hi];
  const dt = kf1.time - kf0.time;
  if (dt <= 0) return kf0.value as number;

  const t = (time - kf0.time) / dt;

  if (kf0.controlPointOut !== undefined && kf1.controlPointIn !== undefined) {
    return bezierInterp(
      kf0.value as number,
      kf0.controlPointOut,
      kf1.controlPointIn,
      kf1.value as number,
      t
    );
  }

  return (kf0.value as number) * (1 - t) + (kf1.value as number) * t;
}

function bezierInterp(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const u = 1 - t;
  return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
}