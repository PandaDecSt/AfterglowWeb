import { vec3, type Vec3 } from "wgpu-matrix";

export const MmdCoord = {
  toRH(x: number, y: number, z: number): Vec3 {
    return vec3.create(x, y, -z);
  },

  toRHTarget(t: Vec3): Vec3 {
    return vec3.create(t[0], t[1], -t[2]);
  },

  toRHUp(u: Vec3): Vec3 {
    return vec3.create(u[0], u[1], -u[2]);
  },

  worldPos(wm: Float32Array, boneIdx: number): Vec3 {
    const o = boneIdx * 16;
    return vec3.create(wm[o + 12], wm[o + 13], -wm[o + 14]);
  },
};