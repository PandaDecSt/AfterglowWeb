import { quat, vec3, mat4, type Vec3, type Quat, type Mat4 } from "wgpu-matrix";
import type { Skeleton } from "./skeleton";
import type { PMXBone } from "../utils/pmx-loader";

export interface IKLink {
  index: number;
  hasLimit: boolean;
  limitMin: [number, number, number];
  limitMax: [number, number, number];
}

export interface IKChain {
  targetIndex: number;
  effectorIndex: number;
  links: IKLink[];
  iterations: number;
  maxAngle: number;
}

export function buildIKChains(bones: PMXBone[]): IKChain[] {
  const chains: IKChain[] = [];
  for (let i = 0; i < bones.length; i++) {
    const b = bones[i];
    if ((b.flag & 0x0020) === 0 || b.ikLinks.length === 0) continue;
    chains.push({
      targetIndex: i,
      effectorIndex: b.ikTargetIndex,
      links: b.ikLinks.map(l => ({
        index: l.linkIndex,
        hasLimit: l.hasLimit,
        limitMin: [
          Math.min(l.limitMin[0], l.limitMax[0]),
          Math.min(l.limitMin[1], l.limitMax[1]),
          Math.min(l.limitMin[2], l.limitMax[2]),
        ],
        limitMax: [
          Math.max(l.limitMin[0], l.limitMax[0]),
          Math.max(l.limitMin[1], l.limitMax[1]),
          Math.max(l.limitMin[2], l.limitMax[2]),
        ],
      })),
      iterations: b.ikLoopCount,
      maxAngle: b.ikUnitLength,
    });
  }
  return chains;
}

const enum SolveAxis { None = 0, Fixed = 1, X = 2, Y = 3, Z = 4 }
const enum EulerOrder { YXZ = 0, ZYX = 1, XZY = 2 }

interface ChainLink {
  boneIndex: number;
  hasLimit: boolean;
  minAngle: [number, number, number];
  maxAngle: [number, number, number];
  solveAxis: SolveAxis;
  rotationOrder: EulerOrder;
}

const HALF_PI = Math.PI * 0.5;

function classifyLink(link: IKLink): ChainLink {
  let solveAxis = SolveAxis.None;
  let rotationOrder = EulerOrder.XZY;
  if (link.hasLimit) {
    const [minX, minY, minZ] = link.limitMin;
    const [maxX, maxY, maxZ] = link.limitMax;
    if (minX === 0 && maxX === 0 && minY === 0 && maxY === 0 && minZ === 0 && maxZ === 0) {
      solveAxis = SolveAxis.Fixed;
    } else if (minY === 0 && maxY === 0 && minZ === 0 && maxZ === 0) {
      solveAxis = SolveAxis.X;
    } else if (minX === 0 && maxX === 0 && minZ === 0 && maxZ === 0) {
      solveAxis = SolveAxis.Y;
    } else if (minX === 0 && maxX === 0 && minY === 0 && maxY === 0) {
      solveAxis = SolveAxis.Z;
    }
    if (-HALF_PI < minX && maxX < HALF_PI) {
      rotationOrder = EulerOrder.YXZ;
    } else if (-HALF_PI < minY && maxY < HALF_PI) {
      rotationOrder = EulerOrder.ZYX;
    } else {
      rotationOrder = EulerOrder.XZY;
    }
  }
  return {
    boneIndex: link.index,
    hasLimit: link.hasLimit,
    minAngle: link.limitMin,
    maxAngle: link.limitMax,
    solveAxis,
    rotationOrder,
  };
}

const _v0 = vec3.create();
const _v1 = vec3.create();
const _v2 = vec3.create();
const _v3 = vec3.create();
const _v4 = vec3.create();
const _q0 = quat.create();
const _q1 = quat.create();
const _q2 = quat.create();
const _q3 = quat.create();
const _m0 = mat4.create();

const EPSILON = 1e-8;
const THRESHOLD = (88 * Math.PI) / 180;

function getWorldPos(wm: Float32Array, idx: number, out: Vec3): void {
  const off = idx * 16;
  out[0] = wm[off + 12];
  out[1] = wm[off + 13];
  out[2] = wm[off + 14];
}

function distance(wm: Float32Array, a: number, b: number): number {
  const dx = wm[a * 16 + 12] - wm[b * 16 + 12];
  const dy = wm[a * 16 + 13] - wm[b * 16 + 13];
  const dz = wm[a * 16 + 14] - wm[b * 16 + 14];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function limitAngle(angle: number, min: number, max: number, useAxis: boolean): number {
  if (angle < min) {
    const diff = 2 * min - angle;
    return diff <= max && useAxis ? diff : min;
  } else if (angle > max) {
    const diff = 2 * max - angle;
    return diff >= min && useAxis ? diff : max;
  }
  return angle;
}

function extractEuler(q: Quat, order: EulerOrder, out: Vec3): void {
  mat4.fromQuat(q, _m0);
  const m = _m0;
  switch (order) {
    case EulerOrder.YXZ: {
      let rX = Math.asin(-m[9]);
      if (Math.abs(rX) > THRESHOLD) rX = rX < 0 ? -THRESHOLD : THRESHOLD;
      let cosX = Math.cos(rX);
      if (cosX !== 0) cosX = 1 / cosX;
      out[0] = rX;
      out[1] = Math.atan2(m[8] * cosX, m[10] * cosX);
      out[2] = Math.atan2(m[1] * cosX, m[5] * cosX);
      break;
    }
    case EulerOrder.ZYX: {
      let rY = Math.asin(-m[2]);
      if (Math.abs(rY) > THRESHOLD) rY = rY < 0 ? -THRESHOLD : THRESHOLD;
      let cosY = Math.cos(rY);
      if (cosY !== 0) cosY = 1 / cosY;
      out[0] = Math.atan2(m[6] * cosY, m[10] * cosY);
      out[1] = rY;
      out[2] = Math.atan2(m[1] * cosY, m[0] * cosY);
      break;
    }
    case EulerOrder.XZY: {
      let rZ = Math.asin(-m[4]);
      if (Math.abs(rZ) > THRESHOLD) rZ = rZ < 0 ? -THRESHOLD : THRESHOLD;
      let cosZ = Math.cos(rZ);
      if (cosZ !== 0) cosZ = 1 / cosZ;
      out[0] = Math.atan2(m[6] * cosZ, m[5] * cosZ);
      out[1] = Math.atan2(m[8] * cosZ, m[0] * cosZ);
      out[2] = rZ;
      break;
    }
  }
}

const EULER_AXES: readonly [(number | number[]), (number | number[]), (number | number[])][] = [
  [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
  [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
  [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
];

function eulerToQuat(euler: Vec3, order: EulerOrder, out: Quat): void {
  const axes = EULER_AXES[order] as (number[])[];
  const ang1 = order === EulerOrder.YXZ ? euler[1] : order === EulerOrder.ZYX ? euler[2] : euler[0];
  const ang2 = order === EulerOrder.YXZ ? euler[0] : order === EulerOrder.ZYX ? euler[1] : euler[2];
  const ang3 = order === EulerOrder.YXZ ? euler[2] : order === EulerOrder.ZYX ? euler[0] : euler[1];
  quat.fromAxisAngle(axes[0] as unknown as Vec3, ang1, out);
  quat.fromAxisAngle(axes[1] as unknown as Vec3, ang2, _q3);
  quat.multiply(out, _q3, out);
  quat.fromAxisAngle(axes[2] as unknown as Vec3, ang3, _q3);
  quat.multiply(out, _q3, out);
}

function updateBoneWorld(
  skeleton: Skeleton,
  boneIdx: number,
  ikRotations: Float32Array,
): void {
  const off3 = boneIdx * 3;
  const off4 = boneIdx * 4;
  const wm = skeleton.worldMatrices;
  const lr = skeleton.localRotations;
  const lp = skeleton.localPositions;
  const bp = skeleton.bindPositions;

  const lx = lr[off4];
  const ly = lr[off4 + 1];
  const lz = lr[off4 + 2];
  const lw = lr[off4 + 3];

  const ix = ikRotations[off4];
  const iy = ikRotations[off4 + 1];
  const iz = ikRotations[off4 + 2];
  const iw = ikRotations[off4 + 3];

  let rx = iw * lx + ix * lw + iy * lz - iz * ly;
  let ry = iw * ly - ix * lz + iy * lw + iz * lx;
  let rz = iw * lz + ix * ly - iy * lx + iz * lw;
  let rw = iw * lw - ix * lx - iy * ly - iz * lz;
  const len = Math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw) || 1;
  rx /= len; ry /= len; rz /= len; rw /= len;

  let addTx = 0, addTy = 0, addTz = 0;
  const aFlags = skeleton.appendFlags[boneIdx];
  if (aFlags !== 0) {
    const aParent = skeleton.appendParentIndices[boneIdx];
    if (aParent >= 0 && aParent < skeleton.boneCount) {
      const ratio = skeleton.appendRatios[boneIdx];
      const absRatio = Math.abs(ratio);
      if (absRatio > 1e-6) {
        if (aFlags & 1) {
          const apOff4 = aParent * 4;
          let ax = lr[apOff4];
          let ay = lr[apOff4 + 1];
          let az = lr[apOff4 + 2];
          let aw = lr[apOff4 + 3];
          const aikx = ikRotations[apOff4];
          const aiky = ikRotations[apOff4 + 1];
          const aikz = ikRotations[apOff4 + 2];
          const aikw = ikRotations[apOff4 + 3];
          if (aikx * aikx + aiky * aiky + aikz * aikz > 1e-12) {
            const nax = aikw * ax + aikx * aw + aiky * az - aikz * ay;
            const nay = aikw * ay - aikx * az + aiky * aw + aikz * ax;
            const naz = aikw * az + aikx * ay - aiky * ax + aikz * aw;
            const naw = aikw * aw - aikx * ax - aiky * ay - aikz * az;
            ax = nax; ay = nay; az = naz; aw = naw;
          }
          if (ratio < 0) { ax = -ax; ay = -ay; az = -az; }
          _q2[0] = 0; _q2[1] = 0; _q2[2] = 0; _q2[3] = 1;
          _q3[0] = ax; _q3[1] = ay; _q3[2] = az; _q3[3] = aw;
          quat.slerp(_q2, _q3, absRatio, _q3);
          const sx = _q3[0], sy = _q3[1], sz = _q3[2], sw = _q3[3];
          const nx = sw * rx + sx * rw + sy * rz - sz * ry;
          const ny = sw * ry - sx * rz + sy * rw + sz * rx;
          const nz = sw * rz + sx * ry - sy * rx + sz * rw;
          const nw = sw * rw - sx * rx - sy * ry - sz * rz;
          rx = nx; ry = ny; rz = nz; rw = nw;
        }
        if (aFlags & 2) {
          const aOff3 = aParent * 3;
          addTx = (lp[aOff3] - bp[aOff3]) * ratio;
          addTy = (lp[aOff3 + 1] - bp[aOff3 + 1]) * ratio;
          addTz = (lp[aOff3 + 2] - bp[aOff3 + 2]) * ratio;
        }
      }
    }
  }

  const posX = lp[off3] + addTx;
  const posY = lp[off3 + 1] + addTy;
  const posZ = lp[off3 + 2] + addTz;

  const x2 = rx + rx, y2 = ry + ry, z2 = rz + rz;
  const xx = rx * x2, xy = rx * y2, xz = rx * z2;
  const yy = ry * y2, yz = ry * z2, zz = rz * z2;
  const wx = rw * x2, wy = rw * y2, wz = rw * z2;

  const wOff = boneIdx * 16;
  wm[wOff + 0] = 1 - (yy + zz); wm[wOff + 1] = xy + wz;       wm[wOff + 2] = xz - wy;       wm[wOff + 3] = 0;
  wm[wOff + 4] = xy - wz;       wm[wOff + 5] = 1 - (xx + zz); wm[wOff + 6] = yz + wx;       wm[wOff + 7] = 0;
  wm[wOff + 8] = xz + wy;       wm[wOff + 9] = yz - wx;       wm[wOff + 10] = 1 - (xx + yy); wm[wOff + 11] = 0;
  wm[wOff + 12] = posX;
  wm[wOff + 13] = posY;
  wm[wOff + 14] = posZ;
  wm[wOff + 15] = 1;

  const parentIdx = skeleton.parentIndices[boneIdx];
  if (parentIdx >= 0) {
    const pOff = parentIdx * 16;
    const p = wm;
    const l0 = wm[wOff], l1 = wm[wOff + 1], l2 = wm[wOff + 2], l3 = wm[wOff + 3];
    const l4 = wm[wOff + 4], l5 = wm[wOff + 5], l6 = wm[wOff + 6], l7 = wm[wOff + 7];
    const l8 = wm[wOff + 8], l9 = wm[wOff + 9], l10 = wm[wOff + 10], l11 = wm[wOff + 11];
    const l12 = wm[wOff + 12], l13 = wm[wOff + 13], l14 = wm[wOff + 14], l15 = wm[wOff + 15];

    wm[wOff + 0] = p[pOff] * l0 + p[pOff + 4] * l1 + p[pOff + 8] * l2 + p[pOff + 12] * l3;
    wm[wOff + 1] = p[pOff + 1] * l0 + p[pOff + 5] * l1 + p[pOff + 9] * l2 + p[pOff + 13] * l3;
    wm[wOff + 2] = p[pOff + 2] * l0 + p[pOff + 6] * l1 + p[pOff + 10] * l2 + p[pOff + 14] * l3;
    wm[wOff + 3] = p[pOff + 3] * l0 + p[pOff + 7] * l1 + p[pOff + 11] * l2 + p[pOff + 15] * l3;
    wm[wOff + 4] = p[pOff] * l4 + p[pOff + 4] * l5 + p[pOff + 8] * l6 + p[pOff + 12] * l7;
    wm[wOff + 5] = p[pOff + 1] * l4 + p[pOff + 5] * l5 + p[pOff + 9] * l6 + p[pOff + 13] * l7;
    wm[wOff + 6] = p[pOff + 2] * l4 + p[pOff + 6] * l5 + p[pOff + 10] * l6 + p[pOff + 14] * l7;
    wm[wOff + 7] = p[pOff + 3] * l4 + p[pOff + 7] * l5 + p[pOff + 11] * l6 + p[pOff + 15] * l7;
    wm[wOff + 8] = p[pOff] * l8 + p[pOff + 4] * l9 + p[pOff + 8] * l10 + p[pOff + 12] * l11;
    wm[wOff + 9] = p[pOff + 1] * l8 + p[pOff + 5] * l9 + p[pOff + 9] * l10 + p[pOff + 13] * l11;
    wm[wOff + 10] = p[pOff + 2] * l8 + p[pOff + 6] * l9 + p[pOff + 10] * l10 + p[pOff + 14] * l11;
    wm[wOff + 11] = p[pOff + 3] * l8 + p[pOff + 7] * l9 + p[pOff + 11] * l10 + p[pOff + 15] * l11;
    wm[wOff + 12] = p[pOff] * l12 + p[pOff + 4] * l13 + p[pOff + 8] * l14 + p[pOff + 12] * l15;
    wm[wOff + 13] = p[pOff + 1] * l12 + p[pOff + 5] * l13 + p[pOff + 9] * l14 + p[pOff + 13] * l15;
    wm[wOff + 14] = p[pOff + 2] * l12 + p[pOff + 6] * l13 + p[pOff + 10] * l14 + p[pOff + 14] * l15;
    wm[wOff + 15] = p[pOff + 3] * l12 + p[pOff + 7] * l13 + p[pOff + 11] * l14 + p[pOff + 15] * l15;
  }

  const wr = skeleton.worldRotations;
  if (parentIdx >= 0) {
    const pOff4 = parentIdx * 4;
    const pwx = wr[pOff4], pwy = wr[pOff4 + 1], pwz = wr[pOff4 + 2], pww = wr[pOff4 + 3];
    wr[off4]     = pww * rx + pwx * rw + pwy * rz - pwz * ry;
    wr[off4 + 1] = pww * ry - pwx * rz + pwy * rw + pwz * rx;
    wr[off4 + 2] = pww * rz + pwx * ry - pwy * rx + pwz * rw;
    wr[off4 + 3] = pww * rw - pwx * rx - pwy * ry - pwz * rz;
  } else {
    wr[off4] = rx; wr[off4 + 1] = ry; wr[off4 + 2] = rz; wr[off4 + 3] = rw;
  }
}

let _ikDebugCount = 0;
let _chainLinkCache: ChainLink[][] | null = null;

export function solveIK(skeleton: Skeleton, chains: IKChain[]): void {
  const wm = skeleton.worldMatrices;
  const boneCount = skeleton.boneCount;

  if (!_ikRotBuf || _ikRotBuf.length < boneCount * 4) {
    _ikRotBuf = new Float32Array(boneCount * 4);
  }
  const ikRot = _ikRotBuf;
  for (let i = 0; i < boneCount; i++) {
    ikRot[i * 4] = 0; ikRot[i * 4 + 1] = 0; ikRot[i * 4 + 2] = 0; ikRot[i * 4 + 3] = 1;
  }

  if (!_chainLinkCache || _chainLinkCache.length !== chains.length) {
    _chainLinkCache = chains.map(c => c.links.map(classifyLink));
  }

  for (let ci = 0; ci < chains.length; ci++) {
    const chain = chains[ci];
    const chainLinks = _chainLinkCache[ci];
    const { targetIndex, effectorIndex, iterations, maxAngle } = chain;

    const doDebug = _ikDebugCount < 1 && ci < 2;
    if (doDebug) {
      const tName = skeleton.boneNames[targetIndex];
      const eName = skeleton.boneNames[effectorIndex];
      console.log(`[IK#${_ikDebugCount}] chain${ci} ${tName}->${eName} BEFORE: dist=${distance(wm, targetIndex, effectorIndex).toFixed(4)} iter=${iterations} maxAngle=${maxAngle.toFixed(3)}`);
      for (let li2 = 0; li2 < chainLinks.length; li2++) {
        const bIdx = chainLinks[li2].boneIndex;
        const o4 = bIdx * 4;
        const lr = skeleton.localRotations;
        const cl2 = chainLinks[li2];
        const orderName = cl2.rotationOrder === EulerOrder.YXZ ? "YXZ" : cl2.rotationOrder === EulerOrder.ZYX ? "ZYX" : "XZY";
        console.log(`  link${li2} ${skeleton.boneNames[bIdx]} localRot=[${lr[o4].toFixed(4)},${lr[o4+1].toFixed(4)},${lr[o4+2].toFixed(4)},${lr[o4+3].toFixed(4)}] solveAxis=${cl2.solveAxis} order=${orderName}`);
      }
    }

    for (const cl of chainLinks) {
      const off = cl.boneIndex * 4;
      ikRot[off] = 0; ikRot[off + 1] = 0; ikRot[off + 2] = 0; ikRot[off + 3] = 1;
    }

    if (distance(wm, targetIndex, effectorIndex) < EPSILON) continue;

    for (let i = chainLinks.length - 1; i >= 0; i--) {
      updateBoneWorld(skeleton, chainLinks[i].boneIndex, ikRot);
    }
    updateBoneWorld(skeleton, effectorIndex, ikRot);

    if (distance(wm, targetIndex, effectorIndex) < EPSILON) continue;

    const halfIter = iterations >> 1;

    for (let iter = 0; iter < iterations; iter++) {
      const useAxis = iter < halfIter;

      for (let li = 0; li < chainLinks.length; li++) {
        const cl = chainLinks[li];
        if (cl.solveAxis === SolveAxis.Fixed) continue;

        const boneIdx = cl.boneIndex;

        getWorldPos(wm, boneIdx, _v0);
        getWorldPos(wm, targetIndex, _v1);
        getWorldPos(wm, effectorIndex, _v2);

        vec3.subtract(_v0, _v2, _v3);
        const lenCT = vec3.length(_v3);
        if (lenCT < EPSILON) continue;
        vec3.scale(_v3, 1 / lenCT, _v3);

        vec3.subtract(_v0, _v1, _v4);
        const lenCI = vec3.length(_v4);
        if (lenCI < EPSILON) continue;
        vec3.scale(_v4, 1 / lenCI, _v4);

        vec3.cross(_v3, _v4, _v0);
        if (vec3.length(_v0) < EPSILON) continue;
        vec3.normalize(_v0, _v0);

        let finalAxis: Vec3;
        if (cl.hasLimit && useAxis && cl.solveAxis >= SolveAxis.X) {
          const parentIdx = skeleton.parentIndices[boneIdx];
          const pOff = parentIdx >= 0 ? parentIdx * 16 : -1;
          const colOff = (cl.solveAxis - SolveAxis.X) * 4;
          let ax: number, ay: number, az: number;
          if (pOff >= 0) {
            ax = wm[pOff + colOff]; ay = wm[pOff + colOff + 1]; az = wm[pOff + colOff + 2];
          } else {
            ax = colOff === 0 ? 1 : 0; ay = colOff === 4 ? 1 : 0; az = colOff === 8 ? 1 : 0;
          }
          const dotA = _v0[0] * ax + _v0[1] * ay + _v0[2] * az;
          const sign = dotA >= 0 ? 1 : -1;
          _v1[0] = cl.solveAxis === SolveAxis.X ? sign : 0;
          _v1[1] = cl.solveAxis === SolveAxis.Y ? sign : 0;
          _v1[2] = cl.solveAxis === SolveAxis.Z ? sign : 0;
          finalAxis = _v1;
        } else {
          const parentIdx = skeleton.parentIndices[boneIdx];
          if (parentIdx >= 0) {
            const pOff = parentIdx * 16;
            const x = _v0[0], y = _v0[1], z = _v0[2];
            _v1[0] = wm[pOff] * x + wm[pOff + 1] * y + wm[pOff + 2] * z;
            _v1[1] = wm[pOff + 4] * x + wm[pOff + 5] * y + wm[pOff + 6] * z;
            _v1[2] = wm[pOff + 8] * x + wm[pOff + 9] * y + wm[pOff + 10] * z;
            vec3.normalize(_v1, _v1);
          } else {
            _v1[0] = _v0[0]; _v1[1] = _v0[1]; _v1[2] = _v0[2];
          }
          finalAxis = _v1;
        }

        let dotTI = vec3.dot(_v3, _v4);
        dotTI = Math.max(-1, Math.min(1, dotTI));
        const angle = Math.min(maxAngle * (li + 1), Math.acos(dotTI));

        quat.fromAxisAngle(finalAxis, angle, _q0);

        if (doDebug && iter < 3) {
          console.log(`  iter${iter} link${li} ${skeleton.boneNames[boneIdx]}: finalAxis=[${finalAxis[0].toFixed(4)},${finalAxis[1].toFixed(4)},${finalAxis[2].toFixed(4)}] angle=${(angle*180/Math.PI).toFixed(1)}° dotTI=${dotTI.toFixed(4)} useAxis=${useAxis}`);
        }

        const off4 = boneIdx * 4;
        _q1[0] = ikRot[off4]; _q1[1] = ikRot[off4 + 1]; _q1[2] = ikRot[off4 + 2]; _q1[3] = ikRot[off4 + 3];
        quat.multiply(_q0, _q1, _q1);

        if (cl.hasLimit) {
          const lx = skeleton.localRotations[off4];
          const ly = skeleton.localRotations[off4 + 1];
          const lz = skeleton.localRotations[off4 + 2];
          const lw = skeleton.localRotations[off4 + 3];
          quat.multiply(_q1, [lx, ly, lz, lw] as unknown as Quat, _q2);

          extractEuler(_q2, cl.rotationOrder, _v2);

          _v2[0] = limitAngle(_v2[0], cl.minAngle[0], cl.maxAngle[0], useAxis);
          _v2[1] = limitAngle(_v2[1], cl.minAngle[1], cl.maxAngle[1], useAxis);
          _v2[2] = limitAngle(_v2[2], cl.minAngle[2], cl.maxAngle[2], useAxis);

          eulerToQuat(_v2, cl.rotationOrder, _q1);

          _q3[0] = -lx; _q3[1] = -ly; _q3[2] = -lz; _q3[3] = lw;
          quat.multiply(_q1, _q3, _q1);
        }

        quat.normalize(_q1, _q1);
        ikRot[off4] = _q1[0]; ikRot[off4 + 1] = _q1[1]; ikRot[off4 + 2] = _q1[2]; ikRot[off4 + 3] = _q1[3];

        for (let ui = li; ui >= 0; ui--) {
          updateBoneWorld(skeleton, chainLinks[ui].boneIndex, ikRot);
        }
        updateBoneWorld(skeleton, effectorIndex, ikRot);
      }

      if (distance(wm, targetIndex, effectorIndex) < 0.1) break;
    }

    if (doDebug) {
      const d = distance(wm, targetIndex, effectorIndex);
      console.log(`[IK#${_ikDebugCount}] chain${ci} AFTER: dist=${d.toFixed(4)}`);
      for (let li2 = 0; li2 < chainLinks.length; li2++) {
        const bIdx = chainLinks[li2].boneIndex;
        const o4 = bIdx * 4;
        const lr = skeleton.localRotations;
        const rw = lr[o4+3];
        const angle = 2 * Math.acos(Math.min(1, Math.abs(rw))) * 180 / Math.PI;
        console.log(`  link${li2} ${skeleton.boneNames[bIdx]} localRot=[${lr[o4].toFixed(4)},${lr[o4+1].toFixed(4)},${lr[o4+2].toFixed(4)},${lr[o4+3].toFixed(4)}] angle=${angle.toFixed(1)}°`);
      }
    }

    if (doDebug) {
      for (let li2 = 0; li2 < chainLinks.length; li2++) {
        const bIdx = chainLinks[li2].boneIndex;
        const o4 = bIdx * 4;
        console.log(`  BEFORE APPLY ${skeleton.boneNames[bIdx]} ikRot=[${ikRot[o4].toFixed(6)},${ikRot[o4+1].toFixed(6)},${ikRot[o4+2].toFixed(6)},${ikRot[o4+3].toFixed(6)}] localRot=[${skeleton.localRotations[o4].toFixed(6)},${skeleton.localRotations[o4+1].toFixed(6)},${skeleton.localRotations[o4+2].toFixed(6)},${skeleton.localRotations[o4+3].toFixed(6)}]`);
      }
    }
    for (const cl of chainLinks) {
      const off = cl.boneIndex * 4;
      const ix = ikRot[off], iy = ikRot[off + 1], iz = ikRot[off + 2], iw = ikRot[off + 3];
      if (ix * ix + iy * iy + iz * iz < EPSILON * EPSILON) continue;
      const lx = skeleton.localRotations[off];
      const ly = skeleton.localRotations[off + 1];
      const lz = skeleton.localRotations[off + 2];
      const lw = skeleton.localRotations[off + 3];
      const nx = iw * lx + ix * lw + iy * lz - iz * ly;
      const ny = iw * ly - ix * lz + iy * lw + iz * lx;
      const nz = iw * lz + ix * ly - iy * lx + iz * lw;
      const nw = iw * lw - ix * lx - iy * ly - iz * lz;
      const len = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw) || 1;
      skeleton.localRotations[off] = nx / len;
      skeleton.localRotations[off + 1] = ny / len;
      skeleton.localRotations[off + 2] = nz / len;
      skeleton.localRotations[off + 3] = nw / len;
      if (doDebug) {
        console.log(`  AFTER APPLY ${skeleton.boneNames[cl.boneIndex]} localRot=[${(nx/len).toFixed(6)},${(ny/len).toFixed(6)},${(nz/len).toFixed(6)},${(nw/len).toFixed(6)}]`);
      }
    }

    skeleton.updateWorldMatrices();
    if (doDebug) {
      for (let li2 = 0; li2 < chainLinks.length; li2++) {
        const bIdx = chainLinks[li2].boneIndex;
        const o4 = bIdx * 4;
        const lr = skeleton.localRotations;
        console.log(`  AFTER updateWM ${skeleton.boneNames[bIdx]} localRot=[${lr[o4].toFixed(6)},${lr[o4+1].toFixed(6)},${lr[o4+2].toFixed(6)},${lr[o4+3].toFixed(6)}]`);
      }
    }
  }

  _ikDebugCount++;
}

let _ikRotBuf: Float32Array | null = null;
