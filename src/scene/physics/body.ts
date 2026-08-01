import { RigidbodyType, RigidbodyShape, type Rigidbody } from "./types";

export class RigidBodyStore {
  readonly count: number;
  readonly positions: Float32Array;
  readonly orientations: Float32Array;
  readonly linearVelocities: Float32Array;
  readonly angularVelocities: Float32Array;
  readonly invMass: Float32Array;
  readonly invInertiaLocal: Float32Array;
  readonly invInertiaWorld: Float32Array;
  readonly linearDamping: Float32Array;
  readonly angularDamping: Float32Array;
  readonly type: Uint8Array;
  readonly aligned: Uint8Array;
  readonly boneIndex: Int32Array;
  readonly friction: Float32Array;
  readonly restitution: Float32Array;
  readonly collisionGroup: Uint16Array;
  readonly willCollideMask: Uint16Array;
  readonly shape: Uint8Array;
  readonly size: Float32Array;
  readonly aabbMin: Float32Array;
  readonly aabbMax: Float32Array;
  readonly bodyOffsetMatrix: Float32Array;
  readonly bodyOffsetInverse: Float32Array;
  private boneOffsetsReady = false;
  private collisionPairs: Uint16Array | null = null;

  constructor(rigidbodies: Rigidbody[]) {
    const N = rigidbodies.length;
    this.count = N;
    this.positions = new Float32Array(N * 3);
    this.orientations = new Float32Array(N * 4);
    this.linearVelocities = new Float32Array(N * 3);
    this.angularVelocities = new Float32Array(N * 3);
    this.invMass = new Float32Array(N);
    this.invInertiaLocal = new Float32Array(N * 3);
    this.invInertiaWorld = new Float32Array(N * 9);
    this.linearDamping = new Float32Array(N);
    this.angularDamping = new Float32Array(N);
    this.type = new Uint8Array(N);
    this.aligned = new Uint8Array(N);
    this.boneIndex = new Int32Array(N);
    this.bodyOffsetMatrix = new Float32Array(N * 16);
    this.bodyOffsetInverse = new Float32Array(N * 16);
    this.friction = new Float32Array(N);
    this.restitution = new Float32Array(N);
    this.collisionGroup = new Uint16Array(N);
    this.willCollideMask = new Uint16Array(N);
    this.shape = new Uint8Array(N);
    this.size = new Float32Array(N * 3);
    this.aabbMin = new Float32Array(N * 3);
    this.aabbMax = new Float32Array(N * 3);

    for (let i = 0; i < N; i++) {
      const rb = rigidbodies[i];
      const i3 = i * 3;
      const i4 = i * 4;
      this.positions[i3] = rb.shapePosition[0];
      this.positions[i3 + 1] = rb.shapePosition[1];
      this.positions[i3 + 2] = rb.shapePosition[2];
      const q = eulerToQuat(rb.shapeRotation[0], rb.shapeRotation[1], rb.shapeRotation[2]);
      this.orientations[i4] = q[0];
      this.orientations[i4 + 1] = q[1];
      this.orientations[i4 + 2] = q[2];
      this.orientations[i4 + 3] = q[3];
      const dynamic = rb.type === RigidbodyType.Dynamic && rb.mass > 0;
      this.invMass[i] = dynamic ? 1 / rb.mass : 0;
      if (dynamic) computeLocalInvInertia(rb, this.invInertiaLocal, i * 3);
      this.linearDamping[i] = rb.linearDamping;
      this.angularDamping[i] = rb.angularDamping;
      this.type[i] = rb.type;
      this.aligned[i] = rb.aligned ? 1 : 0;
      this.boneIndex[i] = rb.boneIndex;
      this.friction[i] = rb.friction;
      this.restitution[i] = rb.restitution;
      this.collisionGroup[i] = 1 << (rb.group & 0xf);
      this.willCollideMask[i] = rb.collisionMask & 0xffff;
      this.shape[i] = rb.shape;
      this.size[i3] = rb.size[0];
      this.size[i3 + 1] = rb.size[1];
      this.size[i3 + 2] = rb.size[2];
    }
  }

  updateInvInertiaWorld(): void {
    const N = this.count;
    const ori = this.orientations;
    const local = this.invInertiaLocal;
    const W = this.invInertiaWorld;
    const invMass = this.invMass;
    for (let i = 0; i < N; i++) {
      if (invMass[i] <= 0) continue;
      const i3 = i * 3, i4 = i * 4, i9 = i * 9;
      const qx = ori[i4], qy = ori[i4 + 1], qz = ori[i4 + 2], qw = ori[i4 + 3];
      const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
      const xx = qx * x2, yy = qy * y2, zz = qz * z2;
      const xy = qx * y2, xz = qx * z2, yz = qy * z2;
      const wx = qw * x2, wy = qw * y2, wz = qw * z2;
      const r00 = 1 - (yy + zz), r01 = xy - wz, r02 = xz + wy;
      const r10 = xy + wz, r11 = 1 - (xx + zz), r12 = yz - wx;
      const r20 = xz - wy, r21 = yz + wx, r22 = 1 - (xx + yy);
      const d0 = local[i3], d1 = local[i3 + 1], d2 = local[i3 + 2];
      const a0 = r00 * d0, a1 = r01 * d1, a2 = r02 * d2;
      const b0 = r10 * d0, b1 = r11 * d1, b2 = r12 * d2;
      const c0 = r20 * d0, c1 = r21 * d1, c2 = r22 * d2;
      const w00 = a0 * r00 + a1 * r01 + a2 * r02;
      const w01 = a0 * r10 + a1 * r11 + a2 * r12;
      const w02 = a0 * r20 + a1 * r21 + a2 * r22;
      const w11 = b0 * r10 + b1 * r11 + b2 * r12;
      const w12 = b0 * r20 + b1 * r21 + b2 * r22;
      const w22 = c0 * r20 + c1 * r21 + c2 * r22;
      W[i9] = w00; W[i9 + 1] = w01; W[i9 + 2] = w02;
      W[i9 + 3] = w01; W[i9 + 4] = w11; W[i9 + 5] = w12;
      W[i9 + 6] = w02; W[i9 + 7] = w12; W[i9 + 8] = w22;
    }
  }

  updateAabbs(margin = 0.5): void {
    const N = this.count;
    const pos = this.positions, ori = this.orientations, shapes = this.shape, sz = this.size;
    const minA = this.aabbMin, maxA = this.aabbMax;
    for (let i = 0; i < N; i++) {
      const i3 = i * 3, i4 = i * 4;
      const px = pos[i3], py = pos[i3 + 1], pz = pos[i3 + 2];
      let hx = 0, hy = 0, hz = 0;
      switch (shapes[i]) {
        case RigidbodyShape.Sphere: { hx = hy = hz = sz[i3]; break; }
        case RigidbodyShape.Box: {
          const qx = ori[i4], qy = ori[i4 + 1], qz = ori[i4 + 2], qw = ori[i4 + 3];
          const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
          const xx = qx * x2, yy = qy * y2, zz = qz * z2;
          const xy = qx * y2, xz = qx * z2, yz = qy * z2;
          const wx = qw * x2, wy = qw * y2, wz = qw * z2;
          const m00 = Math.abs(1 - (yy + zz)), m01 = Math.abs(xy + wz), m02 = Math.abs(xz - wy);
          const m10 = Math.abs(xy - wz), m11 = Math.abs(1 - (xx + zz)), m12 = Math.abs(yz + wx);
          const m20 = Math.abs(xz + wy), m21 = Math.abs(yz - wx), m22 = Math.abs(1 - (xx + yy));
          const sx = sz[i3], sy = sz[i3 + 1], szz = sz[i3 + 2];
          hx = m00 * sx + m01 * sy + m02 * szz;
          hy = m10 * sx + m11 * sy + m12 * szz;
          hz = m20 * sx + m21 * sy + m22 * szz;
          break;
        }
        case RigidbodyShape.Capsule: {
          const r = sz[i3], halfH = sz[i3 + 1] * 0.5;
          const qx = ori[i4], qy = ori[i4 + 1], qz = ori[i4 + 2], qw = ori[i4 + 3];
          const rx = 2 * (qx * qy - qw * qz), ry = 1 - 2 * (qx * qx + qz * qz), rz = 2 * (qy * qz + qw * qx);
          hx = Math.abs(rx) * halfH + r; hy = Math.abs(ry) * halfH + r; hz = Math.abs(rz) * halfH + r;
          break;
        }
      }
      minA[i3] = px - hx - margin; minA[i3 + 1] = py - hy - margin; minA[i3 + 2] = pz - hz - margin;
      maxA[i3] = px + hx + margin; maxA[i3 + 1] = py + hy + margin; maxA[i3 + 2] = pz + hz + margin;
    }
  }

  computeBoneOffsets(boneInverseBindMatrices: Float32Array): void {
    const N = this.count;
    const offsets = this.bodyOffsetMatrix, inverses = this.bodyOffsetInverse;
    const ori = this.orientations, pos = this.positions, boneIdx = this.boneIndex;
    const totalBones = boneInverseBindMatrices.length / 16;
    const shapeWorldBind = new Float32Array(16), offsetMat = new Float32Array(16);
    for (let i = 0; i < N; i++) {
      const dst = i * 16, b = boneIdx[i];
      if (b < 0 || b >= totalBones) { identity16(offsets, dst); identity16(inverses, dst); continue; }
      const i3 = i * 3, i4 = i * 4;
      fromPositionRotation(pos[i3], pos[i3 + 1], pos[i3 + 2], ori[i4], ori[i4 + 1], ori[i4 + 2], ori[i4 + 3], shapeWorldBind);
      mulArrays(boneInverseBindMatrices, b * 16, shapeWorldBind, 0, offsetMat, 0);
      offsets.set(offsetMat, dst);
      const inverseTmp = new Float32Array(16);
      if (inverseInto(offsetMat, inverseTmp)) { inverses.set(inverseTmp, dst); }
      else { identity16(inverses, dst); }
    }
    this.boneOffsetsReady = true;
  }

  isBoneOffsetsReady(): boolean { return this.boneOffsetsReady; }

  getCollisionPairs(): Uint16Array {
    if (this.collisionPairs !== null) return this.collisionPairs;
    const N = this.count, invMass = this.invMass, group = this.collisionGroup, mask = this.willCollideMask;
    const buf: number[] = [];
    for (let i = 0; i < N; i++) {
      const gi = group[i], mi = mask[i], dynA = invMass[i] > 0;
      for (let j = i + 1; j < N; j++) {
        if (!dynA && invMass[j] === 0) continue;
        if ((mi & group[j]) === 0 || (mask[j] & gi) === 0) continue;
        buf.push(i, j);
      }
    }
    this.collisionPairs = new Uint16Array(buf);
    return this.collisionPairs;
  }
}

function computeLocalInvInertia(rb: Rigidbody, out: Float32Array, o: number): void {
  const m = rb.mass;
  if (m <= 0) return;
  let Ix: number, Iy: number, Iz: number;
  switch (rb.shape) {
    case RigidbodyShape.Sphere: { const I = 0.4 * m * rb.size[0] * rb.size[0]; Ix = I; Iy = I; Iz = I; break; }
    case RigidbodyShape.Box: {
      const lx2 = 4 * rb.size[0] * rb.size[0], ly2 = 4 * rb.size[1] * rb.size[1], lz2 = 4 * rb.size[2] * rb.size[2];
      Ix = (m / 12) * (ly2 + lz2); Iy = (m / 12) * (lx2 + lz2); Iz = (m / 12) * (lx2 + ly2); break;
    }
    case RigidbodyShape.Capsule: {
      const lx = 2 * rb.size[0], ly = rb.size[1] + 2 * rb.size[0];
      const lx2 = lx * lx, ly2 = ly * ly;
      Ix = (m / 12) * (ly2 + lx2); Iy = (m / 12) * (lx2 + lx2); Iz = (m / 12) * (lx2 + ly2); break;
    }
    default: { Ix = m; Iy = m; Iz = m; }
  }
  out[o] = Ix > 0 ? 1 / Ix : 0;
  out[o + 1] = Iy > 0 ? 1 / Iy : 0;
  out[o + 2] = Iz > 0 ? 1 / Iz : 0;
}

export function identity16(out: Float32Array, offset: number): void {
  out[offset] = 1; out[offset + 1] = 0; out[offset + 2] = 0; out[offset + 3] = 0;
  out[offset + 4] = 0; out[offset + 5] = 1; out[offset + 6] = 0; out[offset + 7] = 0;
  out[offset + 8] = 0; out[offset + 9] = 0; out[offset + 10] = 1; out[offset + 11] = 0;
  out[offset + 12] = 0; out[offset + 13] = 0; out[offset + 14] = 0; out[offset + 15] = 1;
}

export function fromPositionRotation(px: number, py: number, pz: number, qx: number, qy: number, qz: number, qw: number, out: Float32Array): void {
  const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
  const xx = qx * x2, yy = qy * y2, zz = qz * z2;
  const xy = qx * y2, xz = qx * z2, yz = qy * z2;
  const wx = qw * x2, wy = qw * y2, wz = qw * z2;
  out[0] = 1 - (yy + zz); out[1] = xy + wz; out[2] = xz - wy; out[3] = 0;
  out[4] = xy - wz; out[5] = 1 - (xx + zz); out[6] = yz + wx; out[7] = 0;
  out[8] = xz + wy; out[9] = yz - wx; out[10] = 1 - (xx + yy); out[11] = 0;
  out[12] = px; out[13] = py; out[14] = pz; out[15] = 1;
}

export function mulArrays(a: Float32Array, aOff: number, b: Float32Array, bOff: number, out: Float32Array, outOff: number): void {
  for (let col = 0; col < 4; col++) {
    const b0 = b[bOff + col * 4], b1 = b[bOff + col * 4 + 1], b2 = b[bOff + col * 4 + 2], b3 = b[bOff + col * 4 + 3];
    out[outOff + col * 4] = a[aOff] * b0 + a[aOff + 4] * b1 + a[aOff + 8] * b2 + a[aOff + 12] * b3;
    out[outOff + col * 4 + 1] = a[aOff + 1] * b0 + a[aOff + 5] * b1 + a[aOff + 9] * b2 + a[aOff + 13] * b3;
    out[outOff + col * 4 + 2] = a[aOff + 2] * b0 + a[aOff + 6] * b1 + a[aOff + 10] * b2 + a[aOff + 14] * b3;
    out[outOff + col * 4 + 3] = a[aOff + 3] * b0 + a[aOff + 7] * b1 + a[aOff + 11] * b2 + a[aOff + 15] * b3;
  }
}

export function inverseInto(m: Float32Array, out: Float32Array): boolean {
  const m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
  const m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
  const m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
  const m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];
  const b00 = m00 * m11 - m01 * m10, b01 = m00 * m12 - m02 * m10;
  const b02 = m00 * m13 - m03 * m10, b03 = m01 * m12 - m02 * m11;
  const b04 = m01 * m13 - m03 * m11, b05 = m02 * m13 - m03 * m12;
  const b06 = m20 * m31 - m21 * m30, b07 = m20 * m32 - m22 * m30;
  const b08 = m20 * m33 - m23 * m30, b09 = m21 * m32 - m22 * m31;
  const b10 = m21 * m33 - m23 * m31, b11 = m22 * m33 - m23 * m32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-10) return false;
  det = 1 / det;
  out[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
  out[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
  out[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
  out[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
  out[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
  out[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
  out[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
  out[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
  out[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
  out[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
  out[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
  out[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
  out[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
  out[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
  out[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
  out[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
  return true;
}

export function toQuat(m: Float32Array, off: number): [number, number, number, number] {
  const m00 = m[off], m11 = m[off + 5], m22 = m[off + 10];
  const trace = m00 + m11 + m22;
  let x: number, y: number, z: number, w: number;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    w = 0.25 / s; x = (m[off + 6] - m[off + 9]) * s; y = (m[off + 8] - m[off + 2]) * s; z = (m[off + 1] - m[off + 4]) * s;
  } else if (m00 > m11 && m00 > m22) {
    const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
    x = 0.25 * s; w = (m[off + 6] - m[off + 9]) / s; y = (m[off + 1] + m[off + 4]) / s; z = (m[off + 8] + m[off + 2]) / s;
  } else if (m11 > m22) {
    const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
    y = 0.25 * s; w = (m[off + 8] - m[off + 2]) / s; x = (m[off + 1] + m[off + 4]) / s; z = (m[off + 6] + m[off + 9]) / s;
  } else {
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    z = 0.25 * s; w = (m[off + 1] - m[off + 4]) / s; x = (m[off + 8] + m[off + 2]) / s; y = (m[off + 6] + m[off + 9]) / s;
  }
  const len = Math.sqrt(x * x + y * y + z * z + w * w) || 1;
  return [x / len, y / len, z / len, w / len];
}

export function eulerToQuat(rx: number, ry: number, rz: number): [number, number, number, number] {
  const cx = Math.cos(rx * 0.5), sx = Math.sin(rx * 0.5);
  const cy = Math.cos(ry * 0.5), sy = Math.sin(ry * 0.5);
  const cz = Math.cos(rz * 0.5), sz = Math.sin(rz * 0.5);
  const w = cy * cx * cz + sy * sx * sz;
  const x = cy * sx * cz + sy * cx * sz;
  const y = sy * cx * cz - cy * sx * sz;
  const z = cy * cx * sz - sy * sx * cz;
  const len = Math.hypot(x, y, z, w) || 1;
  return [x / len, y / len, z / len, w / len];
}