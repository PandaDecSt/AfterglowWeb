import { mat4, quat, vec3, type Vec3, type Mat4, type Quat } from "wgpu-matrix";

export interface BoneDesc {
  name: string;
  parentIndex: number;
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
  appendParentIndex?: number;
  appendRatio?: number;
  appendRotate?: boolean;
  appendMove?: boolean;
}

export class Skeleton {
  boneNames: string[];
  parentIndices: Int16Array;
  boneCount: number;

  localPositions: Float32Array;
  localRotations: Float32Array;
  localScales: Float32Array;
  worldMatrices: Float32Array;
  worldRotations: Float32Array;
  inverseBindMatrices: Float32Array;

  appendParentIndices: Int16Array;
  appendRatios: Float32Array;
  appendFlags: Uint8Array;
  bindPositions: Float32Array;

  private _tmpMat = mat4.create();
  private _tmpLocal = mat4.create();
  private _slerpQ0 = quat.create();
  private _slerpQ1 = quat.create();
  private _slerpQ2 = quat.create();

  constructor(descs: BoneDesc[], inverseBindMatrices?: Float32Array) {
    this.boneCount = descs.length;
    this.boneNames = descs.map((d) => d.name);
    this.parentIndices = new Int16Array(this.boneCount);

    this.localPositions = new Float32Array(this.boneCount * 3);
    this.localRotations = new Float32Array(this.boneCount * 4);
    this.localScales = new Float32Array(this.boneCount * 3);
    this.worldMatrices = new Float32Array(this.boneCount * 16);
    this.worldRotations = new Float32Array(this.boneCount * 4);
    this.inverseBindMatrices = new Float32Array(this.boneCount * 16);

    this.appendParentIndices = new Int16Array(this.boneCount);
    this.appendRatios = new Float32Array(this.boneCount);
    this.appendFlags = new Uint8Array(this.boneCount);
    this.bindPositions = new Float32Array(this.boneCount * 3);

    for (let i = 0; i < this.boneCount; i++) {
      const d = descs[i];
      this.parentIndices[i] = d.parentIndex;

      this.localPositions[i * 3] = d.position[0];
      this.localPositions[i * 3 + 1] = d.position[1];
      this.localPositions[i * 3 + 2] = d.position[2];

      this.localRotations[i * 4] = d.rotation[0];
      this.localRotations[i * 4 + 1] = d.rotation[1];
      this.localRotations[i * 4 + 2] = d.rotation[2];
      this.localRotations[i * 4 + 3] = d.rotation[3];

      this.localScales[i * 3] = d.scale[0];
      this.localScales[i * 3 + 1] = d.scale[1];
      this.localScales[i * 3 + 2] = d.scale[2];

      this.appendParentIndices[i] = d.appendParentIndex ?? -1;
      this.appendRatios[i] = d.appendRatio ?? 0;
      let flags = 0;
      if (d.appendRotate) flags |= 1;
      if (d.appendMove) flags |= 2;
      this.appendFlags[i] = flags;
    }

    this.bindPositions.set(this.localPositions);

    this.updateWorldMatrices();

    if (inverseBindMatrices) {
      this.inverseBindMatrices.set(inverseBindMatrices);
    } else {
      this.computeInverseBindMatrices();
    }
  }

  getBoneIndex(name: string): number {
    return this.boneNames.indexOf(name);
  }

  getBoneWorldPosition(index: number, out?: Float32Array): Float32Array {
    const o = index * 16;
    const result = out ?? new Float32Array(3);
    result[0] = this.worldMatrices[o + 12];
    result[1] = this.worldMatrices[o + 13];
    result[2] = this.worldMatrices[o + 14];
    return result;
  }

  setLocalPosition(index: number, x: number, y: number, z: number): void {
    this.localPositions[index * 3] = x;
    this.localPositions[index * 3 + 1] = y;
    this.localPositions[index * 3 + 2] = z;
  }

  setLocalRotation(index: number, x: number, y: number, z: number, w: number): void {
    this.localRotations[index * 4] = x;
    this.localRotations[index * 4 + 1] = y;
    this.localRotations[index * 4 + 2] = z;
    this.localRotations[index * 4 + 3] = w;
  }

  getWorldMatrix(index: number): Mat4 {
    return this.worldMatrices.subarray(index * 16, index * 16 + 16) as unknown as Mat4;
  }

  updateWorldMatrices(): void {
    const lr = this.localRotations;
    const lp = this.localPositions;
    const bp = this.bindPositions;

    for (let i = 0; i < this.boneCount; i++) {
      const off3 = i * 3;
      const off4 = i * 4;

      let rx = lr[off4];
      let ry = lr[off4 + 1];
      let rz = lr[off4 + 2];
      let rw = lr[off4 + 3];

      let addTx = 0, addTy = 0, addTz = 0;

      const aFlags = this.appendFlags[i];
      if (aFlags !== 0) {
        const aParent = this.appendParentIndices[i];
        if (aParent >= 0 && aParent < this.boneCount) {
          const ratio = this.appendRatios[i];
          const absRatio = Math.abs(ratio);
          if (absRatio > 1e-6) {
            if (aFlags & 1) {
              let ax = lr[aParent * 4];
              let ay = lr[aParent * 4 + 1];
              let az = lr[aParent * 4 + 2];
              const aw = lr[aParent * 4 + 3];
              if (ratio < 0) { ax = -ax; ay = -ay; az = -az; }

              this._slerpQ0[0] = 0; this._slerpQ0[1] = 0; this._slerpQ0[2] = 0; this._slerpQ0[3] = 1;
              this._slerpQ1[0] = ax; this._slerpQ1[1] = ay; this._slerpQ1[2] = az; this._slerpQ1[3] = aw;
              quat.slerp(this._slerpQ0, this._slerpQ1, absRatio, this._slerpQ2);

              const sx = this._slerpQ2[0], sy = this._slerpQ2[1], sz = this._slerpQ2[2], sw = this._slerpQ2[3];
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

      const px = lp[off3] + addTx;
      const py = lp[off3 + 1] + addTy;
      const pz = lp[off3 + 2] + addTz;
      const sx = this.localScales[off3];
      const sy = this.localScales[off3 + 1];
      const sz = this.localScales[off3 + 2];

      const x2 = rx + rx, y2 = ry + ry, z2 = rz + rz;
      const xx = rx * x2, xy = rx * y2, xz = rx * z2;
      const yy = ry * y2, yz = ry * z2, zz = rz * z2;
      const wx = rw * x2, wy = rw * y2, wz = rw * z2;

      const l = this._tmpLocal;
      l[0] = (1 - (yy + zz)) * sx; l[1] = (xy + wz) * sx;       l[2] = (xz - wy) * sx;       l[3] = 0;
      l[4] = (xy - wz) * sy;       l[5] = (1 - (xx + zz)) * sy; l[6] = (yz + wx) * sy;       l[7] = 0;
      l[8] = (xz + wy) * sz;       l[9] = (yz - wx) * sz;       l[10] = (1 - (xx + yy)) * sz; l[11] = 0;
      l[12] = px;                  l[13] = py;                  l[14] = pz;                  l[15] = 1;

      const parentIdx = this.parentIndices[i];
      const wOff = i * 16;
      if (parentIdx >= 0 && parentIdx < i) {
        const pOff = parentIdx * 16;
        const p = this.worldMatrices;
        const wm = this.worldMatrices;
        wm[wOff + 0] = p[pOff] * l[0] + p[pOff + 4] * l[1] + p[pOff + 8] * l[2] + p[pOff + 12] * l[3];
        wm[wOff + 1] = p[pOff + 1] * l[0] + p[pOff + 5] * l[1] + p[pOff + 9] * l[2] + p[pOff + 13] * l[3];
        wm[wOff + 2] = p[pOff + 2] * l[0] + p[pOff + 6] * l[1] + p[pOff + 10] * l[2] + p[pOff + 14] * l[3];
        wm[wOff + 3] = p[pOff + 3] * l[0] + p[pOff + 7] * l[1] + p[pOff + 11] * l[2] + p[pOff + 15] * l[3];
        wm[wOff + 4] = p[pOff] * l[4] + p[pOff + 4] * l[5] + p[pOff + 8] * l[6] + p[pOff + 12] * l[7];
        wm[wOff + 5] = p[pOff + 1] * l[4] + p[pOff + 5] * l[5] + p[pOff + 9] * l[6] + p[pOff + 13] * l[7];
        wm[wOff + 6] = p[pOff + 2] * l[4] + p[pOff + 6] * l[5] + p[pOff + 10] * l[6] + p[pOff + 14] * l[7];
        wm[wOff + 7] = p[pOff + 3] * l[4] + p[pOff + 7] * l[5] + p[pOff + 11] * l[6] + p[pOff + 15] * l[7];
        wm[wOff + 8] = p[pOff] * l[8] + p[pOff + 4] * l[9] + p[pOff + 8] * l[10] + p[pOff + 12] * l[11];
        wm[wOff + 9] = p[pOff + 1] * l[8] + p[pOff + 5] * l[9] + p[pOff + 9] * l[10] + p[pOff + 13] * l[11];
        wm[wOff + 10] = p[pOff + 2] * l[8] + p[pOff + 6] * l[9] + p[pOff + 10] * l[10] + p[pOff + 14] * l[11];
        wm[wOff + 11] = p[pOff + 3] * l[8] + p[pOff + 7] * l[9] + p[pOff + 11] * l[10] + p[pOff + 15] * l[11];
        wm[wOff + 12] = p[pOff] * l[12] + p[pOff + 4] * l[13] + p[pOff + 8] * l[14] + p[pOff + 12] * l[15];
        wm[wOff + 13] = p[pOff + 1] * l[12] + p[pOff + 5] * l[13] + p[pOff + 9] * l[14] + p[pOff + 13] * l[15];
        wm[wOff + 14] = p[pOff + 2] * l[12] + p[pOff + 6] * l[13] + p[pOff + 10] * l[14] + p[pOff + 14] * l[15];
        wm[wOff + 15] = p[pOff + 3] * l[12] + p[pOff + 7] * l[13] + p[pOff + 11] * l[14] + p[pOff + 15] * l[15];
      } else {
        this.worldMatrices.set(l, wOff);
      }

      const wr = this.worldRotations;
      if (parentIdx >= 0 && parentIdx < i) {
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
  }

  computeInverseBindMatrices(): void {
    for (let i = 0; i < this.boneCount; i++) {
      const world = this.getWorldMatrix(i);
      const inv = mat4.inverse(world);
      this.inverseBindMatrices.set(inv as unknown as ArrayLike<number>, i * 16);
    }
  }

  computeSkinMatrices(out: Float32Array): void {
    for (let i = 0; i < this.boneCount; i++) {
      const world = this.getWorldMatrix(i);
      const ibm = this.inverseBindMatrices.subarray(i * 16, i * 16 + 16);
      mat4.multiply(world, ibm as unknown as Mat4, this._tmpMat);
      out.set(this._tmpMat, i * 16);
    }
  }
}
