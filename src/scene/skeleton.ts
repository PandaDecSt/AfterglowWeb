import { mat4, quat, vec3, type Vec3, type Mat4, type Quat } from "wgpu-matrix";

export interface BoneDesc {
  name: string;
  parentIndex: number;
  position: Vec3;
  rotation: Quat;
  scale: Vec3;
}

export class Skeleton {
  boneNames: string[];
  parentIndices: Int16Array;
  boneCount: number;

  localPositions: Float32Array;
  localRotations: Float32Array;
  localScales: Float32Array;
  worldMatrices: Float32Array;
  inverseBindMatrices: Float32Array;

  private _tmpMat = mat4.create();
  private _tmpPos = vec3.create();
  private _tmpRot = quat.create();
  private _tmpScl = vec3.create();
  private _tmpLocal = mat4.create();

  constructor(descs: BoneDesc[], inverseBindMatrices?: Float32Array) {
    this.boneCount = descs.length;
    this.boneNames = descs.map((d) => d.name);
    this.parentIndices = new Int16Array(this.boneCount);

    this.localPositions = new Float32Array(this.boneCount * 3);
    this.localRotations = new Float32Array(this.boneCount * 4);
    this.localScales = new Float32Array(this.boneCount * 3);
    this.worldMatrices = new Float32Array(this.boneCount * 16);
    this.inverseBindMatrices = new Float32Array(this.boneCount * 16);

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
    }

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
    for (let i = 0; i < this.boneCount; i++) {
      const off3 = i * 3;
      const off4 = i * 4;
      this._tmpPos[0] = this.localPositions[off3];
      this._tmpPos[1] = this.localPositions[off3 + 1];
      this._tmpPos[2] = this.localPositions[off3 + 2];
      this._tmpRot[0] = this.localRotations[off4];
      this._tmpRot[1] = this.localRotations[off4 + 1];
      this._tmpRot[2] = this.localRotations[off4 + 2];
      this._tmpRot[3] = this.localRotations[off4 + 3];
      this._tmpScl[0] = this.localScales[off3];
      this._tmpScl[1] = this.localScales[off3 + 1];
      this._tmpScl[2] = this.localScales[off3 + 2];

      mat4.identity(this._tmpLocal);
      mat4.translate(this._tmpLocal, this._tmpPos, this._tmpLocal);
      // 这个注释代码是错误的：mat4.multiply(this._tmpLocal, mat4.fromQuat(this._tmpMat, this._tmpRot), this._tmpLocal);
      mat4.multiply(this._tmpLocal, mat4.fromQuat(this._tmpRot, this._tmpMat), this._tmpLocal);
      mat4.scale(this._tmpLocal, this._tmpScl, this._tmpLocal);

      const parentIdx = this.parentIndices[i];
      if (parentIdx >= 0 && parentIdx < i) {
        const parentWorld = this.getWorldMatrix(parentIdx);
        mat4.multiply(parentWorld, this._tmpLocal, this._tmpMat);
      } else {
        mat4.copy(this._tmpLocal, this._tmpMat);
      }

      this.worldMatrices.set(this._tmpMat, i * 16);
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