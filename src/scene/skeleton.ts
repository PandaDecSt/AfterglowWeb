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

    if (inverseBindMatrices) {
      this.inverseBindMatrices.set(inverseBindMatrices);
    } else {
      this.computeInverseBindMatrices();
    }

    this.updateWorldMatrices();
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
    const tmpMat = mat4.create();

    for (let i = 0; i < this.boneCount; i++) {
      const pos = vec3.create(
        this.localPositions[i * 3],
        this.localPositions[i * 3 + 1],
        this.localPositions[i * 3 + 2]
      );
      const rot = quat.create(
        this.localRotations[i * 4],
        this.localRotations[i * 4 + 1],
        this.localRotations[i * 4 + 2],
        this.localRotations[i * 4 + 3]
      );
      const scl = vec3.create(
        this.localScales[i * 3],
        this.localScales[i * 3 + 1],
        this.localScales[i * 3 + 2]
      );

      const localMat = mat4.identity(mat4.create());
      mat4.translate(localMat, pos, localMat);
      mat4.multiply(localMat, mat4.fromQuat(mat4.create(), rot), localMat);
      mat4.scale(localMat, scl, localMat);

      const parentIdx = this.parentIndices[i];
      if (parentIdx >= 0 && parentIdx < i) {
        const parentWorld = this.getWorldMatrix(parentIdx);
        mat4.multiply(parentWorld, localMat, tmpMat);
      } else {
        mat4.copy(localMat, tmpMat);
      }

      this.worldMatrices.set(tmpMat, i * 16);
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
    const tmpMat = mat4.create();
    for (let i = 0; i < this.boneCount; i++) {
      const world = this.getWorldMatrix(i);
      const ibm = this.inverseBindMatrices.subarray(i * 16, i * 16 + 16);
      mat4.multiply(
        world,
        ibm as unknown as Mat4,
        tmpMat
      );
      out.set(tmpMat, i * 16);
    }
  }
}