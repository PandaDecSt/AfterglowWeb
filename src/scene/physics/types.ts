export enum RigidbodyShape {
  Sphere = 0,
  Box = 1,
  Capsule = 2,
}

export enum RigidbodyType {
  Static = 0,
  Dynamic = 1,
  Kinematic = 2,
}

export interface Rigidbody {
  name: string;
  englishName: string;
  boneIndex: number;
  group: number;
  collisionMask: number;
  shape: RigidbodyShape;
  size: [number, number, number];
  shapePosition: [number, number, number];
  shapeRotation: [number, number, number];
  mass: number;
  linearDamping: number;
  angularDamping: number;
  restitution: number;
  friction: number;
  type: RigidbodyType;
  aligned: boolean;
  bodyOffsetMatrixInverse: Float32Array;
  bodyOffsetMatrix: Float32Array;
}

export interface Joint {
  name: string;
  englishName: string;
  type: number;
  rigidbodyIndexA: number;
  rigidbodyIndexB: number;
  position: [number, number, number];
  rotation: [number, number, number];
  positionMin: [number, number, number];
  positionMax: [number, number, number];
  rotationMin: [number, number, number];
  rotationMax: [number, number, number];
  springPosition: [number, number, number];
  springRotation: [number, number, number];
}