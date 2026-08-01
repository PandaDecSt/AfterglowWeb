import { RigidbodyType, RigidbodyShape, type Rigidbody, type Joint } from "./physics/types";
import type { PMXRigidbody, PMXJoint } from "../utils/pmx-loader";

export function buildPhysicsRigidbodies(pmxRbs: PMXRigidbody[]): Rigidbody[] {
  return pmxRbs.map((rb) => {
    let type: RigidbodyType;
    let aligned = false;
    switch (rb.type) {
      case 0: type = RigidbodyType.Static; break;
      case 1: type = RigidbodyType.Dynamic; break;
      case 2: type = RigidbodyType.Dynamic; aligned = true; break;
      default: type = RigidbodyType.Static;
    }

    const shape = rb.shape as unknown as RigidbodyShape;
    const size: [number, number, number] = [rb.size[0], rb.size[1], rb.size[2]];
    const shapePosition: [number, number, number] = [rb.position[0], rb.position[1], rb.position[2]];
    const shapeRotation: [number, number, number] = [rb.rotation[0], rb.rotation[1], rb.rotation[2]];

    return {
      name: rb.name,
      englishName: rb.nameEn,
      boneIndex: rb.boneIndex,
      group: rb.group,
      collisionMask: rb.collisionMask,
      shape,
      size,
      shapePosition,
      shapeRotation,
      mass: rb.mass,
      linearDamping: rb.linearDamping,
      angularDamping: rb.angularDamping,
      restitution: rb.restitution,
      friction: rb.friction,
      type,
      aligned,
      bodyOffsetMatrixInverse: new Float32Array(16),
      bodyOffsetMatrix: new Float32Array(16),
    };
  });
}

export function buildPhysicsJoints(pmxJoints: PMXJoint[]): Joint[] {
  return pmxJoints.map((j) => ({
    name: j.name,
    englishName: j.nameEn,
    type: j.type,
    rigidbodyIndexA: j.rigidbodyIndexA,
    rigidbodyIndexB: j.rigidbodyIndexB,
    position: [j.position[0], j.position[1], j.position[2]],
    rotation: [j.rotation[0], j.rotation[1], j.rotation[2]],
    positionMin: [j.positionMin[0], j.positionMin[1], j.positionMin[2]],
    positionMax: [j.positionMax[0], j.positionMax[1], j.positionMax[2]],
    rotationMin: [j.rotationMin[0], j.rotationMin[1], j.rotationMin[2]],
    rotationMax: [j.rotationMax[0], j.rotationMax[1], j.rotationMax[2]],
    springPosition: [j.springPosition[0], j.springPosition[1], j.springPosition[2]],
    springRotation: [j.springRotation[0], j.springRotation[1], j.springRotation[2]],
  }));
}