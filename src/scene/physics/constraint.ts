import type { Joint, Rigidbody } from "./types";
import { RigidbodyType } from "./types";
import { eulerToQuat, fromPositionRotation, inverseInto, mulArrays } from "./body";

export interface SixDofSpringConstraint {
  bodyA: number;
  bodyB: number;
  frameA: Float32Array;
  frameB: Float32Array;
  linearMin: Float32Array;
  linearMax: Float32Array;
  angularMin: Float32Array;
  angularMax: Float32Array;
  springEnabled: Uint8Array;
  springStiffness: Float32Array;
  equilibriumPoint: Float32Array;
  isLoop: boolean;
  cacheSkip: boolean;
  cacheLeverA: Float32Array;
  cacheLeverB: Float32Array;
  cacheLinAxes: Float32Array;
  cacheLinCrossA: Float32Array;
  cacheLinCrossB: Float32Array;
  cacheLinJacInv: Float32Array;
  cacheLinTargetVel: Float32Array;
  cacheLinActive: Uint8Array;
  cacheLinLimitImp: Float32Array;
  cacheLinSpringTarget: Float32Array;
  cacheLinSpringMaxImp: Float32Array;
  cacheLinSpringImp: Float32Array;
  cacheLinSpringActive: Uint8Array;
  cacheAngAxes: Float32Array;
  cacheAngTargetVel: Float32Array;
  cacheAngActive: Uint8Array;
  cacheAngJacInv: Float32Array;
  cacheAngSpringMaxImp: Float32Array;
  cacheAngSpringImp: Float32Array;
  cacheAngWA: Float32Array;
  cacheAngWB: Float32Array;
  cacheAngLimAxis: Float32Array;
  cacheAngLimWA: Float32Array;
  cacheAngLimWB: Float32Array;
  cacheAngLimJacInv: number;
  cacheAngLimTarget: number;
  cacheAngLimActive: number;
  cacheAngLimImp: number;
  cacheAngPATarget: Float32Array;
  cacheAngPAActive: Uint8Array;
  cacheAngPAImp: Float32Array;
}

export const STOP_ERP = 0.45;

export function buildConstraints(
  rigidbodies: Rigidbody[],
  joints: Joint[],
): SixDofSpringConstraint[] {
  const out: SixDofSpringConstraint[] = [];
  const jointWorld = new Float32Array(16);
  const bodyWorld = new Float32Array(16);
  const bodyInv = new Float32Array(16);

  for (let j = 0; j < joints.length; j++) {
    const joint = joints[j];
    const a = joint.rigidbodyIndexA;
    const b = joint.rigidbodyIndexB;
    if (a < 0 || b < 0 || a >= rigidbodies.length || b >= rigidbodies.length) continue;
    if (a === b) continue;
    const rbA = rigidbodies[a];
    const rbB = rigidbodies[b];

    const jq = eulerToQuat(joint.rotation[0], joint.rotation[1], joint.rotation[2]);
    fromPositionRotation(
      joint.position[0], joint.position[1], joint.position[2],
      jq[0], jq[1], jq[2], jq[3],
      jointWorld,
    );

    const frameA = new Float32Array(16);
    const frameB = new Float32Array(16);
    if (!buildLocalFrame(rbA, jointWorld, bodyWorld, bodyInv, frameA)) continue;
    if (!buildLocalFrame(rbB, jointWorld, bodyWorld, bodyInv, frameB)) continue;

    const linearMin = new Float32Array([joint.positionMin[0], joint.positionMin[1], joint.positionMin[2]]);
    const linearMax = new Float32Array([joint.positionMax[0], joint.positionMax[1], joint.positionMax[2]]);
    const angularMin = new Float32Array([
      normalizeAngle(joint.rotationMin[0]),
      normalizeAngle(joint.rotationMin[1]),
      normalizeAngle(joint.rotationMin[2]),
    ]);
    const angularMax = new Float32Array([
      normalizeAngle(joint.rotationMax[0]),
      normalizeAngle(joint.rotationMax[1]),
      normalizeAngle(joint.rotationMax[2]),
    ]);

    const springEnabled = new Uint8Array(6);
    const springStiffness = new Float32Array(6);
    springStiffness[0] = joint.springPosition[0];
    springStiffness[1] = joint.springPosition[1];
    springStiffness[2] = joint.springPosition[2];
    springStiffness[3] = joint.springRotation[0];
    springStiffness[4] = joint.springRotation[1];
    springStiffness[5] = joint.springRotation[2];
    for (let i = 0; i < 6; i++) springEnabled[i] = springStiffness[i] !== 0 ? 1 : 0;

    out.push({
      bodyA: a,
      bodyB: b,
      frameA,
      frameB,
      linearMin,
      linearMax,
      angularMin,
      angularMax,
      springEnabled,
      springStiffness,
      equilibriumPoint: new Float32Array(6),
      isLoop: false,
      cacheSkip: false,
      cacheLeverA: new Float32Array(3),
      cacheLeverB: new Float32Array(3),
      cacheLinAxes: new Float32Array(9),
      cacheLinCrossA: new Float32Array(9),
      cacheLinCrossB: new Float32Array(9),
      cacheLinJacInv: new Float32Array(3),
      cacheLinTargetVel: new Float32Array(3),
      cacheLinActive: new Uint8Array(3),
      cacheLinLimitImp: new Float32Array(3),
      cacheLinSpringTarget: new Float32Array(3),
      cacheLinSpringMaxImp: new Float32Array(3),
      cacheLinSpringImp: new Float32Array(3),
      cacheLinSpringActive: new Uint8Array(3),
      cacheAngAxes: new Float32Array(9),
      cacheAngTargetVel: new Float32Array(3),
      cacheAngActive: new Uint8Array(3),
      cacheAngJacInv: new Float32Array(3),
      cacheAngSpringMaxImp: new Float32Array(3),
      cacheAngSpringImp: new Float32Array(3),
      cacheAngWA: new Float32Array(9),
      cacheAngWB: new Float32Array(9),
      cacheAngLimAxis: new Float32Array(3),
      cacheAngLimWA: new Float32Array(3),
      cacheAngLimWB: new Float32Array(3),
      cacheAngLimJacInv: 0,
      cacheAngLimTarget: 0,
      cacheAngLimActive: 0,
      cacheAngLimImp: 0,
      cacheAngPATarget: new Float32Array(3),
      cacheAngPAActive: new Uint8Array(3),
      cacheAngPAImp: new Float32Array(3),
    });
  }

  const parent = new Int32Array(rigidbodies.length + 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const world = rigidbodies.length;
  for (let i = 0; i < rigidbodies.length; i++) {
    if (rigidbodies[i].type !== RigidbodyType.Dynamic) parent[find(i)] = find(world);
  }
  for (const con of out) {
    const ra = find(con.bodyA);
    const rb = find(con.bodyB);
    if (ra === rb) con.isLoop = true;
    else parent[ra] = rb;
  }

  return out;
}

function buildLocalFrame(
  rb: Rigidbody,
  jointWorld: Float32Array,
  bodyWorld: Float32Array,
  bodyInv: Float32Array,
  out: Float32Array,
): boolean {
  const q = eulerToQuat(rb.shapeRotation[0], rb.shapeRotation[1], rb.shapeRotation[2]);
  fromPositionRotation(
    rb.shapePosition[0], rb.shapePosition[1], rb.shapePosition[2],
    q[0], q[1], q[2], q[3],
    bodyWorld,
  );
  if (!inverseInto(bodyWorld, bodyInv)) return false;
  mulArrays(bodyInv, 0, jointWorld, 0, out, 0);
  return true;
}

function normalizeAngle(a: number): number {
  const twoPi = Math.PI * 2;
  a = a % twoPi;
  if (a < -Math.PI) a += twoPi;
  else if (a > Math.PI) a -= twoPi;
  return a;
}