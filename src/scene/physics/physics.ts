import type { Rigidbody, Joint } from "./types";
import { RigidbodyType } from "./types";
import { RigidBodyStore, fromPositionRotation, mulArrays, toQuat } from "./body";
import { World } from "./world";
import { buildConstraints, type SixDofSpringConstraint } from "./constraint";
import { ContactPool } from "./contact";

const _bodyMat = new Float32Array(16);
const _boneMat = new Float32Array(16);

export class MMDPhysics {
  private rigidbodies: Rigidbody[];
  private joints: Joint[];
  private store: RigidBodyStore;
  private world: World;
  private constraints: SixDofSpringConstraint[];
  private contacts: ContactPool;
  private firstFrame = true;
  private timeAccum = 0;
  private readonly fixedTimeStep = 1 / 60;
  private readonly maxSubSteps = 6;
  private prevPositions: Float32Array;
  private prevOrientations: Float32Array;
  private kinTargetPos: Float32Array;
  private kinTargetOri: Float32Array;
  private kinTargetVel: Float32Array;
  private kinTargetAngVel: Float32Array;
  private kinRoot: Int32Array;
  private teleportFlags: Uint8Array;
  private alignPinned: Uint8Array;
  teleportCount = 0;

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies;
    this.joints = joints;
    this.store = new RigidBodyStore(rigidbodies);
    this.world = new World(0, -98, 0);
    this.constraints = buildConstraints(rigidbodies, joints);
    this.contacts = new ContactPool();
    this.prevPositions = new Float32Array(this.store.count * 3);
    this.prevOrientations = new Float32Array(this.store.count * 4);
    this.kinTargetPos = new Float32Array(this.store.count * 3);
    this.kinTargetOri = new Float32Array(this.store.count * 4);
    this.kinTargetVel = new Float32Array(this.store.count * 3);
    this.kinTargetAngVel = new Float32Array(this.store.count * 3);
    this.kinRoot = this.buildKinematicRoots();
    this.teleportFlags = new Uint8Array(this.store.count);
    this.alignPinned = new Uint8Array(this.store.count);
    const types = this.store.type;
    for (const c of this.constraints) {
      const tA = types[c.bodyA];
      const tB = types[c.bodyB];
      const aFollows = tA === RigidbodyType.Static || tA === RigidbodyType.Kinematic;
      const bFollows = tB === RigidbodyType.Static || tB === RigidbodyType.Kinematic;
      if (aFollows && this.store.aligned[c.bodyB]) this.alignPinned[c.bodyB] = 1;
      if (bFollows && this.store.aligned[c.bodyA]) this.alignPinned[c.bodyA] = 1;
    }
  }

  private buildKinematicRoots(): Int32Array {
    const N = this.store.count;
    const root = new Int32Array(N).fill(-1);
    const types = this.store.type;
    const adj: number[][] = Array.from({ length: N }, () => []);
    for (const c of this.constraints) {
      adj[c.bodyA].push(c.bodyB);
      adj[c.bodyB].push(c.bodyA);
    }
    const queue: number[] = [];
    for (let i = 0; i < N; i++) {
      if (types[i] === RigidbodyType.Static || types[i] === RigidbodyType.Kinematic) {
        root[i] = i;
        queue.push(i);
      }
    }
    for (let h = 0; h < queue.length; h++) {
      const i = queue[h];
      for (const j of adj[i]) {
        if (root[j] !== -1) continue;
        root[j] = root[i];
        queue.push(j);
      }
    }
    return root;
  }

  private savePrevState(): void {
    this.prevPositions.set(this.store.positions);
    this.prevOrientations.set(this.store.orientations);
  }

  setGravity(gx: number, gy: number, gz: number): void {
    this.world.setGravity(gx, gy, gz);
  }

  getGravity(): [number, number, number] {
    return [this.world.gravityX, this.world.gravityY, this.world.gravityZ];
  }

  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies;
  }

  getJoints(): Joint[] {
    return this.joints;
  }

  getStore(): RigidBodyStore {
    return this.store;
  }

  reset(boneWorldMatrices: Float32Array): void {
    if (this.firstFrame) return;
    this.snapBodiesToBones(boneWorldMatrices);
    this.savePrevState();
    this.kinTargetPos.set(this.store.positions);
    this.kinTargetOri.set(this.store.orientations);
    this.kinTargetVel.fill(0);
    this.kinTargetAngVel.fill(0);
    this.timeAccum = 0;
  }

  step(dt: number, boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array): void {
    if (this.firstFrame) {
      this.store.computeBoneOffsets(boneInverseBindMatrices);
      this.snapBodiesToBones(boneWorldMatrices);
      this.savePrevState();
      this.kinTargetPos.set(this.store.positions);
      this.kinTargetOri.set(this.store.orientations);
      this.firstFrame = false;
    }
    if (this.computeKinematicTargets(boneWorldMatrices, dt)) {
      this.teleportCount++;
      this.carryDynamicThroughTeleport();
      this.snapKinematicToTargets(true);
      this.savePrevState();
    }
    this.timeAccum += dt;
    let nSub = Math.floor(this.timeAccum / this.fixedTimeStep);
    if (nSub > this.maxSubSteps) nSub = this.maxSubSteps;
    for (let k = 0; k < nSub; k++) {
      this.savePrevState();
      this.advanceKinematicToTargets(1 / (nSub - k));
      this.world.step(this.store, this.constraints, this.contacts, this.fixedTimeStep);
      this.restoreNonFiniteBodies();
      this.timeAccum -= this.fixedTimeStep;
    }
    if (this.timeAccum >= this.fixedTimeStep) {
      this.timeAccum = 0;
      this.snapKinematicToTargets(false);
    }
    this.alignPinnedBodiesToBones(boneWorldMatrices);
    const alpha = this.fixedTimeStep > 0 ? this.timeAccum / this.fixedTimeStep : 0;
    this.applyDynamicsToBones(boneWorldMatrices, alpha);
  }

  private snapBodiesToBones(boneWorldMatrices: Float32Array): void {
    const N = this.store.count;
    const offsets = this.store.bodyOffsetMatrix;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const lv = this.store.linearVelocities;
    const av = this.store.angularVelocities;
    const boneIdx = this.store.boneIndex;
    for (let i = 0; i < N; i++) {
      const b = boneIdx[i];
      if (b < 0) continue;
      const boneOff = b * 16;
      if (boneOff + 15 >= boneWorldMatrices.length) continue;
      mulArrays(boneWorldMatrices, boneOff, offsets, i * 16, _bodyMat, 0);
      const i3 = i * 3;
      const i4 = i * 4;
      positions[i3 + 0] = _bodyMat[12];
      positions[i3 + 1] = _bodyMat[13];
      positions[i3 + 2] = _bodyMat[14];
      const q = toQuat(_bodyMat, 0);
      orientations[i4 + 0] = q[0];
      orientations[i4 + 1] = q[1];
      orientations[i4 + 2] = q[2];
      orientations[i4 + 3] = q[3];
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0;
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0;
    }
  }

  private computeKinematicTargets(boneWorldMatrices: Float32Array, dt: number): boolean {
    const N = this.store.count;
    const offsets = this.store.bodyOffsetMatrix;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const types = this.store.type;
    const boneIdx = this.store.boneIndex;
    const tp = this.kinTargetPos;
    const to = this.kinTargetOri;
    const maxJump = Math.max(4, 250 * dt);
    const maxJumpSq = maxJump * maxJump;
    const flags = this.teleportFlags;
    let teleport = false;
    const tv = this.kinTargetVel;
    const tav = this.kinTargetAngVel;
    const invDt = dt > 0 ? 1 / dt : 0;
    for (let i = 0; i < N; i++) {
      flags[i] = 0;
      const t = types[i];
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue;
      const b = boneIdx[i];
      if (b < 0) continue;
      const boneOff = b * 16;
      if (boneOff + 15 >= boneWorldMatrices.length) continue;
      mulArrays(boneWorldMatrices, boneOff, offsets, i * 16, _bodyMat, 0);
      const i3 = i * 3;
      const i4 = i * 4;
      const oldTx = tp[i3 + 0], oldTy = tp[i3 + 1], oldTz = tp[i3 + 2];
      const oldOx = to[i4 + 0], oldOy = to[i4 + 1], oldOz = to[i4 + 2], oldOw = to[i4 + 3];
      tp[i3 + 0] = _bodyMat[12];
      tp[i3 + 1] = _bodyMat[13];
      tp[i3 + 2] = _bodyMat[14];
      const nq = toQuat(_bodyMat, 0);
      const nOx = nq[0], nOy = nq[1], nOz = nq[2], nOw = nq[3];
      to[i4 + 0] = nOx;
      to[i4 + 1] = nOy;
      to[i4 + 2] = nOz;
      to[i4 + 3] = nOw;
      const dx = tp[i3 + 0] - positions[i3 + 0];
      const dy = tp[i3 + 1] - positions[i3 + 1];
      const dz = tp[i3 + 2] - positions[i3 + 2];
      const dot =
        to[i4 + 0] * orientations[i4 + 0] +
        to[i4 + 1] * orientations[i4 + 1] +
        to[i4 + 2] * orientations[i4 + 2] +
        to[i4 + 3] * orientations[i4 + 3];
      if (dx * dx + dy * dy + dz * dz > maxJumpSq || Math.abs(dot) < 0.7071) {
        flags[i] = 1;
        teleport = true;
        tv[i3 + 0] = 0; tv[i3 + 1] = 0; tv[i3 + 2] = 0;
        tav[i3 + 0] = 0; tav[i3 + 1] = 0; tav[i3 + 2] = 0;
        continue;
      }
      tv[i3 + 0] = (tp[i3 + 0] - oldTx) * invDt;
      tv[i3 + 1] = (tp[i3 + 1] - oldTy) * invDt;
      tv[i3 + 2] = (tp[i3 + 2] - oldTz) * invDt;
      const cox = -oldOx, coy = -oldOy, coz = -oldOz, cow = oldOw;
      const qdx = nOw * cox + nOx * cow + nOy * coz - nOz * coy;
      const qdy = nOw * coy - nOx * coz + nOy * cow + nOz * cox;
      const qdz = nOw * coz + nOx * coy - nOy * cox + nOz * cow;
      const qdw = nOw * cow - nOx * cox - nOy * coy - nOz * coz;
      const sign = qdw < 0 ? -1 : 1;
      tav[i3 + 0] = 2 * sign * qdx * invDt;
      tav[i3 + 1] = 2 * sign * qdy * invDt;
      tav[i3 + 2] = 2 * sign * qdz * invDt;
    }
    return teleport;
  }

  private advanceKinematicToTargets(f: number): void {
    const N = this.store.count;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const lv = this.store.linearVelocities;
    const av = this.store.angularVelocities;
    const types = this.store.type;
    const boneIdx = this.store.boneIndex;
    const tp = this.kinTargetPos;
    const to = this.kinTargetOri;
    const tv = this.kinTargetVel;
    const tav = this.kinTargetAngVel;
    for (let i = 0; i < N; i++) {
      const t = types[i];
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue;
      if (boneIdx[i] < 0) continue;
      const i3 = i * 3;
      const i4 = i * 4;
      positions[i3 + 0] += (tp[i3 + 0] - positions[i3 + 0]) * f;
      positions[i3 + 1] += (tp[i3 + 1] - positions[i3 + 1]) * f;
      positions[i3 + 2] += (tp[i3 + 2] - positions[i3 + 2]) * f;
      lv[i3 + 0] = tv[i3 + 0];
      lv[i3 + 1] = tv[i3 + 1];
      lv[i3 + 2] = tv[i3 + 2];
      av[i3 + 0] = tav[i3 + 0];
      av[i3 + 1] = tav[i3 + 1];
      av[i3 + 2] = tav[i3 + 2];
      const oldOx = orientations[i4 + 0], oldOy = orientations[i4 + 1];
      const oldOz = orientations[i4 + 2], oldOw = orientations[i4 + 3];
      let tx = to[i4 + 0], ty = to[i4 + 1], tz = to[i4 + 2], tw = to[i4 + 3];
      if (oldOx * tx + oldOy * ty + oldOz * tz + oldOw * tw < 0) {
        tx = -tx; ty = -ty; tz = -tz; tw = -tw;
      }
      let newOx = oldOx + (tx - oldOx) * f;
      let newOy = oldOy + (ty - oldOy) * f;
      let newOz = oldOz + (tz - oldOz) * f;
      let newOw = oldOw + (tw - oldOw) * f;
      const len2 = newOx * newOx + newOy * newOy + newOz * newOz + newOw * newOw;
      if (len2 > 1e-12) {
        const inv = 1 / Math.sqrt(len2);
        newOx *= inv; newOy *= inv; newOz *= inv; newOw *= inv;
      } else {
        newOx = tx; newOy = ty; newOz = tz; newOw = tw;
      }
      orientations[i4 + 0] = newOx;
      orientations[i4 + 1] = newOy;
      orientations[i4 + 2] = newOz;
      orientations[i4 + 3] = newOw;
    }
  }

  private snapKinematicToTargets(onlyFlagged: boolean): void {
    const N = this.store.count;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const lv = this.store.linearVelocities;
    const av = this.store.angularVelocities;
    const types = this.store.type;
    const boneIdx = this.store.boneIndex;
    const tp = this.kinTargetPos;
    const to = this.kinTargetOri;
    const flags = this.teleportFlags;
    for (let i = 0; i < N; i++) {
      const t = types[i];
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue;
      if (boneIdx[i] < 0) continue;
      if (onlyFlagged && !flags[i]) continue;
      const i3 = i * 3;
      const i4 = i * 4;
      positions[i3 + 0] = tp[i3 + 0];
      positions[i3 + 1] = tp[i3 + 1];
      positions[i3 + 2] = tp[i3 + 2];
      orientations[i4 + 0] = to[i4 + 0];
      orientations[i4 + 1] = to[i4 + 1];
      orientations[i4 + 2] = to[i4 + 2];
      orientations[i4 + 3] = to[i4 + 3];
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0;
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0;
    }
  }

  private carryDynamicThroughTeleport(): void {
    const N = this.store.count;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const lv = this.store.linearVelocities;
    const av = this.store.angularVelocities;
    const types = this.store.type;
    const tp = this.kinTargetPos;
    const to = this.kinTargetOri;
    const root = this.kinRoot;
    const flags = this.teleportFlags;
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic) continue;
      const k = root[i];
      if (k < 0 || !flags[k] || this.store.boneIndex[k] < 0) continue;
      const k3 = k * 3;
      const k4 = k * 4;
      const cx = -orientations[k4 + 0], cy = -orientations[k4 + 1], cz = -orientations[k4 + 2], cw = orientations[k4 + 3];
      const txq = to[k4 + 0], tyq = to[k4 + 1], tzq = to[k4 + 2], twq = to[k4 + 3];
      const rx = twq * cx + txq * cw + tyq * cz - tzq * cy;
      const ry = twq * cy - txq * cz + tyq * cw + tzq * cx;
      const rz = twq * cz + txq * cy - tyq * cx + tzq * cw;
      const rw = twq * cw - txq * cx - tyq * cy - tzq * cz;
      const i3 = i * 3;
      const i4 = i * 4;
      const ox = positions[i3 + 0] - positions[k3 + 0];
      const oy = positions[i3 + 1] - positions[k3 + 1];
      const oz = positions[i3 + 2] - positions[k3 + 2];
      const c1x = ry * oz - rz * oy;
      const c1y = rz * ox - rx * oz;
      const c1z = rx * oy - ry * ox;
      const c2x = ry * c1z - rz * c1y;
      const c2y = rz * c1x - rx * c1z;
      const c2z = rx * c1y - ry * c1x;
      positions[i3 + 0] = tp[k3 + 0] + ox + 2 * (rw * c1x + c2x);
      positions[i3 + 1] = tp[k3 + 1] + oy + 2 * (rw * c1y + c2y);
      positions[i3 + 2] = tp[k3 + 2] + oz + 2 * (rw * c1z + c2z);
      const qx = orientations[i4 + 0], qy = orientations[i4 + 1], qz = orientations[i4 + 2], qw = orientations[i4 + 3];
      let nx = rw * qx + rx * qw + ry * qz - rz * qy;
      let ny = rw * qy - rx * qz + ry * qw + rz * qx;
      let nz = rw * qz + rx * qy - ry * qx + rz * qw;
      let nw = rw * qw - rx * qx - ry * qy - rz * qz;
      const len2 = nx * nx + ny * ny + nz * nz + nw * nw;
      if (len2 > 1e-12) {
        const inv = 1 / Math.sqrt(len2);
        nx *= inv; ny *= inv; nz *= inv; nw *= inv;
        orientations[i4 + 0] = nx;
        orientations[i4 + 1] = ny;
        orientations[i4 + 2] = nz;
        orientations[i4 + 3] = nw;
      }
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0;
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0;
    }
  }

  private alignPinnedBodiesToBones(boneWorldMatrices: Float32Array): void {
    const N = this.store.count;
    const pinned = this.alignPinned;
    const boneIdx = this.store.boneIndex;
    const offsets = this.store.bodyOffsetMatrix;
    const positions = this.store.positions;
    const prevPos = this.prevPositions;
    for (let i = 0; i < N; i++) {
      if (!pinned[i]) continue;
      const b = boneIdx[i];
      if (b < 0) continue;
      const boneOff = b * 16;
      if (boneOff + 15 >= boneWorldMatrices.length) continue;
      mulArrays(boneWorldMatrices, boneOff, offsets, i * 16, _bodyMat, 0);
      const i3 = i * 3;
      positions[i3 + 0] = _bodyMat[12];
      positions[i3 + 1] = _bodyMat[13];
      positions[i3 + 2] = _bodyMat[14];
      prevPos[i3 + 0] = _bodyMat[12];
      prevPos[i3 + 1] = _bodyMat[13];
      prevPos[i3 + 2] = _bodyMat[14];
    }
  }

  private restoreNonFiniteBodies(): void {
    const N = this.store.count;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const lv = this.store.linearVelocities;
    const av = this.store.angularVelocities;
    const types = this.store.type;
    const prevPos = this.prevPositions;
    const prevOri = this.prevOrientations;
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic) continue;
      const i3 = i * 3;
      const i4 = i * 4;
      const s =
        positions[i3 + 0] + positions[i3 + 1] + positions[i3 + 2] +
        orientations[i4 + 0] + orientations[i4 + 1] + orientations[i4 + 2] + orientations[i4 + 3] +
        lv[i3 + 0] + lv[i3 + 1] + lv[i3 + 2] +
        av[i3 + 0] + av[i3 + 1] + av[i3 + 2];
      if (Number.isFinite(s)) continue;
      positions[i3 + 0] = prevPos[i3 + 0];
      positions[i3 + 1] = prevPos[i3 + 1];
      positions[i3 + 2] = prevPos[i3 + 2];
      orientations[i4 + 0] = prevOri[i4 + 0];
      orientations[i4 + 1] = prevOri[i4 + 1];
      orientations[i4 + 2] = prevOri[i4 + 2];
      orientations[i4 + 3] = prevOri[i4 + 3];
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0;
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0;
    }
  }

  private applyDynamicsToBones(boneWorldMatrices: Float32Array, alpha: number): void {
    const N = this.store.count;
    const inv = this.store.bodyOffsetInverse;
    const positions = this.store.positions;
    const orientations = this.store.orientations;
    const prevPos = this.prevPositions;
    const prevOri = this.prevOrientations;
    const types = this.store.type;
    const boneIdx = this.store.boneIndex;
    const oneMinus = 1 - alpha;
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic) continue;
      const b = boneIdx[i];
      if (b < 0) continue;
      const boneOff = b * 16;
      if (boneOff + 15 >= boneWorldMatrices.length) continue;
      const i3 = i * 3;
      const i4 = i * 4;
      const px = prevPos[i3 + 0] * oneMinus + positions[i3 + 0] * alpha;
      const py = prevPos[i3 + 1] * oneMinus + positions[i3 + 1] * alpha;
      const pz = prevPos[i3 + 2] * oneMinus + positions[i3 + 2] * alpha;
      const ax = prevOri[i4 + 0], ay = prevOri[i4 + 1], az = prevOri[i4 + 2], aw = prevOri[i4 + 3];
      let bx = orientations[i4 + 0], by = orientations[i4 + 1], bz = orientations[i4 + 2], bw = orientations[i4 + 3];
      if (ax * bx + ay * by + az * bz + aw * bw < 0) {
        bx = -bx; by = -by; bz = -bz; bw = -bw;
      }
      let qx = ax * oneMinus + bx * alpha;
      let qy = ay * oneMinus + by * alpha;
      let qz = az * oneMinus + bz * alpha;
      let qw = aw * oneMinus + bw * alpha;
      const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
      if (ql > 0) {
        const invL = 1 / ql;
        qx *= invL; qy *= invL; qz *= invL; qw *= invL;
      } else {
        qx = 0; qy = 0; qz = 0; qw = 1;
      }
      fromPositionRotation(px, py, pz, qx, qy, qz, qw, _bodyMat);
      mulArrays(_bodyMat, 0, inv, i * 16, _boneMat, 0);
      if (Number.isFinite(_boneMat[0]) && Math.abs(_boneMat[0]) < 1e6) {
        if (this.alignPinned[i]) {
          boneWorldMatrices[boneOff + 0] = _boneMat[0]; boneWorldMatrices[boneOff + 1] = _boneMat[1]; boneWorldMatrices[boneOff + 2] = _boneMat[2];
          boneWorldMatrices[boneOff + 4] = _boneMat[4]; boneWorldMatrices[boneOff + 5] = _boneMat[5]; boneWorldMatrices[boneOff + 6] = _boneMat[6];
          boneWorldMatrices[boneOff + 8] = _boneMat[8]; boneWorldMatrices[boneOff + 9] = _boneMat[9]; boneWorldMatrices[boneOff + 10] = _boneMat[10];
        } else {
          for (let j = 0; j < 16; j++) boneWorldMatrices[boneOff + j] = _boneMat[j];
        }
      }
    }
  }
}