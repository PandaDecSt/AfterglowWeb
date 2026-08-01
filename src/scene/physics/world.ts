import type { RigidBodyStore } from "./body";
import { RigidbodyType } from "./types";
import type { SixDofSpringConstraint } from "./constraint";
import { solveConstraints } from "./solver";
import { findContacts, type ContactPool } from "./contact";

export class World {
  gravityX: number;
  gravityY: number;
  gravityZ: number;
  solverIterations = 10;
  private dampCacheDt = -1;
  private linDampFactor: Float32Array | null = null;
  private angDampFactor: Float32Array | null = null;

  constructor(gx: number, gy: number, gz: number) {
    this.gravityX = gx;
    this.gravityY = gy;
    this.gravityZ = gz;
  }

  setGravity(gx: number, gy: number, gz: number): void {
    this.gravityX = gx;
    this.gravityY = gy;
    this.gravityZ = gz;
  }

  step(store: RigidBodyStore, constraints: SixDofSpringConstraint[], contacts: ContactPool, dt: number): void {
    if (dt <= 0) return;
    const N = store.count;
    const types = store.type;
    const lv = store.linearVelocities;
    const av = store.angularVelocities;
    const pos = store.positions;
    const ori = store.orientations;
    const ldamp = store.linearDamping;
    const adamp = store.angularDamping;
    const invMass = store.invMass;
    const gx = this.gravityX;
    const gy = this.gravityY;
    const gz = this.gravityZ;
    if (this.dampCacheDt !== dt || !this.linDampFactor || this.linDampFactor.length !== N) {
      this.dampCacheDt = dt;
      this.linDampFactor = new Float32Array(N);
      this.angDampFactor = new Float32Array(N);
      for (let i = 0; i < N; i++) {
        this.linDampFactor[i] = Math.pow(Math.max(0, 1 - ldamp[i]), dt);
        this.angDampFactor[i] = Math.pow(Math.max(0, 1 - adamp[i]), dt);
      }
    }
    const linDamp = this.linDampFactor;
    const angDamp = this.angDampFactor!;
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic || invMass[i] <= 0) continue;
      const i3 = i * 3;
      lv[i3 + 0] += gx * dt;
      lv[i3 + 1] += gy * dt;
      lv[i3 + 2] += gz * dt;
      const ld = linDamp[i];
      const ad = angDamp[i];
      lv[i3 + 0] *= ld; lv[i3 + 1] *= ld; lv[i3 + 2] *= ld;
      av[i3 + 0] *= ad; av[i3 + 1] *= ad; av[i3 + 2] *= ad;
    }
    contacts.reset();
    findContacts(store, contacts);
    if (constraints.length > 0 || contacts.count > 0) {
      solveConstraints(store, constraints, contacts, dt, this.solverIterations);
    }
    const POS_CORRECTION_FACTOR = 0.4;
    const POS_SLOP = 0.005;
    for (let ci = 0; ci < contacts.count; ci++) {
      const c = contacts.get(ci);
      if (c.depth <= POS_SLOP) continue;
      const imA = invMass[c.bodyA];
      const imB = invMass[c.bodyB];
      const total = imA + imB;
      if (total <= 0) continue;
      const correction = (c.depth - POS_SLOP) * POS_CORRECTION_FACTOR;
      const dx = correction * c.nx;
      const dy = correction * c.ny;
      const dz = correction * c.nz;
      const ai = c.bodyA * 3;
      const bi = c.bodyB * 3;
      if (imA > 0) {
        const fA = imA / total;
        pos[ai + 0] -= dx * fA;
        pos[ai + 1] -= dy * fA;
        pos[ai + 2] -= dz * fA;
      }
      if (imB > 0) {
        const fB = imB / total;
        pos[bi + 0] += dx * fB;
        pos[bi + 1] += dy * fB;
        pos[bi + 2] += dz * fB;
      }
    }
    const MAX_ANGVEL_DT = Math.PI * 0.5;
    const MAX_LINVEL_DT = 5;
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic || invMass[i] <= 0) continue;
      const i3 = i * 3;
      const i4 = i * 4;
      const vx = lv[i3 + 0], vy = lv[i3 + 1], vz = lv[i3 + 2];
      const vmag = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (vmag * dt > MAX_LINVEL_DT) {
        const scale = MAX_LINVEL_DT / (vmag * dt);
        lv[i3 + 0] = vx * scale;
        lv[i3 + 1] = vy * scale;
        lv[i3 + 2] = vz * scale;
      }
      pos[i3 + 0] += lv[i3 + 0] * dt;
      pos[i3 + 1] += lv[i3 + 1] * dt;
      pos[i3 + 2] += lv[i3 + 2] * dt;
      let wx = av[i3 + 0];
      let wy = av[i3 + 1];
      let wz = av[i3 + 2];
      const wmag = Math.sqrt(wx * wx + wy * wy + wz * wz);
      if (wmag * dt > MAX_ANGVEL_DT) {
        const scale = MAX_ANGVEL_DT / (wmag * dt);
        wx *= scale; wy *= scale; wz *= scale;
        av[i3 + 0] = wx; av[i3 + 1] = wy; av[i3 + 2] = wz;
      }
      if (wx !== 0 || wy !== 0 || wz !== 0) {
        const qx = ori[i4 + 0];
        const qy = ori[i4 + 1];
        const qz = ori[i4 + 2];
        const qw = ori[i4 + 3];
        const dx = qw * wx + wy * qz - wz * qy;
        const dy = qw * wy + wz * qx - wx * qz;
        const dz = qw * wz + wx * qy - wy * qx;
        const dw = -(wx * qx + wy * qy + wz * qz);
        const half = 0.5 * dt;
        const nx = qx + dx * half;
        const ny = qy + dy * half;
        const nz = qz + dz * half;
        const nw = qw + dw * half;
        const len2 = nx * nx + ny * ny + nz * nz + nw * nw;
        if (len2 > 0) {
          const inv = 1 / Math.sqrt(len2);
          ori[i4 + 0] = nx * inv;
          ori[i4 + 1] = ny * inv;
          ori[i4 + 2] = nz * inv;
          ori[i4 + 3] = nw * inv;
        }
      }
    }
  }
}