import { mat4, quat, vec3, type Vec3, type Mat4 } from "wgpu-matrix";
import type { CameraPose } from "./camera-animation";

export class Camera {
  position: Vec3 = vec3.create(0, 2, 6);
  target: Vec3 = vec3.create(0, 0, 0);
  up: Vec3 = vec3.create(0, 1, 0);
  fov = 60;
  near = 0.1;
  far = 100;

  vmdDriven = false;
  private _vmdTarget: Vec3 = vec3.create(0, 0, 0);
  private _vmdRotation: Vec3 = vec3.create(0, 0, 0);
  private _vmdDistance = 0;
  private _vmdFov = 30;

  private yaw = -Math.PI / 2;
  private pitch = -0.3;
  private distance = 7;
  private isDragging = false;
  private lastX = 0;
  private lastY = 0;

  constructor(canvas: HTMLCanvasElement) {
    canvas.addEventListener("pointerdown", (e) => {
      this.isDragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointerup", (e) => {
      this.isDragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.isDragging) return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      this.yaw += dx * 0.005;
      this.pitch = Math.max(
        -Math.PI / 2 + 0.01,
        Math.min(Math.PI / 2 - 0.01, this.pitch + dy * 0.005)
      );
    });
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.distance = Math.max(
          1,
          Math.min(50, this.distance + e.deltaY * 0.01)
        );
      },
      { passive: false }
    );
  }

  update() {
    this.position = vec3.create(
      this.target[0] +
        this.distance * Math.cos(this.pitch) * Math.cos(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] +
        this.distance * Math.cos(this.pitch) * Math.sin(this.yaw)
    );
  }

  setVmdPose(pose: CameraPose): void {
    this._vmdTarget = vec3.create(pose.target[0], pose.target[1], pose.target[2]);
    this._vmdRotation = vec3.create(pose.rotation[0], pose.rotation[1], pose.rotation[2]);
    this._vmdDistance = pose.distance;
    this._vmdFov = pose.fov > 0 ? pose.fov : 30;
  }

  faceTarget: Vec3 | null = null;

  getViewMatrix(): Mat4 {
    if (this.vmdDriven) {
      const r = this._vmdRotation;
      const q = quat.fromEuler(-r[0], -r[1], -r[2], 'zyx', quat.create());
      const rm = mat4.fromQuat(q, mat4.create());
      const t = this._vmdTarget;
      const d = this._vmdDistance;
      const ex = t[0] + rm[8] * d;
      const ey = t[1] + rm[9] * d;
      const ez = t[2] + rm[10] * d;
      this.position = vec3.create(ex, ey, -ez);
      this.target = vec3.create(t[0], t[1], -t[2]);
      this.fov = this._vmdFov;
      const up = vec3.create(rm[4], rm[5], -rm[6]);
      return mat4.lookAt(this.position, this.target, up);
    }
    if (this.faceTarget) {
      this.target = this.faceTarget;
    }
    this.update();
    return mat4.lookAt(this.position, this.target, this.up);
  }

  private _zRemap: Mat4 = (() => { const m = mat4.identity(mat4.create()); m[10] = 0.5; m[14] = 0.5; return m; })();

  getProjectionMatrix(aspect: number): Mat4 {
    return mat4.mul(this._zRemap, mat4.perspective((this.fov * Math.PI) / 180, aspect, this.near, this.far));
  }

  getViewProjectionMatrix(aspect: number): Mat4 {
    return mat4.mul(this.getProjectionMatrix(aspect), this.getViewMatrix());
  }

  orbit(target: Vec3, distance: number, near?: number, far?: number, yaw?: number, pitch?: number): void {
    this.target = target;
    this.distance = distance;
    if (near !== undefined) this.near = near;
    if (far !== undefined) this.far = far;
    if (yaw !== undefined) this.yaw = yaw;
    if (pitch !== undefined) this.pitch = pitch;
  }

  debugInfo(): string {
    const p = this.position, t = this.target;
    return `vmdDriven=${this.vmdDriven} fov=${this.fov} near=${this.near} far=${this.far} eye=[${p[0].toFixed(2)},${p[1].toFixed(2)},${p[2].toFixed(2)}] target=[${t[0].toFixed(2)},${t[1].toFixed(2)},${t[2].toFixed(2)}]`;
  }
}
