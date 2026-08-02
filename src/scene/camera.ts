import { mat4, quat, vec3, type Vec3, type Mat4 } from "wgpu-matrix";
import type { CameraPose } from "./camera-animation";
import { MmdCoord } from "./mmd-coord";

export type CameraMode = "orbit" | "vmd";

export class Camera {
  position: Vec3 = vec3.create(0, 2, 6);
  target: Vec3 = vec3.create(0, 0, 0);
  up: Vec3 = vec3.create(0, 1, 0);
  fov = 60;
  near = 0.1;
  far = 100;

  mode: CameraMode = "orbit";

  private _vmdTarget: Vec3 = vec3.create(0, 0, 0);
  private _vmdRotation: Vec3 = vec3.create(0, 0, 0);
  private _vmdDistance = 0;
  private _vmdFov = 30;
  private _savedFov = 60;

  private yaw = -Math.PI / 2;
  private pitch = -0.3;
  distance = 7;
  private isDragging = false;
  private lastX = 0;
  private lastY = 0;

  private _scratchQ = quat.create();
  private _scratchRM = mat4.create();
  private _scratchV3 = vec3.create();
  private _scratchView = mat4.create();
  private _scratchProj = mat4.create();
  private _scratchZProj = mat4.create();
  private _scratchVP = mat4.create();

  constructor(canvas: HTMLCanvasElement) {
    canvas.addEventListener("pointerdown", (e) => {
      if (this.mode === "vmd") return;
      this.isDragging = true;
      this.lastX = e.clientX; this.lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointerup", (e) => {
      this.isDragging = false;
      canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.isDragging || this.mode === "vmd") return;
      const dx = e.clientX - this.lastX;
      const dy = e.clientY - this.lastY;
      this.lastX = e.clientX; this.lastY = e.clientY;
      this.yaw += dx * 0.005;
      this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch + dy * 0.005));
    });
    canvas.addEventListener("wheel", (e) => {
      if (this.mode === "vmd") return;
      e.preventDefault();
      this.distance = Math.max(1, Math.min(50, this.distance + e.deltaY * 0.01));
    }, { passive: false });
  }

  setVmdPose(pose: CameraPose): void {
    this._vmdTarget[0] = pose.target[0]; this._vmdTarget[1] = pose.target[1]; this._vmdTarget[2] = pose.target[2];
    this._vmdRotation[0] = pose.rotation[0]; this._vmdRotation[1] = pose.rotation[1]; this._vmdRotation[2] = pose.rotation[2];
    this._vmdDistance = pose.distance;
    this._vmdFov = pose.fov > 0 ? pose.fov : 30;
  }

  setMode(mode: CameraMode): void {
    if (mode === this.mode) return;
    if (mode === "vmd") {
      this._savedFov = this.fov;
    } else {
      this.fov = this._savedFov;
    }
    this.mode = mode;
  }

  private orbitUpdate(): void {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    this.position[0] = this.target[0] + this.distance * cp * cy;
    this.position[1] = this.target[1] + this.distance * sp;
    this.position[2] = this.target[2] + this.distance * cp * sy;
  }

  getViewMatrix(): Mat4 {
    if (this.mode === "vmd") {
      const r = this._vmdRotation;
      quat.fromEuler(-r[0], -r[1], -r[2], 'zyx', this._scratchQ);
      mat4.fromQuat(this._scratchQ, this._scratchRM);
      const t = this._vmdTarget;
      const d = this._vmdDistance;
      const rm = this._scratchRM;
      const ex = t[0] + rm[8] * d;
      const ey = t[1] + rm[9] * d;
      const ez = t[2] + rm[10] * d;
      this.position[0] = ex; this.position[1] = ey; this.position[2] = -ez;
      this.target[0] = t[0]; this.target[1] = t[1]; this.target[2] = -t[2];
      this.fov = this._vmdFov;
      const sv = this._scratchV3;
      sv[0] = rm[4]; sv[1] = rm[5]; sv[2] = -rm[6];
      return mat4.lookAt(this.position, this.target, sv, this._scratchView);
    }

    this.orbitUpdate();
    return mat4.lookAt(this.position, this.target, this.up, this._scratchView);
  }

  private _zRemap: Mat4 = (() => { const m = mat4.identity(mat4.create()); m[10] = 0.5; m[14] = 0.5; return m; })();

  getProjectionMatrix(aspect: number): Mat4 {
    mat4.perspective((this.fov * Math.PI) / 180, aspect, this.near, this.far, this._scratchProj);
    return mat4.mul(this._zRemap, this._scratchProj, this._scratchZProj);
  }

  getViewProjectionMatrix(aspect: number): Mat4 {
    const proj = this.getProjectionMatrix(aspect);
    const view = this.getViewMatrix();
    return mat4.mul(proj, view, this._scratchVP);
  }

  orbit(target: Vec3, distance: number, near?: number, far?: number, yaw?: number, pitch?: number): void {
    this.target = target;
    this.distance = distance;
    if (near !== undefined) this.near = near;
    if (far !== undefined) this.far = far;
    if (yaw !== undefined) this.yaw = yaw;
    if (pitch !== undefined) this.pitch = pitch;
  }
}
