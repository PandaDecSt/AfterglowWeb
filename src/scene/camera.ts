import { mat4, vec3, type Vec3, type Mat4 } from "wgpu-matrix";

export class Camera {
  position: Vec3 = vec3.create(0, 2, 6);
  target: Vec3 = vec3.create(0, 0, 0);
  up: Vec3 = vec3.create(0, 1, 0);
  fov = 60;
  near = 0.1;
  far = 100;

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

  getViewMatrix(): Mat4 {
    this.update();
    return mat4.lookAt(this.position, this.target, this.up);
  }

  getProjectionMatrix(aspect: number): Mat4 {
    return mat4.perspective((this.fov * Math.PI) / 180, aspect, this.near, this.far);
  }

  getViewProjectionMatrix(aspect: number): Mat4 {
    return mat4.mul(this.getProjectionMatrix(aspect), this.getViewMatrix());
  }
}
