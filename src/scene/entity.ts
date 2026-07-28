import { mat4, vec3, type Vec3, type Mat4 } from "wgpu-matrix";

export interface Component {
  update?(dt: number, time: number): void;
}

export class Entity {
  name: string;
  position: Vec3 = vec3.create(0, 0, 0);
  rotation: Vec3 = vec3.create(0, 0, 0);
  scale: Vec3 = vec3.create(1, 1, 1);
  components: Component[] = [];
  visible = true;

  constructor(name: string) {
    this.name = name;
  }

  addComponent(c: Component): this {
    this.components.push(c);
    return this;
  }

  update(dt: number, time: number) {
    for (const c of this.components) {
      c.update?.(dt, time);
    }
  }

  getModelMatrix(): Mat4 {
    const t = mat4.translation(this.position);
    const rx = mat4.rotationX(this.rotation[0]);
    const ry = mat4.rotationY(this.rotation[1]);
    const rz = mat4.rotationZ(this.rotation[2]);
    const s = mat4.scaling(this.scale);
    return mat4.mul(mat4.mul(mat4.mul(t, ry), mat4.mul(rx, rz)), s);
  }
}

export class Scene {
  entities: Entity[] = [];

  add(entity: Entity): Entity {
    this.entities.push(entity);
    return entity;
  }

  remove(name: string) {
    this.entities = this.entities.filter((e) => e.name !== name);
  }

  get(name: string): Entity | undefined {
    return this.entities.find((e) => e.name === name);
  }

  update(dt: number, time: number) {
    for (const e of this.entities) {
      e.update(dt, time);
    }
  }
}
