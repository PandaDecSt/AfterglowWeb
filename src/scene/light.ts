export enum LightType {
  Directional = 0,
  Point = 1,
  Spot = 2,
  Area = 3,
}

export interface DirectionalLight {
  type: LightType.Directional;
  direction: [number, number, number];
  color: [number, number, number];
  intensity: number;
  castShadow: boolean;
  shadowBias: number;
  shadowNormalBias: number;
}

export interface PointLight {
  type: LightType.Point;
  position: [number, number, number];
  color: [number, number, number];
  intensity: number;
  range: number;
  falloffExponent: number;
}

export interface SpotLight {
  type: LightType.Spot;
  position: [number, number, number];
  direction: [number, number, number];
  color: [number, number, number];
  intensity: number;
  range: number;
  innerConeAngle: number;
  outerConeAngle: number;
}

export interface AreaLight {
  type: LightType.Area;
  position: [number, number, number];
  direction: [number, number, number];
  tangent: [number, number, number];
  color: [number, number, number];
  intensity: number;
  width: number;
  height: number;
}

export type Light = DirectionalLight | PointLight | SpotLight | AreaLight;

export function createDirectionalLight(
  direction: [number, number, number] = [0, -1, 0],
  color: [number, number, number] = [1, 1, 1],
  intensity = 3.0,
): DirectionalLight {
  return {
    type: LightType.Directional,
    direction,
    color,
    intensity,
    castShadow: true,
    shadowBias: 0.001,
    shadowNormalBias: 0.08,
  };
}

export function createPointLight(
  position: [number, number, number] = [0, 2, 0],
  color: [number, number, number] = [1, 1, 1],
  intensity = 10.0,
  range = 20.0,
): PointLight {
  return {
    type: LightType.Point,
    position,
    color,
    intensity,
    range,
    falloffExponent: 2.0,
  };
}

export function createSpotLight(
  position: [number, number, number] = [0, 5, 0],
  direction: [number, number, number] = [0, -1, 0],
  color: [number, number, number] = [1, 1, 1],
  intensity = 15.0,
  range = 25.0,
  innerConeAngle = 0.4,
  outerConeAngle = 0.6,
): SpotLight {
  return {
    type: LightType.Spot,
    position,
    direction,
    color,
    intensity,
    range,
    innerConeAngle,
    outerConeAngle,
  };
}

export const MAX_LIGHTS = 256;
export const CLUSTER_SIZE_X = 16;
export const CLUSTER_SIZE_Y = 16;
export const CLUSTER_SIZE_Z = 24;

export class LightScene {
  private _lights: Light[] = [];
  private _ambientColor: [number, number, number] = [0.03, 0.03, 0.03];
  private _ambientIntensity = 1.0;

  get lights(): readonly Light[] { return this._lights; }
  get ambientColor(): [number, number, number] { return this._ambientColor; }
  set ambientColor(v: [number, number, number]) { this._ambientColor = v; }
  get ambientIntensity(): number { return this._ambientIntensity; }
  set ambientIntensity(v: number) { this._ambientIntensity = v; }

  addLight(light: Light): number {
    if (this._lights.length >= MAX_LIGHTS) {
      console.warn(`[LightScene] Exceeded MAX_LIGHTS=${MAX_LIGHTS}, dropping light`);
      return -1;
    }
    this._lights.push(light);
    return this._lights.length - 1;
  }

  removeLight(index: number): void {
    if (index >= 0 && index < this._lights.length) {
      this._lights.splice(index, 1);
    }
  }

  getLight(index: number): Light | undefined {
    return this._lights[index];
  }

  get directionalLights(): DirectionalLight[] {
    return this._lights.filter(l => l.type === LightType.Directional) as DirectionalLight[];
  }

  get pointLights(): PointLight[] {
    return this._lights.filter(l => l.type === LightType.Point) as PointLight[];
  }

  get spotLights(): SpotLight[] {
    return this._lights.filter(l => l.type === LightType.Spot) as SpotLight[];
  }

  get count(): number { return this._lights.length; }

  buildLightBuffer(): Float32Array {
    const FLOATS_PER_LIGHT = 12;
    const data = new Float32Array((1 + this._lights.length * FLOATS_PER_LIGHT));
    data[0] = this._lights.length;
    let offset = 1;

    for (const light of this._lights) {
      switch (light.type) {
        case LightType.Directional: {
          const dl = light as DirectionalLight;
          data[offset + 0] = LightType.Directional;
          data[offset + 1] = dl.intensity;
          data[offset + 2] = dl.castShadow ? 1.0 : 0.0;
          data[offset + 3] = 0;
          data[offset + 4] = dl.direction[0];
          data[offset + 5] = dl.direction[1];
          data[offset + 6] = dl.direction[2];
          data[offset + 7] = 0;
          data[offset + 8] = dl.color[0];
          data[offset + 9] = dl.color[1];
          data[offset + 10] = dl.color[2];
          data[offset + 11] = 0;
          break;
        }
        case LightType.Point: {
          const pl = light as PointLight;
          data[offset + 0] = LightType.Point;
          data[offset + 1] = pl.intensity;
          data[offset + 2] = pl.range;
          data[offset + 3] = pl.falloffExponent;
          data[offset + 4] = pl.position[0];
          data[offset + 5] = pl.position[1];
          data[offset + 6] = pl.position[2];
          data[offset + 7] = 0;
          data[offset + 8] = pl.color[0];
          data[offset + 9] = pl.color[1];
          data[offset + 10] = pl.color[2];
          data[offset + 11] = 0;
          break;
        }
        case LightType.Spot: {
          const sl = light as SpotLight;
          data[offset + 0] = LightType.Spot;
          data[offset + 1] = sl.intensity;
          data[offset + 2] = sl.range;
          data[offset + 3] = Math.cos(sl.outerConeAngle);
          data[offset + 4] = sl.position[0];
          data[offset + 5] = sl.position[1];
          data[offset + 6] = sl.position[2];
          data[offset + 7] = Math.cos(sl.innerConeAngle);
          data[offset + 8] = sl.direction[0];
          data[offset + 9] = sl.direction[1];
          data[offset + 10] = sl.direction[2];
          data[offset + 11] = 0;
          break;
        }
        case LightType.Area: {
          const al = light as AreaLight;
          data[offset + 0] = LightType.Area;
          data[offset + 1] = al.intensity;
          data[offset + 2] = al.width;
          data[offset + 3] = al.height;
          data[offset + 4] = al.position[0];
          data[offset + 5] = al.position[1];
          data[offset + 6] = al.position[2];
          data[offset + 7] = 0;
          data[offset + 8] = al.direction[0];
          data[offset + 9] = al.direction[1];
          data[offset + 10] = al.direction[2];
          data[offset + 11] = 0;
          break;
        }
      }
      offset += FLOATS_PER_LIGHT;
    }

    return data;
  }

  clear(): void {
    this._lights.length = 0;
  }
}