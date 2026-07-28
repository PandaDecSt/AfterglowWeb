import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import type { EngineContext } from "../core/engine";
import type GUI from "lil-gui";

export interface DemoStats {
  drawCalls?: number;
  triangles?: number;
  instances?: number;
  computeDispatches?: number;
  custom?: Record<string, string | number>;
}

export type ShaderStageType = "vertex" | "fragment" | "compute" | "postprocess";

export interface ShaderStageDesc {
  label: string;
  type: ShaderStageType;
  code: string;
}

export interface Demo {
  label: string;
  init(ctx: GPUContext, camera: Camera, engine?: EngineContext): Promise<void> | void;
  update(time: number, deltaTime: number): void;
  render(encoder: GPUCommandEncoder, view: GPUTextureView): void;
  destroy(): void;
  stats?(): DemoStats;
  registerGUI?(gui: GUI): void;
  getShaderStages?(): ShaderStageDesc[];
  onShaderReload?(stageLabel: string, code: string): boolean;
}
