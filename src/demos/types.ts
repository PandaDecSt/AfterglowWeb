import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";
import type { RenderGraph } from "../core/render-graph";
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

  /**
   * Create render passes for the Renderer pipeline.
   * Called once after init(). Each pass's execute() has full control
   * over the GPUCommandEncoder (can do compute/render/copy passes).
   */
  createPasses(): RenderPass[];

  /**
   * Setup render graph passes with dependency declarations.
   * If implemented, Renderer will use RenderGraph mode for this demo.
   * Passes declare reads/writes on named resources; the graph
   * auto-sorts by dependency and manages resource allocation.
   */
  setupGraph?(graph: RenderGraph): void;

  /** @deprecated Use createPasses() instead. Kept for backward compat. */
  render?(encoder: GPUCommandEncoder, view: GPUTextureView): void;

  destroy(): void;
  stats?(): DemoStats;
  registerGUI?(gui: GUI): void;
  getShaderStages?(): ShaderStageDesc[];
  onShaderReload?(stageLabel: string, code: string): boolean;
}
