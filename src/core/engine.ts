import { GPUContext } from "./device";
import { PipelineManager } from "./pipeline";
import { ResourceManager } from "./resource";
import { BindGroupCache } from "./bindgroup-cache";
import { AutoBinder } from "./auto-bind";
import { ShaderHotReload } from "../shader/hotreload";
import { ShaderModuleSystem } from "../shader/module-system";

export class EngineContext {
  readonly gpu: GPUContext;
  readonly device: GPUDevice;
  readonly pipelines: PipelineManager;
  readonly resources: ResourceManager;
  readonly bindGroups: BindGroupCache;
  readonly autoBind: AutoBinder;
  readonly shaderReload: ShaderHotReload;
  readonly modules: ShaderModuleSystem;

  constructor(gpu: GPUContext) {
    this.gpu = gpu;
    this.device = gpu.device;
    this.pipelines = new PipelineManager(gpu.device);
    this.resources = new ResourceManager(gpu.device);
    this.bindGroups = new BindGroupCache(gpu.device);
    this.autoBind = new AutoBinder(gpu.device, this.resources);
    this.shaderReload = new ShaderHotReload();
    this.modules = new ShaderModuleSystem();
  }

  compileShader(label: string, code: string): GPUShaderModule {
    return this.modules.resolveAndCompile(this.device, label, code);
  }

  get canvas() {
    return this.gpu.canvas;
  }

  get format() {
    return this.gpu.format;
  }

  get width() {
    return this.gpu.width;
  }

  get height() {
    return this.gpu.height;
  }

  destroy() {
    this.resources.destroy();
    this.bindGroups.invalidateAll();
  }
}
