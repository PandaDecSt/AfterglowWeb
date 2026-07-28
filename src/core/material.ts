// AfterglowRender Material System
// Ported from AfterglowMaterial.h/cpp to TypeScript + WebGPU

// ============================================================
// Enums & Constants
// ============================================================

export enum ShaderStage {
  Vertex = 'vertex',
  Fragment = 'fragment',
  Compute = 'compute',
}

export enum RenderDomain {
  DepthPrepass = 'depth-prepass',
  Shadow = 'shadow',
  DeferredGeometry = 'deferred-geometry',
  Decal = 'decal',
  DeferredLighting = 'deferred-lighting',
  Forward = 'forward',
  Transparency = 'transparency',
  PostProcess = 'post-process',
  UserInterface = 'ui',
}

export enum CullMode {
  None = 'none',
  Front = 'front',
  Back = 'back',
  FrontBack = 'front-and-back',
}

export enum Topology {
  TriangleList = 'triangle-list',
  TriangleStrip = 'triangle-strip',
  PointList = 'point-list',
  LineList = 'line-list',
  LineStrip = 'line-strip',
}

export enum CompareOp {
  Never = 'never',
  Less = 'less',
  Equal = 'equal',
  LessEqual = 'less-equal',
  Greater = 'greater',
  NotEqual = 'not-equal',
  GreaterEqual = 'greater-equal',
  Always = 'always',
}

export enum StencilOp {
  Keep = 'keep',
  Zero = 'zero',
  Replace = 'replace',
  IncrementClamp = 'increment-clamp',
  DecrementClamp = 'decrement-clamp',
  Invert = 'invert',
  IncrementWrap = 'increment-wrap',
  DecrementWrap = 'decrement-wrap',
}

// ============================================================
// Stencil Configuration
// ============================================================

export interface StencilInfo {
  stencilValue: number;
  compareMask: number;
  writeMask: number;
  compareOp: CompareOp;
  failOp: StencilOp;
  passOp: StencilOp;
  depthFailOp: StencilOp;
}

export interface FaceStencilInfos {
  front: StencilInfo;
  back: StencilInfo;
}

function defaultStencilInfo(): StencilInfo {
  return {
    stencilValue: 0,
    compareMask: 0xff,
    writeMask: 0xff,
    compareOp: CompareOp.Always,
    failOp: StencilOp.Keep,
    passOp: StencilOp.Keep,
    depthFailOp: StencilOp.Keep,
  };
}

function defaultFaceStencilInfos(): FaceStencilInfos {
  return {
    front: defaultStencilInfo(),
    back: defaultStencilInfo(),
  };
}

// ============================================================
// Parameter Types
// ============================================================

export type MaterialScalar = number;
export type MaterialVector = [number, number, number, number];
export type MaterialTextureInfo = { textureUrl: string; sampler?: GPUSamplerDescriptor };

interface Parameter<T> {
  name: string;
  value: T;
  modified: boolean;
}

// ============================================================
// Material Class
// ============================================================

export class Material {
  // Per-stage parameter storage
  private _scalars = new Map<ShaderStage, Parameter<MaterialScalar>[]>();
  private _vectors = new Map<ShaderStage, Parameter<MaterialVector>[]>();
  private _textures = new Map<ShaderStage, Parameter<MaterialTextureInfo>[]>();

  // Render state
  private _domain: RenderDomain = RenderDomain.Forward;
  private _topology: Topology = Topology.TriangleList;
  private _cullMode: CullMode = CullMode.Back;
  private _wireframe = false;
  private _depthWrite = true;
  private _depthCompare: GPUCompareFunction = 'less';
  private _blendState: GPUBlendState | undefined = undefined;

  // Stencil
  private _faceStencilInfos: FaceStencilInfos = defaultFaceStencilInfos();

  // Shader paths
  private _vertexShaderPath = '';
  private _fragmentShaderPath = '';
  private _computeShaderPath = '';

  // Material identity
  private _name = '';

  // Compute task (lazily created)
  private _computeTask: any = null;

  // GPU resources (set during upload)
  private _uniformBuffer: GPUBuffer | null = null;
  private _uniformData: Float32Array | null = null;
  private _textureCache = new Map<string, GPUTexture>();

  constructor(name: string) {
    this._name = name;
  }

  get name(): string {
    return this._name;
  }

  // ============================================================
  // Parameter Accessors
  // ============================================================

  private findParameter<T>(container: Map<ShaderStage, Parameter<T>[]>, stage: ShaderStage, name: string): Parameter<T> | undefined {
    const params = container.get(stage);
    if (!params) return undefined;
    return params.find(p => p.name === name);
  }

  private upsertParameter<T>(container: Map<ShaderStage, Parameter<T>[]>, stage: ShaderStage, name: string, value: T) {
    let params = container.get(stage);
    if (!params) {
      params = [];
      container.set(stage, params);
    }
    const existing = params.find(p => p.name === name);
    if (existing) {
      existing.value = value;
      existing.modified = true;
    } else {
      params.push({ name, value, modified: true });
    }
  }

  // Scalar accessors
  scalar(stage: ShaderStage, name: string): MaterialScalar | undefined {
    return this.findParameter(this._scalars, stage, name)?.value;
  }

  setScalar(stage: ShaderStage, name: string, value: MaterialScalar) {
    this.upsertParameter(this._scalars, stage, name, value);
  }

  // Vector accessors
  vector(stage: ShaderStage, name: string): MaterialVector | undefined {
    return this.findParameter(this._vectors, stage, name)?.value;
  }

  setVector(stage: ShaderStage, name: string, value: MaterialVector) {
    this.upsertParameter(this._vectors, stage, name, value);
  }

  // Texture accessors
  texture(stage: ShaderStage, name: string): MaterialTextureInfo | undefined {
    return this.findParameter(this._textures, stage, name)?.value;
  }

  setTexture(stage: ShaderStage, name: string, value: MaterialTextureInfo) {
    this.upsertParameter(this._textures, stage, name, value);
  }

  // ============================================================
  // Modified Flag Queries
  // ============================================================

  isModified(stage: ShaderStage): boolean {
    const scalars = this._scalars.get(stage);
    const vectors = this._vectors.get(stage);
    const textures = this._textures.get(stage);
    return (
      (scalars?.some(p => p.modified) ?? false) ||
      (vectors?.some(p => p.modified) ?? false) ||
      (textures?.some(p => p.modified) ?? false)
    );
  }

  hasAnyModified(): boolean {
    for (const stage of Object.values(ShaderStage)) {
      if (this.isModified(stage)) return true;
    }
    return false;
  }

  clearModified(stage: ShaderStage) {
    const scalars = this._scalars.get(stage);
    const vectors = this._vectors.get(stage);
    const textures = this._textures.get(stage);
    scalars?.forEach(p => p.modified = false);
    vectors?.forEach(p => p.modified = false);
    textures?.forEach(p => p.modified = false);
  }

  clearAllModified() {
    for (const stage of Object.values(ShaderStage)) {
      this.clearModified(stage);
    }
  }

  // ============================================================
  // Scalar Padding for Uniform Buffer Alignment (to vec4 = 4 floats)
  // ============================================================

  scalarPaddingSize(stage: ShaderStage): number {
    const count = this._scalars.get(stage)?.length ?? 0;
    const alignment = 4;
    return (Math.ceil(count / alignment) * alignment) - count;
  }

  // ============================================================
  // Render State
  // ============================================================

  get domain(): RenderDomain { return this._domain; }
  set domain(value: RenderDomain) { this._domain = value; }

  get topology(): Topology { return this._topology; }
  set topology(value: Topology) { this._topology = value; }

  get cullMode(): CullMode { return this._cullMode; }
  set cullMode(value: CullMode) { this._cullMode = value; }

  get wireframe(): boolean { return this._wireframe; }
  set wireframe(value: boolean) { this._wireframe = value; }

  get depthWrite(): boolean { return this._depthWrite; }
  set depthWrite(value: boolean) { this._depthWrite = value; }

  get depthCompare(): GPUCompareFunction { return this._depthCompare; }
  set depthCompare(value: GPUCompareFunction) { this._depthCompare = value; }

  get blendState(): GPUBlendState | undefined { return this._blendState; }
  set blendState(value: GPUBlendState | undefined) { this._blendState = value; }

  // ============================================================
  // Stencil Configuration
  // ============================================================

  get faceStencilInfos(): FaceStencilInfos { return this._faceStencilInfos; }
  setFaceStencilInfo(face: 'front' | 'back', info: Partial<StencilInfo>) {
    Object.assign(this._faceStencilInfos[face], info);
  }
  setStencilCompareOp(face: 'front' | 'back', op: CompareOp) {
    this._faceStencilInfos[face].compareOp = op;
  }
  setStencilOps(face: 'front' | 'back', ops: { failOp?: StencilOp; passOp?: StencilOp; depthFailOp?: StencilOp }) {
    Object.assign(this._faceStencilInfos[face], ops);
  }

  // ============================================================
  // Shader Paths
  // ============================================================

  get vertexShaderPath(): string { return this._vertexShaderPath; }
  set vertexShaderPath(value: string) { this._vertexShaderPath = value; }

  get fragmentShaderPath(): string { return this._fragmentShaderPath; }
  set fragmentShaderPath(value: string) { this._fragmentShaderPath = value; }

  get computeShaderPath(): string { return this._computeShaderPath; }
  set computeShaderPath(value: string) { this._computeShaderPath = value; }

  // ============================================================
  // Compute Task Integration
  // ============================================================

  get hasComputeTask(): boolean {
    return this._computeTask !== null;
  }

  initComputeTask(): any {
    if (!this._computeTask) {
      // Lazy creation - import ComputeTask class dynamically to avoid circular deps
      // In practice, the caller should create and attach the compute task
      this._computeTask = {};
    }
    return this._computeTask;
  }

  get computeTask(): any {
    if (!this._computeTask) {
      throw new Error(`Material '${this._name}' has no compute task. Call initComputeTask() first.`);
    }
    return this._computeTask;
  }

  // ============================================================
  // Uniform Buffer Building
  // ============================================================

  /**
   * Builds uniform buffer data for a given stage.
   * Layout: scalars first (padded to vec4 alignment), then vectors.
   */
  buildUniformData(stage: ShaderStage): Float32Array {
    const scalars = this._scalars.get(stage) ?? [];
    const vectors = this._vectors.get(stage) ?? [];

    const padding = this.scalarPaddingSize(stage);
    const totalScalars = scalars.length + padding;
    const totalVectors = vectors.length;
    const totalFloats = totalScalars + totalVectors * 4;

    const data = new Float32Array(totalFloats);

    // Pack scalars
    for (let i = 0; i < scalars.length; i++) {
      data[i] = scalars[i].value;
    }

    // Pack vectors (each is vec4 = 4 floats)
    const vectorOffset = totalScalars;
    for (let i = 0; i < vectors.length; i++) {
      const v = vectors[i].value;
      data[vectorOffset + i * 4 + 0] = v[0];
      data[vectorOffset + i * 4 + 1] = v[1];
      data[vectorOffset + i * 4 + 2] = v[2];
      data[vectorOffset + i * 4 + 3] = v[3];
    }

    return data;
  }

  /**
   * Upload uniform data to GPU buffer. Only writes if any parameter is modified.
   */
  uploadUniforms(device: GPUDevice, stage: ShaderStage): GPUBuffer | null {
    const scalars = this._scalars.get(stage) ?? [];
    const vectors = this._vectors.get(stage) ?? [];

    // Check if any parameter is modified
    const anyModified = scalars.some(p => p.modified) || vectors.some(p => p.modified);
    if (!anyModified) return this._uniformBuffer;

    // Build data
    const data = this.buildUniformData(stage);

    // Create or update buffer
    if (this._uniformBuffer) {
      // Re-create (WebGPU doesn't have in-place resize for uniform buffers)
      this._uniformBuffer.destroy();
    }

    this._uniformBuffer = device.createBuffer({
      label: `${this._name}-uniforms-${stage}`,
      size: Math.max(data.byteLength, 4),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    const mapped = new Float32Array(this._uniformBuffer.getMappedRange());
    mapped.set(data);
    this._uniformBuffer.unmap();

    this._uniformData = data;
    this.clearModified(stage);

    return this._uniformBuffer;
  }

  get uniformBuffer(): GPUBuffer | null {
    return this._uniformBuffer;
  }

  // ============================================================
  // Bind Group Helpers
  // ============================================================

  /**
   * Create bind group entries for this material's stage parameters.
   * Returns an array of GPUBindGroupLayoutEntry for the given stage.
   */
  createBindGroupLayoutEntries(stage: ShaderStage): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [];
    let binding = 0;

    // Uniform buffer (if any scalars or vectors)
    const hasUniforms = (this._scalars.get(stage)?.length ?? 0) + (this._vectors.get(stage)?.length ?? 0) > 0;
    if (hasUniforms) {
      entries.push({
        binding: binding++,
        visibility: stage === ShaderStage.Vertex ? GPUShaderStage.VERTEX : GPUShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      });
    }

    // Textures
    const textures = this._textures.get(stage) ?? [];
    for (const tex of textures) {
      entries.push({
        binding: binding++,
        visibility: stage === ShaderStage.Vertex ? GPUShaderStage.VERTEX : GPUShaderStage.FRAGMENT,
        texture: { sampleType: 'float', viewDimension: '2d' },
      });
      entries.push({
        binding: binding++,
        visibility: stage === ShaderStage.Vertex ? GPUShaderStage.VERTEX : GPUShaderStage.FRAGMENT,
        sampler: { type: 'filtering' },
      });
    }

    return entries;
  }

  // ============================================================
  // Cleanup
  // ============================================================

  destroy() {
    if (this._uniformBuffer) {
      this._uniformBuffer.destroy();
      this._uniformBuffer = null;
    }
    for (const tex of this._textureCache.values()) {
      tex.destroy();
    }
    this._textureCache.clear();
  }

  // ============================================================
  // Clone
  // ============================================================

  clone(): Material {
    const mat = new Material(`${this._name}_clone`);

    // Deep copy parameters
    for (const [stage, params] of this._scalars) {
      mat._scalars.set(stage, params.map(p => ({ ...p })));
    }
    for (const [stage, params] of this._vectors) {
      mat._vectors.set(stage, params.map(p => ({ ...p })));
    }
    for (const [stage, params] of this._textures) {
      mat._textures.set(stage, params.map(p => ({ ...p })));
    }

    // Copy render state
    mat._domain = this._domain;
    mat._topology = this._topology;
    mat._cullMode = this._cullMode;
    mat._wireframe = this._wireframe;
    mat._depthWrite = this._depthWrite;
    mat._depthCompare = this._depthCompare;
    mat._blendState = this._blendState;
    mat._faceStencilInfos = {
      front: { ...this._faceStencilInfos.front },
      back: { ...this._faceStencilInfos.back },
    };
    mat._vertexShaderPath = this._vertexShaderPath;
    mat._fragmentShaderPath = this._fragmentShaderPath;
    mat._computeShaderPath = this._computeShaderPath;

    return mat;
  }
}

// ============================================================
// Material Preset System
// ============================================================

export interface MaterialPreset {
  name: string;
  domain: RenderDomain;
  cullMode: CullMode;
  depthWrite: boolean;
  blendState?: GPUBlendState;
  scalars?: { stage: ShaderStage; name: string; value: MaterialScalar }[];
  vectors?: { stage: ShaderStage; name: string; value: MaterialVector }[];
  vertexShader: string;
  fragmentShader: string;
}

export const MATERIAL_PRESETS: Record<string, MaterialPreset> = {
  'default-opaque': {
    name: 'default-opaque',
    domain: RenderDomain.Forward,
    cullMode: CullMode.Back,
    depthWrite: true,
    scalars: [
      { stage: ShaderStage.Fragment, name: 'metallic', value: 0.0 },
      { stage: ShaderStage.Fragment, name: 'roughness', value: 0.5 },
      { stage: ShaderStage.Fragment, name: 'specular', value: 0.5 },
    ],
    vertexShader: 'builtin/defaultlit.wgsl',
    fragmentShader: 'builtin/defaultlit.wgsl',
  },
  'default-transparent': {
    name: 'default-transparent',
    domain: RenderDomain.Transparency,
    cullMode: CullMode.Back,
    depthWrite: false,
    blendState: {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    },
    scalars: [
      { stage: ShaderStage.Fragment, name: 'metallic', value: 0.0 },
      { stage: ShaderStage.Fragment, name: 'roughness', value: 0.5 },
    ],
    vertexShader: 'builtin/defaultlit.wgsl',
    fragmentShader: 'builtin/defaultlit.wgsl',
  },
  'alpha-cutout': {
    name: 'alpha-cutout',
    domain: RenderDomain.Forward,
    cullMode: CullMode.Back,
    depthWrite: true,
    scalars: [
      { stage: ShaderStage.Fragment, name: 'metallic', value: 0.0 },
      { stage: ShaderStage.Fragment, name: 'roughness', value: 0.5 },
      { stage: ShaderStage.Fragment, name: 'alphaCutoff', value: 0.5 },
    ],
    vertexShader: 'builtin/defaultlit.wgsl',
    fragmentShader: 'builtin/defaultlit.wgsl',
  },
  'shadow-caster': {
    name: 'shadow-caster',
    domain: RenderDomain.Shadow,
    cullMode: CullMode.Back,
    depthWrite: true,
    vertexShader: 'builtin/shadow.wgsl',
    fragmentShader: 'builtin/shadow.wgsl',
  },
  'depth-prepass': {
    name: 'depth-prepass',
    domain: RenderDomain.DepthPrepass,
    cullMode: CullMode.Back,
    depthWrite: true,
    vertexShader: 'builtin/depth-prepass.wgsl',
    fragmentShader: 'builtin/depth-prepass.wgsl',
  },
};

/**
 * Create a material from a preset.
 */
export function createMaterialFromPreset(presetName: string, device?: GPUDevice): Material {
  const preset = MATERIAL_PRESETS[presetName];
  if (!preset) {
    throw new Error(`Unknown material preset: '${presetName}'`);
  }

  const mat = new Material(preset.name);
  mat.domain = preset.domain;
  mat.cullMode = preset.cullMode;
  mat.depthWrite = preset.depthWrite;
  mat.blendState = preset.blendState;
  mat.vertexShaderPath = preset.vertexShader;
  mat.fragmentShaderPath = preset.fragmentShader;

  // Apply scalars
  for (const s of preset.scalars ?? []) {
    mat.setScalar(s.stage, s.name, s.value);
  }

  // Apply vectors
  for (const v of preset.vectors ?? []) {
    mat.setVector(v.stage, v.name, v.value);
  }

  return mat;
}
