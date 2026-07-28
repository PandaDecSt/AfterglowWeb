// AfterglowRender Compute Task System
// Ported from AfterglowComputeTask.h/cpp to TypeScript + WebGPU

// ============================================================
// Enums & Constants
// ============================================================

export enum DispatchFrequency {
  Never = 'never',
  Once = 'once',
  PerFrame = 'per-frame',
}

export enum SSBOUsage {
  Buffer = 'buffer',
  VertexInput = 'vertex-input',
  IndexInput = 'index-input',
  Instancing = 'instancing',
  Indirect = 'indirect',
}

export enum SSBOAccessMode {
  ReadOnly = 'read-only',
  ReadWrite = 'read-write',
}

export enum SSBOTextureMode {
  Unused = 'unused',
  RGBA8 = 'rgba8',
  RG16Float = 'rg16float',
  R32Float = 'r32float',
}

export enum SSBOTextureDimension {
  Texture1D = '1d',
  Texture2D = '2d',
  Texture3D = '3d',
}

export enum SSBOTextureSampleMode {
  NearestClamp = 'nearest-clamp',
  LinearClamp = 'linear-clamp',
  NearestRepeat = 'nearest-repeat',
  LinearRepeat = 'linear-repeat',
}

export enum SSBOInitMode {
  Zero = 'zero',
  StructuredData = 'structured-data',
  ComputeShader = 'compute-shader',
}

export enum DispatchStatus {
  None = 'none',
  Initialized = 'initialized',
  OnceCompleted = 'once-completed',
}

export interface DispatchGroup {
  x: number;
  y: number;
  z: number;
}

export interface SSBOElementLayout {
  /** Size of one element in bytes */
  elementSize: number;
  /** Array of field descriptors for structured access */
  fields?: { name: string; offset: number; format: string }[];
}

// ============================================================
// SSBO Info
// ============================================================

export class SSBOInfo {
  private _name: string;
  private _stage: string = 'compute';
  private _usage: SSBOUsage = SSBOUsage.Buffer;
  private _accessMode: SSBOAccessMode = SSBOAccessMode.ReadWrite;
  private _textureMode: SSBOTextureMode = SSBOTextureMode.Unused;
  private _textureDimension: SSBOTextureDimension = SSBOTextureDimension.Texture2D;
  private _textureSampleMode: SSBOTextureSampleMode = SSBOTextureSampleMode.LinearRepeat;
  private _initMode: SSBOInitMode = SSBOInitMode.Zero;
  private _initResource: string = '';
  private _elementLayout: SSBOElementLayout = { elementSize: 4 };
  private _numElements: number = 0;

  // GPU resources
  private _buffer: GPUBuffer | null = null;
  private _texture: GPUTexture | null = null;
  private _textureView: GPUTextureView | null = null;
  private _sampler: GPUSampler | null = null;

  constructor(name: string) {
    this._name = name;
  }

  get name(): string { return this._name; }

  // Getters/setters
  get stage(): string { return this._stage; }
  set stage(value: string) { this._stage = value; }

  get usage(): SSBOUsage { return this._usage; }
  set usage(value: SSBOUsage) { this._usage = value; }

  get accessMode(): SSBOAccessMode { return this._accessMode; }
  set accessMode(value: SSBOAccessMode) { this._accessMode = value; }

  get textureMode(): SSBOTextureMode { return this._textureMode; }
  set textureMode(value: SSBOTextureMode) { this._textureMode = value; }

  get textureDimension(): SSBOTextureDimension { return this._textureDimension; }
  set textureDimension(value: SSBOTextureDimension) { this._textureDimension = value; }

  get textureSampleMode(): SSBOTextureSampleMode { return this._textureSampleMode; }
  set textureSampleMode(value: SSBOTextureSampleMode) { this._textureSampleMode = value; }

  get initMode(): SSBOInitMode { return this._initMode; }
  set initMode(value: SSBOInitMode) { this._initMode = value; }

  get initResource(): string { return this._initResource; }
  set initResource(value: string) { this._initResource = value; }

  get elementLayout(): SSBOElementLayout { return this._elementLayout; }
  set elementLayout(value: SSBOElementLayout) { this._elementLayout = value; }

  get numElements(): number { return this._numElements; }
  set numElements(value: number) { this._numElements = value; }

  // GPU resources
  get buffer(): GPUBuffer | null { return this._buffer; }
  set buffer(value: GPUBuffer | null) { this._buffer = value; }

  get texture(): GPUTexture | null { return this._texture; }
  set texture(value: GPUTexture | null) { this._texture = value; }

  get textureView(): GPUTextureView | null { return this._textureView; }
  set textureView(value: GPUTextureView | null) { this._textureView = value; }

  get sampler(): GPUSampler | null { return this._sampler; }
  set sampler(value: GPUSampler | null) { this._sampler = value; }

  /** Total buffer size in bytes */
  get bufferSize(): number {
    return this._elementLayout.elementSize * this._numElements;
  }

  /** WebGPU binding type for this SSBO */
  get bindingType(): GPUBufferBindingType {
    return this._accessMode === SSBOAccessMode.ReadWrite ? 'storage' : 'read-only-storage';
  }

  /** WebGPU texture format when used as texture */
  get textureFormat(): GPUTextureFormat {
    switch (this._textureMode) {
      case SSBOTextureMode.RGBA8: return 'rgba8unorm';
      case SSBOTextureMode.RG16Float: return 'rg16float';
      case SSBOTextureMode.R32Float: return 'r32float';
      default: return 'rgba8unorm';
    }
  }

  destroy() {
    if (this._buffer) { this._buffer.destroy(); this._buffer = null; }
    if (this._texture) { this._texture.destroy(); this._texture = null; }
    this._textureView = null;
    this._sampler = null;
  }
}

// ============================================================
// Compute Task Class
// ============================================================

const MAX_FRAME_IN_FLIGHT = 2;

export class ComputeTask {
  private _computeOnly = false;
  private _dispatchFrequency: DispatchFrequency = DispatchFrequency.Never;
  private _inFlightDispatchStatuses: DispatchStatus[] = [DispatchStatus.None, DispatchStatus.None];
  private _computeShaderPath = '';
  private _dispatchGroup: DispatchGroup = { x: 1, y: 1, z: 1 };
  private _ssboInfos: SSBOInfo[] = [];
  private _externalSSBOs: { materialName: string; ssboName: string }[] = [];

  // GPU pipeline
  private _pipeline: GPUComputePipeline | null = null;
  private _bindGroupLayout: GPUBindGroupLayout | null = null;
  private _bindGroups: GPUBindGroup[] = [];

  constructor() {}

  // ============================================================
  // Properties
  // ============================================================

  get computeOnly(): boolean { return this._computeOnly; }
  set computeOnly(value: boolean) { this._computeOnly = value; }

  get dispatchFrequency(): DispatchFrequency { return this._dispatchFrequency; }
  set dispatchFrequency(value: DispatchFrequency) {
    this._dispatchFrequency = value;
    // Reset once-completed status to allow re-dispatch
    if (value === DispatchFrequency.Once) {
      this._inFlightDispatchStatuses = this._inFlightDispatchStatuses.map(s =>
        s === DispatchStatus.OnceCompleted ? DispatchStatus.Initialized : s
      );
    }
  }

  get computeShaderPath(): string { return this._computeShaderPath; }
  set computeShaderPath(value: string) { this._computeShaderPath = value; }

  get dispatchGroup(): DispatchGroup { return this._dispatchGroup; }
  set dispatchGroup(value: DispatchGroup) { this._dispatchGroup = value; }

  // ============================================================
  // SSBO Management
  // ============================================================

  get ssboInfos(): readonly SSBOInfo[] { return this._ssboInfos; }

  appendSSBOInfo(ssboInfo: SSBOInfo) {
    // Special handling for specific usages
    if (ssboInfo.usage === SSBOUsage.IndexInput) {
      ssboInfo.elementLayout = {
        elementSize: 4, // 32-bit indices
        fields: [{ name: 'index', offset: 0, format: 'uint32' }],
      };
    } else if (ssboInfo.usage === SSBOUsage.Indirect) {
      ssboInfo.elementLayout = {
        elementSize: 20, // VkIndexedIndirectCommand
        fields: [
          { name: 'indexCount', offset: 0, format: 'uint32' },
          { name: 'instanceCount', offset: 4, format: 'uint32' },
          { name: 'firstIndex', offset: 8, format: 'uint32' },
          { name: 'vertexOffset', offset: 12, format: 'int32' },
          { name: 'firstInstance', offset: 16, format: 'uint32' },
        ],
      };
    }

    this._ssboInfos.push(ssboInfo);
  }

  removeSSBOInfo(name: string): boolean {
    const index = this._ssboInfos.findIndex(s => s.name === name);
    if (index === -1) return false;
    this._ssboInfos[index].destroy();
    this._ssboInfos.splice(index, 1);
    return true;
  }

  findSSBOInfo(name: string): SSBOInfo | undefined {
    return this._ssboInfos.find(s => s.name === name);
  }

  vertexInputSSBOInfo(): SSBOInfo | undefined {
    return this._ssboInfos.find(s => s.usage === SSBOUsage.VertexInput);
  }

  indexInputSSBOInfo(): SSBOInfo | undefined {
    return this._ssboInfos.find(s => s.usage === SSBOUsage.IndexInput);
  }

  instancingSSBOInfo(): SSBOInfo | undefined {
    return this._ssboInfos.find(s => s.usage === SSBOUsage.Instancing);
  }

  indirectSSBOInfo(): SSBOInfo | undefined {
    return this._ssboInfos.find(s => s.usage === SSBOUsage.Indirect);
  }

  // ============================================================
  // Double Buffering
  // ============================================================

  isMultipleSSBOs(ssboInfo: SSBOInfo): boolean {
    return ssboInfo.accessMode === SSBOAccessMode.ReadWrite;
  }

  numSSBOs(ssboInfo: SSBOInfo): number {
    return this.isMultipleSSBOs(ssboInfo) ? MAX_FRAME_IN_FLIGHT : 1;
  }

  /**
   * Get the shader binding name for an SSBO at a given frame index.
   * For ReadWrite SSBOs: frame 0 -> "NameIn", frame 1 -> "NameOut"
   * For ReadOnly SSBOs: always "Name"
   */
  ssboBindingName(ssboInfo: SSBOInfo, frameIndex: number): string {
    if (!this.isMultipleSSBOs(ssboInfo)) return ssboInfo.name;
    return frameIndex === 0 ? `${ssboInfo.name}In` : `${ssboInfo.name}Out`;
  }

  // ============================================================
  // External SSBOs
  // ============================================================

  get externalSSBOs(): readonly { materialName: string; ssboName: string }[] {
    return this._externalSSBOs;
  }

  addExternalSSBO(materialName: string, ssboName: string) {
    this._externalSSBOs.push({ materialName, ssboName });
  }

  removeExternalSSBO(materialName: string, ssboName: string): boolean {
    const index = this._externalSSBOs.findIndex(
      e => e.materialName === materialName && e.ssboName === ssboName
    );
    if (index === -1) return false;
    this._externalSSBOs.splice(index, 1);
    return true;
  }

  // ============================================================
  // Dispatch Status Management
  // ============================================================

  queryDispatchable(frameIndex: number): boolean {
    const fi = frameIndex % MAX_FRAME_IN_FLIGHT;

    if (this._dispatchFrequency === DispatchFrequency.Never) return false;

    if (this._dispatchFrequency === DispatchFrequency.Once) {
      if (this._inFlightDispatchStatuses[fi] === DispatchStatus.OnceCompleted) return false;
      if (this._inFlightDispatchStatuses[fi] === DispatchStatus.Initialized) {
        this._inFlightDispatchStatuses[fi] = DispatchStatus.OnceCompleted;
        return true;
      }
      return false;
    }

    // PerFrame
    return true;
  }

  setDispatchStatuses(status: DispatchStatus) {
    this._inFlightDispatchStatuses = [status, status];
  }

  getDispatchStatus(frameIndex: number): DispatchStatus {
    return this._inFlightDispatchStatuses[frameIndex % MAX_FRAME_IN_FLIGHT];
  }

  // ============================================================
  // GPU Pipeline Management
  // ============================================================

  createPipeline(device: GPUDevice, shaderCode: string) {
    this._bindGroupLayout = device.createBindGroupLayout({
      label: `compute-task-bgl-${this._computeShaderPath}`,
      entries: this._ssboInfos.map((ssbo, i) => ({
        binding: i,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: ssbo.bindingType as GPUBufferBindingType },
      })),
    });

    this._pipeline = device.createComputePipeline({
      label: `compute-task-${this._computeShaderPath}`,
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this._bindGroupLayout],
      }),
      compute: {
        module: device.createShaderModule({ code: shaderCode }),
        entryPoint: 'cs_main',
      },
    });
  }

  createBindGroups(device: GPUDevice, frameIndex: number) {
    if (!this._bindGroupLayout) return;

    this._bindGroups = [];

    // Create one bind group per double-buffered frame
    for (let fi = 0; fi < MAX_FRAME_IN_FLIGHT; fi++) {
      const entries: GPUBindGroupEntry[] = this._ssboInfos.map((ssbo, i) => {
        const buffer = ssbo.buffer;
        if (!buffer) throw new Error(`SSBO '${ssbo.name}' has no buffer`);

        return {
          binding: i,
          resource: { buffer, offset: 0, size: buffer.size },
        };
      });

      this._bindGroups.push(
        device.createBindGroup({
          label: `compute-task-bg-${fi}`,
          layout: this._bindGroupLayout,
          entries,
        })
      );
    }
  }

  // ============================================================
  // SSBO Buffer Creation
  // ============================================================

  createSSBOBuffers(device: GPUDevice) {
    for (const ssbo of this._ssboInfos) {
      const totalSize = ssbo.bufferSize;
      if (totalSize === 0) continue;

      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

      if (this.isMultipleSSBOs(ssbo)) {
        // Double-buffered: create two buffers
        ssbo.buffer = device.createBuffer({
          label: `ssbo-${ssbo.name}-${0}`,
          size: totalSize,
          usage,
        });
        // Note: The second buffer is created as a separate resource
        // In practice, the caller should manage both buffers
      } else {
        ssbo.buffer = device.createBuffer({
          label: `ssbo-${ssbo.name}`,
          size: totalSize,
          usage,
        });
      }

      // Initialize to zero if needed
      if (ssbo.initMode === SSBOInitMode.Zero) {
        const zeros = new Uint8Array(totalSize);
        device.queue.writeBuffer(ssbo.buffer, 0, zeros);
      }
    }
  }

  // ============================================================
  // Dispatch
  // ============================================================

  dispatch(passEncoder: GPUComputePassEncoder, frameIndex: number) {
    if (!this._pipeline) return;

    const fi = frameIndex % MAX_FRAME_IN_FLIGHT;
    if (fi < this._bindGroups.length) {
      passEncoder.setPipeline(this._pipeline);
      passEncoder.setBindGroup(0, this._bindGroups[fi]);
      passEncoder.dispatchWorkgroups(
        this._dispatchGroup.x,
        this._dispatchGroup.y,
        this._dispatchGroup.z
      );
    }
  }

  // ============================================================
  // Cleanup
  // ============================================================

  destroy() {
    for (const ssbo of this._ssboInfos) {
      ssbo.destroy();
    }
    this._ssboInfos = [];

    // Note: pipeline and bind groups are device-owned, no explicit destroy needed
    this._pipeline = null;
    this._bindGroupLayout = null;
    this._bindGroups = [];
  }
}

// ============================================================
// Pre-built SSBO Layouts
// ============================================================

export function makeIndexSSBOLayout(): SSBOElementLayout {
  return {
    elementSize: 4,
    fields: [{ name: 'index', offset: 0, format: 'uint32' }],
  };
}

export function makeIndexedIndirectSSBOLayout(): SSBOElementLayout {
  return {
    elementSize: 20,
    fields: [
      { name: 'indexCount', offset: 0, format: 'uint32' },
      { name: 'instanceCount', offset: 4, format: 'uint32' },
      { name: 'firstIndex', offset: 8, format: 'uint32' },
      { name: 'vertexOffset', offset: 12, format: 'int32' },
      { name: 'firstInstance', offset: 16, format: 'uint32' },
    ],
  };
}

export function makeVertexSSBOLayout(
  fields: { name: string; offset: number; format: string }[],
  stride: number
): SSBOElementLayout {
  return { elementSize: stride, fields };
}
