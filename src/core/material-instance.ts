// MaterialInstance — a real, data-driven material uniform abstraction.
//
// This is the piece that was missing from the ported `Material` class:
// the original AfterglowRender *auto-generates the shader code* for a
// material's parameters ("Auto generate shader codes" in its design doc).
// The ported Material only stored values and packed bytes; it never
// produced the matching WGSL struct or bind group layout.
//
// MaterialInstance closes that gap: give it a list of typed fields and it
// will (1) emit the WGSL `struct` that matches its packed layout,
// (2) build the GPUBindGroupLayout, and (3) pack + upload the uniform
// buffer. The same blueprint can later be compiled to HLSL / ShaderLab for
// Unity export — which is the whole point of the project's "material graph
// format + multi-backend compiler" direction.

export type MaterialFieldType = "f32" | "vec2" | "vec3" | "vec4" | "mat4";

export type MaterialFieldValue =
  | number
  | [number, number]
  | [number, number, number]
  | [number, number, number, number]
  | Float32Array
  | number[];

export interface MaterialField {
  name: string;
  type: MaterialFieldType;
  value: MaterialFieldValue;
}

/**
 * Serializable description of a material's uniform block.
 * This is the in-memory form of the future `.mat` JSON file.
 */
export interface MaterialBlueprint {
  name: string;
  /** Bind group slot the material's uniform occupies (default 0). */
  group: number;
  /** Which shader stages can see this uniform block. */
  visibility?: GPUShaderStageFlags;
  /**
   * Override the WGSL struct type name. By default it is `Mat_<name>`.
   * Use this when the shader already declares the binding variable and
   * you only want the struct definition (e.g. to match `struct Mat`).
   */
  structName?: string;
  fields: MaterialField[];
}

// WGSL uniform address-space layout rules (std uniform layout):
//   f32  -> align 4,  size 4
//   vec2 -> align 8,  size 8
//   vec3 -> align 16, size 12  (trailing 4 bytes are implicit padding)
//   vec4 -> align 16, size 16
//   mat4 -> align 16, size 64
const FIELD_ALIGN_BYTES: Record<MaterialFieldType, number> = {
  f32: 4,
  vec2: 8,
  vec3: 16,
  vec4: 16,
  mat4: 16,
};

const FIELD_SIZE_BYTES: Record<MaterialFieldType, number> = {
  f32: 4,
  vec2: 8,
  vec3: 12,
  vec4: 16,
  mat4: 64,
};

function wgslType(t: MaterialFieldType): string {
  switch (t) {
    case "f32": return "f32";
    case "vec2": return "vec2<f32>";
    case "vec3": return "vec3<f32>";
    case "vec4": return "vec4<f32>";
    case "mat4": return "mat4x4<f32>";
  }
}

function writeField(data: Float32Array, floatOffset: number, f: MaterialField): void {
  const v = f.value;
  switch (f.type) {
    case "f32":
      data[floatOffset] = v as number;
      break;
    case "vec2": {
      const a = v as [number, number];
      data[floatOffset] = a[0];
      data[floatOffset + 1] = a[1];
      break;
    }
    case "vec3": {
      const a = v as [number, number, number];
      data[floatOffset] = a[0];
      data[floatOffset + 1] = a[1];
      data[floatOffset + 2] = a[2];
      break;
    }
    case "vec4": {
      const a = v as [number, number, number, number];
      data[floatOffset] = a[0];
      data[floatOffset + 1] = a[1];
      data[floatOffset + 2] = a[2];
      data[floatOffset + 3] = a[3];
      break;
    }
    case "mat4": {
      const m = v as Float32Array | number[];
      for (let i = 0; i < 16; i++) data[floatOffset + i] = m[i];
      break;
    }
  }
}

export class MaterialInstance {
  readonly name: string;
  readonly group: number;
  readonly fields: MaterialField[];
  readonly visibility: GPUShaderStageFlags;
  private readonly structNameOverride?: string;

  private buffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;

  constructor(blueprint: MaterialBlueprint) {
    this.name = blueprint.name;
    this.group = blueprint.group;
    this.fields = blueprint.fields.map((f) => ({ ...f }));
    this.visibility = blueprint.visibility ?? GPUShaderStage.FRAGMENT;
    this.structNameOverride = blueprint.structName;
  }

  /** WGSL struct type name, derived from the material name. */
  get structName(): string {
    return this.structNameOverride ?? `Mat_${this.name}`;
  }

  /** WGSL uniform variable name, derived from the material name. */
  get instanceName(): string {
    return `mat_${this.name}`;
  }

  /** Auto-generate the WGSL `struct` declaration matching the packed layout. */
  generateWGSLStruct(): string {
    const lines = this.fields.map((f) => `  ${f.name}: ${wgslType(f.type)},`);
    return `struct ${this.structName} {\n${lines.join("\n")}\n};`;
  }

  /** The `var<uniform>` binding declaration for use inside a shader. */
  generateWGSLBinding(): string {
    return `@group(${this.group}) @binding(0) var<uniform> ${this.instanceName}: ${this.structName};`;
  }

  /** Combined struct + binding, ready to prepend to a fragment/vertex shader. */
  generateWGSL(): string {
    return `${this.generateWGSLStruct()}\n${this.generateWGSLBinding()}`;
  }

  /** Total uniform block size in bytes, following WGSL uniform layout rules. */
  get byteLength(): number {
    let offset = 0;
    let align = 1;
    for (const f of this.fields) {
      const a = FIELD_ALIGN_BYTES[f.type];
      offset = Math.ceil(offset / a) * a;
      offset += FIELD_SIZE_BYTES[f.type];
      align = Math.max(align, a);
    }
    return Math.ceil(offset / align) * align;
  }

  /**
   * Pack fields into a Float32Array following WGSL uniform layout rules.
   * vec3 occupies 16 bytes (4 floats) so callers must pad after it.
   */
  buildUniformData(): Float32Array {
    const data = new Float32Array(this.byteLength / 4);
    let floatOffset = 0;
    for (const f of this.fields) {
      const alignFloats = FIELD_ALIGN_BYTES[f.type] / 4;
      floatOffset = Math.ceil(floatOffset / alignFloats) * alignFloats;
      writeField(data, floatOffset, f);
      floatOffset += FIELD_SIZE_BYTES[f.type] / 4;
    }
    return data;
  }

  /** Upload (or re-upload) the packed data to a GPU uniform buffer. */
  upload(device: GPUDevice): GPUBuffer {
    const data = this.buildUniformData();
    if (!this.buffer) {
      this.buffer = device.createBuffer({
        label: `ubo-${this.name}`,
        size: Math.max(data.byteLength, 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    device.queue.writeBuffer(this.buffer, 0, data as unknown as GPUAllowSharedBufferSource);
    return this.buffer;
  }

  setField(name: string, value: MaterialFieldValue): void {
    const f = this.fields.find((x) => x.name === name);
    if (!f) throw new Error(`Material '${this.name}' has no field '${name}'`);
    f.value = value;
  }

  get uniformBuffer(): GPUBuffer | null {
    return this.buffer;
  }

  /**
   * Create the bind group for this material bound to a specific pipeline.
   * Uploads the buffer first if needed. Recreate after any pipeline rebuild.
   */
  createBindGroup(device: GPUDevice, pipeline: GPURenderPipeline): GPUBindGroup {
    if (!this.buffer) this.upload(device);
    this.bindGroup = device.createBindGroup({
      label: `bg-${this.name}`,
      layout: pipeline.getBindGroupLayout(this.group),
      entries: [{ binding: 0, resource: { buffer: this.buffer! } }],
    });
    return this.bindGroup;
  }

  get bindGroupCache(): GPUBindGroup | null {
    return this.bindGroup;
  }

  destroy(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.bindGroup = null;
  }
}
