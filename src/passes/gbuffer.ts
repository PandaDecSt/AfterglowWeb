export class GBuffer {
  private device: GPUDevice;
  private width = 0;
  private height = 0;

  albedoTexture!: GPUTexture;
  albedoView!: GPUTextureView;
  normalTexture!: GPUTexture;
  normalView!: GPUTextureView;
  materialTexture!: GPUTexture;
  materialView!: GPUTextureView;
  motionTexture!: GPUTexture;
  motionView!: GPUTextureView;
  depthTexture!: GPUTexture;
  depthView!: GPUTextureView;
  depthSampledView!: GPUTextureView;
  depthCopyTexture!: GPUTexture;
  depthCopyView!: GPUTextureView;

  static readonly ALBEDO_FORMAT: GPUTextureFormat = "rgba8unorm";
  static readonly NORMAL_FORMAT: GPUTextureFormat = "rgba16float";
  // MUST stay a float format. material.a carries the raw ShadingModelID (2, 3,
  // 5, 9, ...) and material.g can carry values > 1 (e.g. toon band count), both
  // of which an 8-bit unorm target silently clamps to 1.0 — that clamp is what
  // made every SKIN/HAIR/EYE surface fall through to the TOON branch.
  static readonly MATERIAL_FORMAT: GPUTextureFormat = "rgba16float";
  static readonly MOTION_FORMAT: GPUTextureFormat = "rg16float";
  static readonly DEPTH_COPY_FORMAT: GPUTextureFormat = "rgba16float";
  static readonly DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

  constructor(device: GPUDevice) {
    this.device = device;
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.destroy();
    this.width = width;
    this.height = height;

    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;

    this.albedoTexture = this.device.createTexture({
      label: "gbuffer-albedo",
      size: [width, height],
      format: GBuffer.ALBEDO_FORMAT,
      usage,
    });
    this.albedoView = this.albedoTexture.createView();

    this.normalTexture = this.device.createTexture({
      label: "gbuffer-normal",
      size: [width, height],
      format: GBuffer.NORMAL_FORMAT,
      usage,
    });
    this.normalView = this.normalTexture.createView();

    this.materialTexture = this.device.createTexture({
      label: "gbuffer-material",
      size: [width, height],
      format: GBuffer.MATERIAL_FORMAT,
      usage,
    });
    this.materialView = this.materialTexture.createView();

    this.motionTexture = this.device.createTexture({
      label: "gbuffer-motion",
      size: [width, height],
      format: GBuffer.MOTION_FORMAT,
      usage,
    });
    this.motionView = this.motionTexture.createView();

    this.depthTexture = this.device.createTexture({
      label: "gbuffer-depth",
      size: [width, height],
      format: GBuffer.DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthView = this.depthTexture.createView();
    this.depthSampledView = this.depthView;

    this.depthCopyTexture = this.device.createTexture({
      label: "gbuffer-depth-copy",
      size: [width, height],
      format: GBuffer.DEPTH_COPY_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthCopyView = this.depthCopyTexture.createView();
  }

  get w(): number { return this.width; }
  get h(): number { return this.height; }

  beginGBufferPass(encoder: GPUCommandEncoder, clear = true): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      label: "gbuffer-pass",
      colorAttachments: [
        {
          view: this.albedoView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
        {
          view: this.normalView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
        {
          view: this.materialView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
        {
          view: this.motionView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
        {
          view: this.depthCopyView,
          clearValue: { r: 1, g: 0, b: 0, a: 0 },
          loadOp: clear ? "clear" : "load",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: clear ? "clear" : "load",
        depthStoreOp: "store",
      },
    });
  }

  destroy(): void {
    this.albedoTexture?.destroy();
    this.normalTexture?.destroy();
    this.materialTexture?.destroy();
    this.motionTexture?.destroy();
    this.depthTexture?.destroy();
    this.depthCopyTexture?.destroy();
    this.width = 0;
    this.height = 0;
  }
}

export const GBUFFER_GEOMETRY_WGSL = `
struct GBufferOutput {
  @location(0) albedo: vec4<f32>,
  @location(1) normal: vec4<f32>,
  @location(2) material: vec4<f32>,
  @location(3) motion: vec2<f32>,
  @location(4) depthCopy: vec4<f32>,
};

fn encodeGBuffer(
  baseColor: vec3<f32>,
  ao: f32,
  worldNormal: vec3<f32>,
  metallic: f32,
  roughness: f32,
  emissiveStrength: f32,
  materialID: f32,
  motionVec: vec2<f32>,
  ndcDepth: f32,
) -> GBufferOutput {
  var out: GBufferOutput;
  out.albedo = vec4<f32>(baseColor, ao);
  out.normal = vec4<f32>(normalize(worldNormal), 0.0);
  out.material = vec4<f32>(metallic, roughness, emissiveStrength, materialID);
  out.motion = motionVec;
  out.depthCopy = vec4<f32>(ndcDepth, 0.0, 0.0, 0.0);
  return out;
}

fn decodeGBufferAlbedo(albedo: vec4<f32>) -> vec3<f32> {
  return albedo.rgb;
}

fn decodeGBufferAO(albedo: vec4<f32>) -> f32 {
  return albedo.a;
}

fn decodeGBufferNormal(normal: vec4<f32>) -> vec3<f32> {
  return normalize(normal.rgb);
}

fn decodeGBufferMetallic(material: vec4<f32>) -> f32 {
  return material.r;
}

fn decodeGBufferRoughness(material: vec4<f32>) -> f32 {
  return material.g;
}

fn decodeGBufferEmissive(material: vec4<f32>) -> f32 {
  return material.b;
}

fn decodeGBufferMaterialID(material: vec4<f32>) -> f32 {
  return material.a;
}
`;

/** Standard GBuffer vertex-stage output. Any GBuffer VS must match this. */
export const GBUFFER_VSOUT_WGSL = `
struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) prevClipXY: vec2<f32>,
  @location(4) prevClipW: f32,
  @location(5) curClipXY: vec2<f32>,
  @location(6) curClipW: f32,
};
`;

export interface CharGbufferFSOptions {
  /** WGSL expression (vec3<f32>) producing the base color. */
  albedoExpr: string;
  /** WGSL expression (vec3<f32>) packed into material.rgb = (paramR, paramG, emissive). */
  packExpr: string;
  /**
   * WGSL expression (f32) for the ShadingModelID stamped into material.a.
   * A literal for single-look pipelines, a uniform read when one pipeline
   * serves many materials (glb-viewer stamps a per-material id).
   */
  idExpr: string;
  /** WGSL expression (f32) for the object id stamped into normal.w. */
  objectIdExpr: string;
  /** WGSL expression (f32) written to albedo.a (ambient occlusion). Default "1.0". */
  aoExpr?: string;
  /** WGSL expression (vec3<f32>) for the world normal. Default: interpolated geometric normal. */
  normalExpr?: string;
  /** WGSL declarations emitted before fs_main (extra bindings, helper fns). */
  declarations?: string;
  /**
   * WGSL statements injected at the very top of fs_main, before anything else.
   * This is where texture sampling / normal-map decoding belongs: it runs in
   * UNIFORM control flow, so textureSample and dpdx/dpdy are legal there.
   */
  preludeStmts?: string;
}

// Generic GBuffer fragment-shader factory shared by every demo that writes the
// GBuffer (deferred-pbr character parts, glb-viewer, ...). Stamps albedo +
// ShadingModelID (material.a) + object id (normal.w) so the deferred lighting
// pass can dispatch by id. Everything look-specific arrives as a WGSL
// expression, so this stays a dumb writer with no per-model branching — and
// therefore no uniform-control-flow hazards.
export function makeCharGbufferFS(opts: CharGbufferFSOptions): string {
  const {
    albedoExpr,
    packExpr,
    idExpr,
    objectIdExpr,
    aoExpr = "1.0",
    normalExpr = "in.worldNormal",
    declarations = "",
    preludeStmts = "",
  } = opts;

  return `
${GBUFFER_VSOUT_WGSL}
struct GBufferOutput {
  @location(0) albedo: vec4<f32>,
  @location(1) normal: vec4<f32>,
  @location(2) material: vec4<f32>,
  @location(3) motion: vec2<f32>,
  @location(4) depthCopy: vec4<f32>,
};
${declarations}
@fragment
fn fs_main(in: VSOut, @builtin(front_facing) isFront: bool) -> GBufferOutput {
${preludeStmts}
  var out: GBufferOutput;
  var baseColor = ${albedoExpr};
  out.albedo = vec4<f32>(baseColor, ${aoExpr});
  // Double-sided geometry: flip the normal on back faces so lighting is sane.
  var gbufN = normalize(${normalExpr});
  if (!isFront) { gbufN = -gbufN; }
  // normal.w carries the object ID for the screen-space selection outline.
  out.normal = vec4<f32>(gbufN, ${objectIdExpr});
  // material = (paramR, paramG, emissive, shadingModelId)
  out.material = vec4<f32>(${packExpr}, ${idExpr});
  let prevNDC = in.prevClipXY / in.prevClipW;
  let curNDC = in.curClipXY / in.curClipW;
  out.motion = vec2<f32>((curNDC.x - prevNDC.x) * 0.5, (prevNDC.y - curNDC.y) * 0.5);
  out.depthCopy = vec4<f32>(in.position.z, 0.0, 0.0, 0.0);
  return out;
}
`;
}