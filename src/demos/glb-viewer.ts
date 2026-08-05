// GLB Viewer — a *shell* demo, not a renderer.
//
// This file used to own a 200-line forward fragment shader with a PBR branch
// and five hand-rolled toon branches. That was the architectural bug: the look
// of a character lived inside one viewer, so "switch to realistic" meant
// editing demo code and nothing else in the project could reuse it.
//
// Now the demo does exactly three things:
//   1. load a GLB and classify each material by name (body / face / hair / ...)
//   2. let a RenderPreset map each category to a ShadingModelID
//   3. stamp that id into the GBuffer and hand everything to DeferredLightingPass
//
// All shading lives in the kernel (passes/deferred-lighting.ts +
// core/shading-registry.ts). Swapping "Endfield Default" for "Realistic
// Character (SSS)" re-skins Scarlet without touching a line of shader code —
// which is the whole point of the north star: one deferred pipeline, one set of
// lights, both a photoreal and an anime character in it.

import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { mat4, type Mat4 } from "wgpu-matrix";
import { loadGLTF, interleaveMesh, type MeshData, type MaterialData } from "../utils/gltf-loader";
import type { RenderPass } from "../core/renderer";
import { MaterialInstance, type MaterialBlueprint } from "../core/material-instance";
import {
  MaterialCategory, classifyMaterial, RenderPreset, loadPresets, savePresets, PRESET_VERSION,
} from "../render/render-preset";
import { GBuffer, GBUFFER_VSOUT_WGSL, makeCharGbufferFS } from "../passes/gbuffer";
import { DeferredLightingPass } from "../passes/deferred-lighting";
import { TonemapBlitPass } from "../passes/tonemap-blit";
import { LightScene, createDirectionalLight, type DirectionalLight } from "../scene/light";
import { ShadingModel } from "../core/shading-model";
import {
  shadingModelOptions, packForModel, TOON_BODY, TOON_EYELASH, UNLIT,
} from "../core/shading-registry";

// Single source of truth for a GLB material's uniform block. The WGSL
// `struct MatUniforms` is generated from this blueprint, so the shader's struct
// and the CPU packing can never drift apart.
//
//   params  = (packR, packG, hasBaseColorTex, lightmapStrength)
//   params2 = (alphaCutoff, shadingModelId, outlineWidth, hasNormalTex)
//   params3 = (objectId, _, _, _)
//
// packR/packG are whatever the *chosen shading model* wants in GBuffer
// material.rg (metallic/roughness for STANDARD, sssStrength/roughness for SKIN,
// rimWidth for the toon variants, ...). That decision is made on the CPU by
// packForModel(), which is why the GBuffer fragment shader below has zero
// per-model branching — and therefore no uniform-control-flow hazards.
const GLB_MAT_BLUEPRINT: MaterialBlueprint = {
  name: "glb",
  group: 1,
  structName: "MatUniforms",
  fields: [
    { name: "baseColorFactor", type: "vec4", value: [1, 1, 1, 1] },
    { name: "params", type: "vec4", value: [0, 0.5, 0, 0] },
    { name: "params2", type: "vec4", value: [0, TOON_BODY, 0.015, 0] },
    { name: "params3", type: "vec4", value: [1, 0, 0, 0] },
  ],
};
const GLB_MAT_WGSL = new MaterialInstance(GLB_MAT_BLUEPRINT).generateWGSLStruct();

function buildGLBMaterialInstance(mat: MaterialData, outlineWidth: number, hasNormalTex: number, objectId: number): MaterialInstance {
  const instance = new MaterialInstance(GLB_MAT_BLUEPRINT);
  const hasBaseColorTex = mat.baseColorImage ? 1 : 0;
  const lightmapStrength = mat.occlusionImage ? mat.occlusionStrength : 0;
  const [pr, pg] = packForModel(TOON_BODY, mat.metallicFactor, mat.roughnessFactor);
  instance.setField("baseColorFactor", [...mat.baseColorFactor]);
  instance.setField("params", [pr, pg, hasBaseColorTex, lightmapStrength]);
  instance.setField("params2", [0.0, TOON_BODY, outlineWidth, hasNormalTex]);
  instance.setField("params3", [objectId, 0, 0, 0]);
  return instance;
}

// === Shaders ===
const FRAME_UNIFORMS_WGSL = `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  prevModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
`;
/** floats in FrameUniforms: 5 mat4 + 1 vec4 */
const FRAME_FLOATS = 5 * 16 + 4;

// GBuffer vertex stage. Emits the standard VSOut (including the previous-frame
// clip position) so motion vectors come out for free.
const gbufferVS = `
${FRAME_UNIFORMS_WGSL}
${GBUFFER_VSOUT_WGSL}
@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = frame.model * vec4<f32>(pos, 1.0);
  let clip = frame.viewProj * worldPos;
  out.position = clip;
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((frame.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  let prevClip = frame.prevViewProj * (frame.prevModel * vec4<f32>(pos, 1.0));
  out.prevClipXY = prevClip.xy;
  out.prevClipW = prevClip.w;
  out.curClipXY = clip.xy;
  out.curClipW = clip.w;
  return out;
}
`;

const GBUFFER_FS_DECLARATIONS = `
@group(1) @binding(0) var<uniform> mat: MatUniforms;
@group(1) @binding(1) var baseColorTex: texture_2d<f32>;
@group(1) @binding(2) var lightmapTex: texture_2d<f32>;
@group(1) @binding(3) var texSampler: sampler;
@group(1) @binding(4) var normalTex: texture_2d<f32>;

// Rebuild a tangent frame from screen-space derivatives (Mikkelsen / Schüler
// "bump mapping unparametrized surfaces"). Lets us apply a tangent-space normal
// map without per-vertex tangents, which GLB characters rarely carry.
// MUST be called in uniform control flow.
fn cotangentFrame(N: vec3<f32>, p: vec3<f32>, uv: vec2<f32>) -> mat3x3<f32> {
  let dp1 = dpdx(p);
  let dp2 = dpdy(p);
  let duv1 = dpdx(uv);
  let duv2 = dpdy(uv);
  let dp2perp = cross(dp2, N);
  let dp1perp = cross(N, dp1);
  let T = dp2perp * duv1.x + dp1perp * duv2.x;
  let B = dp2perp * duv1.y + dp1perp * duv2.y;
  let invmax = inverseSqrt(max(dot(T, T), dot(B, B)));
  return mat3x3<f32>(T * invmax, B * invmax, N);
}
`;

// Everything look-specific is a WGSL *expression* handed to the shared factory,
// so this stays a dumb GBuffer writer. Texture sampling and dpdx/dpdy live in
// the prelude, which the factory injects at the very top of fs_main — i.e. in
// UNIFORM control flow. Putting them behind `if (mat.params.z > 0.5)` is what
// used to make Tint reject the whole pipeline.
const GBUFFER_FS_PRELUDE = `
  let baseTex = textureSample(baseColorTex, texSampler, in.uv);
  let normTex = textureSample(normalTex, texSampler, in.uv);
  let lightTex = textureSample(lightmapTex, texSampler, in.uv);

  let geoN = normalize(in.worldNormal);
  let TBN = cotangentFrame(geoN, in.worldPos, in.uv);
  let mappedN = normalize(TBN * (normTex.xyz * 2.0 - vec3<f32>(1.0)));
  // No normal map -> blend factor 0, so the sampled value is discarded rather
  // than skipped. Same result, but the sample stays unconditional.
  let shadedN = normalize(mix(geoN, mappedN, select(0.0, 1.0, mat.params2.w > 0.5)));

  let useTex = select(0.0, 1.0, mat.params.z > 0.5);
  let texAlpha = mix(1.0, baseTex.a, useTex) * mat.baseColorFactor.a;
  let cutoff = mat.params2.x;
  if (cutoff > 0.001 && texAlpha < cutoff) { discard; }

  // This asset ships a baked lightmap in the glTF occlusion slot, so it
  // modulates albedo directly (matching the old forward look). A true AO map
  // would instead go to albedo.a, which the lighting pass applies to
  // ambient/IBL only.
  let lightmap = mix(vec3<f32>(1.0), lightTex.rgb, clamp(mat.params.w, 0.0, 1.0));
  let albedoRGB = mix(mat.baseColorFactor.rgb, baseTex.rgb * mat.baseColorFactor.rgb, useTex) * lightmap;
`;

const gbufferFS = GLB_MAT_WGSL + "\n" + makeCharGbufferFS({
  declarations: GBUFFER_FS_DECLARATIONS,
  preludeStmts: GBUFFER_FS_PRELUDE,
  albedoExpr: "albedoRGB",
  normalExpr: "shadedN",
  aoExpr: "1.0",
  packExpr: "vec3<f32>(mat.params.x, mat.params.y, 0.0)",
  // The one line that makes the whole preset system work: the shading model is
  // data, uploaded per material, not a shader variant.
  idExpr: "mat.params2.y",
  objectIdExpr: "mat.params3.x",
});

// Outline = inverted-hull shell written into the GBuffer as an UNLIT surface.
// Doing it here rather than as a separate forward pass means the outline gets
// correct depth against the character for free and the lighting pass composites
// it in one go.
const outlineVS = `
${GLB_MAT_WGSL}
${FRAME_UNIFORMS_WGSL}
@group(1) @binding(0) var<uniform> mat: MatUniforms;
${GBUFFER_VSOUT_WGSL}

fn expandOutline(pos: vec3<f32>, normal: vec3<f32>, model: mat4x4<f32>, invTransModel: mat4x4<f32>) -> vec4<f32> {
  let outlineWidth = mat.params2.z;
  let worldPos = (model * vec4<f32>(pos, 1.0)).xyz;
  let worldNormal = normalize((invTransModel * vec4<f32>(normal, 0.0)).xyz);
  // Endfield-style: expand along a view/normal blend, scaled by distance so the
  // outline keeps a constant screen-space width.
  let camToVert = worldPos - frame.cameraPosition.xyz;
  let dist = length(camToVert);
  let viewDir = camToVert / max(dist, 0.001);
  let expandDir = normalize(viewDir * 0.5 + worldNormal * 0.5);
  return vec4<f32>(worldPos + expandDir * outlineWidth * (dist * 0.04), 1.0);
}

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let expanded = expandOutline(pos, normal, frame.model, frame.invTransModel);
  let clip = frame.viewProj * expanded;
  out.position = clip;
  out.worldPos = expanded.xyz;
  out.worldNormal = normalize((frame.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  let prevExpanded = expandOutline(pos, normal, frame.prevModel, frame.invTransModel);
  let prevClip = frame.prevViewProj * prevExpanded;
  out.prevClipXY = prevClip.xy;
  out.prevClipW = prevClip.w;
  out.curClipXY = clip.xy;
  out.curClipW = clip.w;
  return out;
}
`;

// Near-black in LINEAR space. The old forward shader wrote 0.02 straight to the
// swapchain after its own gamma pass; now the shared tonemap applies gamma once
// at the end, so the linear value has to be much smaller to read as black.
const outlineFS = makeCharGbufferFS({
  albedoExpr: "vec3<f32>(0.002, 0.002, 0.0024)",
  packExpr: "vec3<f32>(0.0, 0.0, 0.0)",
  idExpr: `${UNLIT}.0`,
  objectIdExpr: "0.0",
});

// === Main Demo Class ===
interface MeshBuffers {
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  use32bit: boolean;
  materialIndex: number;
  name: string;
  visible: boolean;
  vertexData: Float32Array;
  indexData: Uint16Array | Uint32Array;
}

interface MaterialResources {
  bindGroup: GPUBindGroup;
  materialInstance: MaterialInstance;
  name: string;
  category: MaterialCategory;
  objectId: number;
  baseColorFactor: [number, number, number, number];
  metallic: number;
  roughness: number;
  hasBaseColorTex: number;
  lightmapStrength: number;
  hasLightmap: boolean;
  hasNormalTex: number;
  previewImage: ImageBitmap | null;
  alphaCutoff: number;
  /** ShadingModelID — see core/shading-registry. Chosen by the active preset. */
  shadingModelId: number;
  outlineWidth: number;
}

/** Models whose BxDF bakes its own shadow floor and wants flat, unmuddied colors. */
function isToonModel(id: number): boolean {
  return id === ShadingModel.TOON || (id >= TOON_BODY && id <= TOON_EYELASH);
}

export class GLBViewerDemo implements Demo {
  label = "GLB Viewer";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;
  private gbufferPipeline!: GPURenderPipeline;
  private outlinePipeline!: GPURenderPipeline;
  private frameUbo!: GPUBuffer;
  private meshBuffers: MeshBuffers[] = [];
  private materialResources: MaterialResources[] = [];
  private frameData = new Float32Array(FRAME_FLOATS);
  private loaded = false;
  private totalTriangles = 0;
  private modelName = "";
  private defaultTexture!: GPUTexture;
  private defaultSampler!: GPUSampler;
  private frameBindGroup!: GPUBindGroup;
  private presets: RenderPreset[] = [];

  // --- Deferred kernel ---
  private gbuffer!: GBuffer;
  private lightScene!: LightScene;
  private keyLight!: DirectionalLight;
  private deferredLighting!: DeferredLightingPass;
  private tonemap!: TonemapBlitPass;
  private hdrTexture: GPUTexture | null = null;
  private hdrView: GPUTextureView | null = null;
  private prevViewProj: Mat4 = mat4.identity(mat4.create());
  private prevModel: Mat4 = mat4.identity(mat4.create());

  rotationSpeed = 0.3;
  showOutline = true;
  globalOutlineWidth = 0.015;
  private activePresetName = "";

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;
    this.presets = loadPresets();

    this.defaultTexture = this.device.createTexture({
      label: "default-white",
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture(
      { texture: this.defaultTexture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1]
    );
    this.defaultSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      maxAnisotropy: 4,
    });

    this.frameUbo = this.device.createBuffer({
      label: "glb-frame-ubo",
      size: FRAME_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- Deferred kernel wiring -------------------------------------------
    this.gbuffer = new GBuffer(this.device);
    this.lightScene = new LightScene();
    this.keyLight = createDirectionalLight([-0.5, -1.0, -0.3], [1, 1, 1], 1.0);
    this.keyLight.castShadow = false;
    this.lightScene.addLight(this.keyLight);
    // A soft sky fill so the realistic presets are not lit by the key light
    // alone (this demo has no environment probe yet).
    this.lightScene.ambientColor = [0.35, 0.38, 0.45];
    this.lightScene.ambientIntensity = 0.5;
    this.deferredLighting = new DeferredLightingPass(this.device, this.lightScene, "rgba16float");
    // No IBL cubemap here, so tell the pass not to reserve headroom for one —
    // otherwise ambient gets scaled to zero and STANDARD surfaces go black.
    this.deferredLighting.envIntensity = 0.0;
    this.tonemap = new TonemapBlitPass(this.device, this.format);

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const vsModule = this.device.createShaderModule({ label: "glb-gbuffer-vs", code: gbufferVS });
    const fsModule = this.device.createShaderModule({ label: "glb-gbuffer-fs", code: gbufferFS });
    const outlineVsModule = this.device.createShaderModule({ label: "glb-outline-vs", code: outlineVS });
    const outlineFsModule = this.device.createShaderModule({ label: "glb-outline-fs", code: outlineFS });

    const frameBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });
    const matBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [frameBGL, matBGL] });
    const gbufferTargets: GPUColorTargetState[] = [
      { format: GBuffer.ALBEDO_FORMAT },
      { format: GBuffer.NORMAL_FORMAT },
      { format: GBuffer.MATERIAL_FORMAT },
      { format: GBuffer.MOTION_FORMAT },
      { format: GBuffer.DEPTH_COPY_FORMAT },
    ];

    this.gbufferPipeline = this.device.createRenderPipeline({
      label: "glb-gbuffer",
      layout: pipelineLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: gbufferTargets },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: GBuffer.DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });

    this.outlinePipeline = this.device.createRenderPipeline({
      label: "glb-outline",
      layout: pipelineLayout,
      vertex: { module: outlineVsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outlineFsModule, entryPoint: "fs_main", targets: gbufferTargets },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: GBuffer.DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less", depthBias: 5 },
    });

    this.frameBindGroup = this.device.createBindGroup({
      layout: frameBGL,
      entries: [{ binding: 0, resource: { buffer: this.frameUbo } }],
    });

    try {
      const model = await loadGLTF("/SB_X_Nikke_Scarlet.glb");
      this.modelName = model.name;

      console.log(`[GLBViewer] Materials found: ${model.materials.length}`);
      model.materials.forEach((m, i) => {
        const cat = classifyMaterial(m.name);
        console.log(`[GLBViewer]   Mat[${i}] "${m.name}" [${cat}] | diffuse:${!!m.baseColorImage} | lightmap:${!!m.occlusionImage}`);
      });

      for (const mat of model.materials) {
        this.createMaterialResources(mat, matBGL);
      }
      if (model.materials.length === 0) {
        this.createMaterialResources({
          name: "default",
          baseColorFactor: [0.8, 0.75, 0.7, 1],
          metallicFactor: 0.0,
          roughnessFactor: 0.7,
          baseColorImage: null,
          metallicRoughnessImage: null,
          normalImage: null,
          occlusionImage: null,
          occlusionStrength: 0,
        }, matBGL);
      }

      for (const mesh of model.meshes) {
        this.uploadMesh(mesh);
      }

      this.applyPreset(this.presets[0]);

      this.loaded = true;
      console.log(`[GLBViewer] Loaded ${model.name}: ${model.meshes.length} meshes, ${model.materials.length} materials, ${this.totalTriangles.toLocaleString()} triangles`);
    } catch (e) {
      console.error("[GLBViewer] Failed to load model:", e);
    }
  }

  private createMaterialResources(matData: MaterialData, matBGL: GPUBindGroupLayout) {
    const baseColorView = matData.baseColorImage
      ? this.createTextureFromBitmap(matData.baseColorImage, `baseColor-${matData.name}`, "rgba8unorm-srgb").createView()
      : this.defaultTexture.createView();
    const lightmapView = matData.occlusionImage
      ? this.createTextureFromBitmap(matData.occlusionImage, `lightmap-${matData.name}`, "rgba8unorm").createView()
      : this.defaultTexture.createView();
    const normalView = matData.normalImage
      ? this.createTextureFromBitmap(matData.normalImage, `normal-${matData.name}`, "rgba8unorm").createView()
      : this.defaultTexture.createView();
    const hasNormalTex = matData.normalImage ? 1 : 0;

    const category = classifyMaterial(matData.name);
    // normal.w object id — 0 is reserved for the outline shell / background.
    const objectId = this.materialResources.length + 1;

    // MaterialInstance owns the uniform buffer + packing + upload.
    const materialInstance = buildGLBMaterialInstance(matData, this.globalOutlineWidth, hasNormalTex, objectId);
    materialInstance.upload(this.device);

    const bindGroup = this.device.createBindGroup({
      layout: matBGL,
      entries: [
        { binding: 0, resource: { buffer: materialInstance.uniformBuffer! } },
        { binding: 1, resource: baseColorView },
        { binding: 2, resource: lightmapView },
        { binding: 3, resource: this.defaultSampler },
        { binding: 4, resource: normalView },
      ],
    });

    this.materialResources.push({
      bindGroup,
      materialInstance,
      name: matData.name,
      category,
      objectId,
      baseColorFactor: matData.baseColorFactor,
      metallic: matData.metallicFactor,
      roughness: matData.roughnessFactor,
      hasBaseColorTex: matData.baseColorImage ? 1 : 0,
      lightmapStrength: matData.occlusionImage ? matData.occlusionStrength : 0,
      hasLightmap: !!matData.occlusionImage,
      hasNormalTex,
      previewImage: matData.baseColorImage ?? null,
      alphaCutoff: 0.0,
      shadingModelId: TOON_BODY,
      outlineWidth: this.globalOutlineWidth,
    });
  }

  private createTextureFromBitmap(bitmap: ImageBitmap, label: string, format: GPUTextureFormat = "rgba8unorm"): GPUTexture {
    const texture = this.device.createTexture({
      label,
      size: [bitmap.width, bitmap.height],
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      mipLevelCount: 1,
    });
    this.device.queue.copyExternalImageToTexture(
      { source: bitmap },
      { texture },
      [bitmap.width, bitmap.height]
    );
    return texture;
  }

  private uploadMesh(mesh: MeshData) {
    const { vertices, indices } = interleaveMesh(mesh);
    const use32bit = indices instanceof Uint32Array;

    const vertexBuffer = this.device.createBuffer({
      label: "glb-vb",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();

    const indexBufferSize = Math.ceil(indices.byteLength / 4) * 4;
    const indexBuffer = this.device.createBuffer({
      label: "glb-ib",
      size: indexBufferSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    if (use32bit) {
      new Uint32Array(indexBuffer.getMappedRange()).set(indices as Uint32Array);
    } else {
      new Uint16Array(indexBuffer.getMappedRange()).set(indices as Uint16Array);
    }
    indexBuffer.unmap();

    this.totalTriangles += indices.length / 3;
    this.meshBuffers.push({ vertexBuffer, indexBuffer, indexCount: indices.length, use32bit, materialIndex: mesh.materialIndex, name: mesh.name, visible: true, vertexData: vertices, indexData: indices });
  }

  private recalculateNormals(mb: MeshBuffers) {
    const verts = mb.vertexData;
    const indices = mb.indexData;
    const stride = 8; // pos(3) + normal(3) + uv(2)
    const vertexCount = verts.length / stride;

    // Zero out all normals
    for (let i = 0; i < vertexCount; i++) {
      verts[i * stride + 3] = 0;
      verts[i * stride + 4] = 0;
      verts[i * stride + 5] = 0;
    }

    // Accumulate face normals per vertex
    for (let i = 0; i < indices.length; i += 3) {
      const i0 = indices[i] * stride;
      const i1 = indices[i + 1] * stride;
      const i2 = indices[i + 2] * stride;

      const ax = verts[i1] - verts[i0], ay = verts[i1 + 1] - verts[i0 + 1], az = verts[i1 + 2] - verts[i0 + 2];
      const bx = verts[i2] - verts[i0], by = verts[i2 + 1] - verts[i0 + 1], bz = verts[i2 + 2] - verts[i0 + 2];

      // Cross product (face normal, not normalized = area-weighted)
      const nx = ay * bz - az * by;
      const ny = az * bx - ax * bz;
      const nz = ax * by - ay * bx;

      for (const idx of [i0, i1, i2]) {
        verts[idx + 3] += nx;
        verts[idx + 4] += ny;
        verts[idx + 5] += nz;
      }
    }

    // Normalize
    for (let i = 0; i < vertexCount; i++) {
      const base = i * stride + 3;
      const x = verts[base], y = verts[base + 1], z = verts[base + 2];
      const len = Math.sqrt(x * x + y * y + z * z);
      if (len > 1e-8) {
        verts[base] = x / len;
        verts[base + 1] = y / len;
        verts[base + 2] = z / len;
      }
    }

    // Upload to GPU
    this.device.queue.writeBuffer(mb.vertexBuffer, 0, verts as unknown as GPUAllowSharedBufferSource);
    console.log(`[GLBViewer] Recalculated normals for "${mb.name}" (${vertexCount} verts)`);
  }

  private ensureTargets() {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (w === 0 || h === 0) return;
    this.gbuffer.resize(w, h);
    if (this.hdrTexture && this.hdrTexture.width === w && this.hdrTexture.height === h) return;
    this.hdrTexture?.destroy();
    this.hdrTexture = this.device.createTexture({
      label: "glb-hdr",
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrView = this.hdrTexture.createView();
  }

  update(time: number) {
    if (!this.loaded) return;

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    // camera.getViewProjectionMatrix() hands back a scratch matrix, so snapshot
    // it before anything (including the lighting pass) stores a reference.
    const viewProj = mat4.copy(this.camera.getViewProjectionMatrix(w / h));
    const model = mat4.rotationY(time * this.rotationSpeed);
    const invTrans = mat4.transpose(mat4.inverse(model));
    const invViewProj = mat4.inverse(viewProj);

    const d = this.frameData;
    d.set(viewProj as unknown as ArrayLike<number>, 0);
    d.set(model as unknown as ArrayLike<number>, 16);
    d.set(invTrans as unknown as ArrayLike<number>, 32);
    d.set(this.prevViewProj as unknown as ArrayLike<number>, 48);
    d.set(this.prevModel as unknown as ArrayLike<number>, 64);
    d[80] = this.camera.position[0];
    d[81] = this.camera.position[1];
    d[82] = this.camera.position[2];
    d[83] = 1;
    this.device.queue.writeBuffer(this.frameUbo, 0, d as unknown as GPUAllowSharedBufferSource);

    this.deferredLighting.update(
      viewProj,
      [this.camera.position[0], this.camera.position[1], this.camera.position[2]],
      invViewProj,
      w,
      h,
      this.camera.near,
      this.camera.far,
    );

    this.prevViewProj = viewProj;
    this.prevModel = model;
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        if (!this.loaded || this.meshBuffers.length === 0) return;
        this.ensureTargets();
        if (!this.hdrView) return;

        // --- 1. Geometry -> GBuffer -----------------------------------------
        const pass = this.gbuffer.beginGBufferPass(encoder);
        pass.setBindGroup(0, this.frameBindGroup);

        // Outline shell first: it is written as an UNLIT surface, so depth sorts
        // it behind the character automatically.
        if (this.showOutline) {
          pass.setPipeline(this.outlinePipeline);
          for (const mb of this.meshBuffers) {
            if (!mb.visible) continue;
            const matRes = this.materialResources[Math.min(mb.materialIndex, this.materialResources.length - 1)];
            if (matRes.outlineWidth < 0.001) continue;
            pass.setBindGroup(1, matRes.bindGroup);
            pass.setVertexBuffer(0, mb.vertexBuffer);
            pass.setIndexBuffer(mb.indexBuffer, mb.use32bit ? "uint32" : "uint16");
            pass.drawIndexed(mb.indexCount);
          }
        }

        pass.setPipeline(this.gbufferPipeline);
        for (const mb of this.meshBuffers) {
          if (!mb.visible) continue;
          const matRes = this.materialResources[Math.min(mb.materialIndex, this.materialResources.length - 1)];
          pass.setBindGroup(1, matRes.bindGroup);
          pass.setVertexBuffer(0, mb.vertexBuffer);
          pass.setIndexBuffer(mb.indexBuffer, mb.use32bit ? "uint32" : "uint16");
          pass.drawIndexed(mb.indexCount);
        }
        pass.end();

        // --- 2. One deferred pass shades every model in the GBuffer ---------
        this.deferredLighting.execute(encoder, this.gbuffer, this.hdrView);

        // --- 3. HDR -> swapchain --------------------------------------------
        this.tonemap.execute(encoder, this.hdrView, view);
      },
    }];
  }

  // === Preset System ===
  // A preset is the rendering scheme: it decides the shading model per material
  // category *and* the exposure/key-light regime those models were authored
  // for. Applying one is the only way the demo changes how anything looks.
  applyPreset(preset: RenderPreset) {
    if (!preset) return;
    for (const matRes of this.materialResources) {
      matRes.shadingModelId = preset.mapping[matRes.category] ?? preset.mapping.other ?? TOON_BODY;
      this.updateMatUbo(matRes);
    }
    this.activePresetName = preset.name;

    // Toon looks are authored at a key-light intensity of ~1 and get muddied by
    // the ACES curve; PBR wants a brighter key and the filmic rolloff.
    const toonHeavy = this.materialResources.length > 0
      && this.materialResources.every(m => isToonModel(m.shadingModelId));
    this.keyLight.intensity = toonHeavy ? 1.0 : 3.0;
    this.tonemap.mode = toonHeavy ? 1 : 0;
  }

  private updateMatUbo(matRes: MaterialResources) {
    // packForModel decides what material.rg means for the chosen model, so the
    // GBuffer shader never has to know.
    const [pr, pg] = packForModel(matRes.shadingModelId, matRes.metallic, matRes.roughness);
    matRes.materialInstance.setField("baseColorFactor", [...matRes.baseColorFactor]);
    matRes.materialInstance.setField("params", [pr, pg, matRes.hasBaseColorTex, matRes.lightmapStrength]);
    matRes.materialInstance.setField("params2", [matRes.alphaCutoff, matRes.shadingModelId, matRes.outlineWidth, matRes.hasNormalTex]);
    matRes.materialInstance.setField("params3", [matRes.objectId, 0, 0, 0]);
    matRes.materialInstance.upload(this.device);
  }

  private exportPresetJSON(preset: RenderPreset) {
    const json = JSON.stringify(preset, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `preset-${preset.name.replace(/\s+/g, "_")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  private importPresetJSON(): Promise<RenderPreset> {
    return new Promise((resolve, reject) => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".json";
      input.onchange = async () => {
        const file = input.files?.[0];
        if (!file) { reject(new Error("No file")); return; }
        try {
          const text = await file.text();
          const preset = JSON.parse(text) as RenderPreset;
          if (!preset.name || !preset.mapping) throw new Error("Invalid preset format");
          resolve(preset);
        } catch (e) {
          reject(e);
        }
      };
      input.click();
    });
  }

  stats() {
    return {
      drawCalls: this.meshBuffers.length * (this.showOutline ? 2 : 1) + 2,
      triangles: this.totalTriangles,
      custom: {
        "Model": this.modelName || "Loading...",
        "Meshes": this.meshBuffers.length,
        "Materials": this.materialResources.length,
        "Pipeline": "Deferred",
        "Preset": this.activePresetName || "-",
        "Outline": this.showOutline ? "On" : "Off",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "rotationSpeed", 0, 2, 0.05).name("Rotation Speed");
    gui.add(this, "showOutline").name("Show Outline");
    gui.add(this, "globalOutlineWidth", 0, 0.05, 0.001).name("Outline Width").onChange((v: number) => {
      for (const matRes of this.materialResources) {
        matRes.outlineWidth = v;
        this.updateMatUbo(matRes);
      }
    });

    // === Lighting / Tonemap (kernel-level, shared by every shading model) ===
    const lightFolder = gui.addFolder("Lighting");
    lightFolder.add(this.keyLight, "intensity", 0, 8, 0.05).name("Key Intensity");
    lightFolder.add({ ambient: this.lightScene.ambientIntensity }, "ambient", 0, 2, 0.01)
      .name("Ambient")
      .onChange((v: number) => { this.lightScene.ambientIntensity = v; });
    lightFolder.add(this.tonemap, "exposure", 0.1, 4, 0.05).name("Exposure");
    lightFolder.add(this.tonemap, "mode", { "ACES Filmic": 0, "Clamp (toon-safe)": 1 }).name("Tonemap");

    // === Preset System ===
    const presetFolder = gui.addFolder("Presets");
    const presetCtrl = {
      current: this.presets[0]?.name ?? "None",
      apply: () => {
        const p = this.presets.find(p => p.name === presetCtrl.current);
        if (p) this.applyPreset(p);
      },
      save: () => {
        const mapping: Record<string, number> = {};
        for (const matRes of this.materialResources) {
          mapping[matRes.category] = matRes.shadingModelId;
        }
        const name = prompt("Preset name:", "My Preset") ?? "My Preset";
        const preset: RenderPreset = {
          name,
          version: PRESET_VERSION,
          mapping: mapping as Record<MaterialCategory, number>,
        };
        this.presets.push(preset);
        savePresets(this.presets);
        presetCtrl.current = name;
      },
      export: () => {
        const p = this.presets.find(p => p.name === presetCtrl.current);
        if (p) this.exportPresetJSON(p);
      },
      import: async () => {
        try {
          const preset = await this.importPresetJSON();
          this.presets.push(preset);
          savePresets(this.presets);
          presetCtrl.current = preset.name;
          this.applyPreset(preset);
        } catch (e) {
          console.warn("[GLBViewer] Import failed:", e);
        }
      },
    };

    const presetNames = this.presets.map(p => p.name);
    presetFolder.add(presetCtrl, "current", presetNames).name("Preset");
    presetFolder.add(presetCtrl, "apply").name("Apply Preset");
    presetFolder.add(presetCtrl, "save").name("Save Current as Preset");
    presetFolder.add(presetCtrl, "export").name("Export Preset JSON");
    presetFolder.add(presetCtrl, "import").name("Import Preset JSON");

    // === Asset Inspector ===
    const inspector = gui.addFolder("Asset Inspector");
    inspector.add({ info: `${this.meshBuffers.length} meshes` }, "info").name("Meshes").disable();
    inspector.add({ info: `${this.materialResources.length} materials` }, "info").name("Materials").disable();
    inspector.add({ info: this.totalTriangles.toLocaleString() }, "info").name("Triangles").disable();

    // === Mesh Visibility ===
    const meshFolder = gui.addFolder("Mesh Visibility");
    this.meshBuffers.forEach((mb, i) => {
      const matName = this.materialResources[Math.min(mb.materialIndex, this.materialResources.length - 1)]?.name ?? "?";
      const sub = meshFolder.addFolder(`${i}: ${mb.name} [${matName}]`);
      sub.add(mb, "visible").name("Visible");
      sub.add({ recalc: () => this.recalculateNormals(mb) }, "recalc").name("Recalculate Normals");
    });

    // === Per-Material Controls ===
    const modelOptions = shadingModelOptions();
    const matFolder = gui.addFolder("Materials");
    this.materialResources.forEach((matRes, i) => {
      const f = matFolder.addFolder(`${i}: ${matRes.name} [${matRes.category}]`);

      if (matRes.previewImage) {
        const previewCanvas = document.createElement("canvas");
        const img = matRes.previewImage;
        const maxW = 232, maxH = 120;
        const scale = Math.min(maxW / img.width, maxH / img.height, 1);
        previewCanvas.width = Math.max(1, Math.floor(img.width * scale));
        previewCanvas.height = Math.max(1, Math.floor(img.height * scale));
        previewCanvas.style.cssText = "display:block;width:100%;image-rendering:pixelated;border:1px solid #444;margin:4px 0;background:#222;";
        const pctx = previewCanvas.getContext("2d");
        if (pctx) pctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
        f.domElement.appendChild(previewCanvas);
      }

      const ctrl = {
        shadingModelId: matRes.shadingModelId,
        metallic: matRes.metallic,
        roughness: matRes.roughness,
        alphaCutoff: matRes.alphaCutoff,
        outlineWidth: matRes.outlineWidth,
      };
      // The dropdown lists the shading-model registry, not a demo-local enum:
      // register a new look once and every viewer picks it up.
      f.add(ctrl, "shadingModelId", modelOptions).name("Shading Model").onChange((v: number) => {
        matRes.shadingModelId = Number(v);
        this.updateMatUbo(matRes);
      });
      f.add(ctrl, "metallic", 0, 1, 0.01).name("Metallic").onChange((v: number) => {
        matRes.metallic = v;
        this.updateMatUbo(matRes);
      });
      f.add(ctrl, "roughness", 0.01, 1, 0.01).name("Roughness").onChange((v: number) => {
        matRes.roughness = v;
        this.updateMatUbo(matRes);
      });
      f.add(ctrl, "alphaCutoff", 0, 1, 0.01).name("Alpha Cutoff").onChange((v: number) => {
        matRes.alphaCutoff = v;
        this.updateMatUbo(matRes);
      });
      f.add(ctrl, "outlineWidth", 0, 0.05, 0.001).name("Outline Width").onChange((v: number) => {
        matRes.outlineWidth = v;
        this.updateMatUbo(matRes);
      });
      f.add({ diffuse: matRes.hasBaseColorTex ? "Yes" : "No" }, "diffuse").name("Diffuse Tex").disable();
      f.add({ lightmap: matRes.hasLightmap ? "Yes" : "No" }, "lightmap").name("Lightmap Tex").disable();
      f.add({ normal: matRes.hasNormalTex ? "Yes" : "No" }, "normal").name("Normal Tex").disable();
    });
  }

  destroy() {
    for (const mb of this.meshBuffers) {
      mb.vertexBuffer.destroy();
      mb.indexBuffer.destroy();
    }
    for (const mr of this.materialResources) {
      mr.materialInstance.destroy();
      mr.previewImage?.close();
    }
    this.frameUbo.destroy();
    this.defaultTexture.destroy();
    this.hdrTexture?.destroy();
    this.gbuffer?.destroy();
    this.deferredLighting?.destroy();
    this.tonemap?.destroy();
  }
}
