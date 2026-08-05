import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import { PassManager } from "../passes/render-target";
import { GBuffer, GBUFFER_GEOMETRY_WGSL, makeCharGbufferFS } from "../passes/gbuffer";
import { OUTLINE_SCREEN_SHADER } from "../passes/outline-screen";
import { DeferredLightingPass } from "../passes/deferred-lighting";
import { CascadedShadowMap, CSM_CASCADE_COUNT } from "../passes/csm";
import { GTAOPass } from "../passes/ssao";
import { BloomPass } from "../passes/bloom";
import { PostProcessPass } from "../passes/post-process";
import { TAAPass } from "../passes/taa";
import { LightScene, createDirectionalLight, createPointLight, createSpotLight } from "../scene/light";
import { EnvironmentMap } from "../passes/environment";
import { BrdfLut } from "../passes/brdf-lut";
import { createCubeGeometry, createSphereGeometry } from "../utils/geometry";
import { mat4, vec3, vec4, type Mat4 } from "wgpu-matrix";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";
import { MaterialInstance, type MaterialBlueprint } from "../core/material-instance";
import { ShadingModel } from "../core/shading-model";

// Per-object / per-frame matrices + selection tint. These are camera/object
// state, NOT material state, so they stay in an explicit "globals" block.
const GLOBALS_STRUCT = `
struct Globals {
  viewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  prevModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  selectedTint: f32,
};
@group(0) @binding(0) var<uniform> globals: Globals;
`;

// Material parameters (metallic / roughness / time) live in a MaterialInstance.
// The WGSL struct + binding for it are AUTO-GENERATED at runtime from the
// blueprint (see init()), exactly like the original AfterglowRender's
// "auto generate shader codes" feature.
const STANDARD_MATERIAL_BLUEPRINT: MaterialBlueprint = {
  name: "standard",
  group: 1,
  fields: [
    { name: "time", type: "f32", value: 0 },
    { name: "metallic", type: "f32", value: 0.1 },
    { name: "roughness", type: "f32", value: 0.5 },
  ],
};

// Toon material: flat base color + band count. The GBuffer FS stamps
// materialID = ShadingModel.TOON (1) and stashes bandCount in material.g so
// the lighting pass can read it straight from the GBuffer.
const TOON_MATERIAL_BLUEPRINT: MaterialBlueprint = {
  name: "toon",
  group: 1,
  fields: [
    { name: "time", type: "f32", value: 0 },
    { name: "r", type: "f32", value: 0.26 },
    { name: "g", type: "f32", value: 0.56 },
    { name: "b", type: "f32", value: 0.95 },
    { name: "bandCount", type: "f32", value: 3 },
  ],
};

const gbufferVSBody = `
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

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = globals.model * vec4<f32>(pos, 1.0);
  let clip = globals.viewProj * worldPos;
  out.position = clip;
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((globals.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  let prevWorld = globals.prevModel * vec4<f32>(pos, 1.0);
  let prevClip = globals.prevViewProj * prevWorld;
  out.prevClipXY = prevClip.xy;
  out.prevClipW = prevClip.w;
  out.curClipXY = clip.xy;
  out.curClipW = clip.w;
  return out;
}
`;

const gbufferFSBody = `
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

struct GBufferOutput {
  @location(0) albedo: vec4<f32>,
  @location(1) normal: vec4<f32>,
  @location(2) material: vec4<f32>,
  @location(3) motion: vec2<f32>,
  @location(4) depthCopy: vec4<f32>,
};

@fragment
fn fs_main(in: VSOut) -> GBufferOutput {
  var out: GBufferOutput;

  var baseColor = vec3<f32>(
    0.5 + 0.5 * sin(mat_standard.time * 0.3 + in.uv.x * 6.28),
    0.5 + 0.5 * cos(mat_standard.time * 0.5 + in.uv.y * 6.28),
    0.7
  );

  out.albedo = vec4<f32>(baseColor, 1.0);
  // normal.w carries the object ID (1 = standard cube) for the screen-space outline pass.
  out.normal = vec4<f32>(normalize(in.worldNormal), 1.0);
  // material = (metallic, roughness, emissive, shadingModelId=STANDARD)
  out.material = vec4<f32>(mat_standard.metallic, mat_standard.roughness, 0.0, ${ShadingModel.STANDARD}.0);

  // perspective-correct NDC: clip.xy and clip.w are interpolated separately
  // note: NDC.y grows upward but texture uv.y grows downward, so flip y
  let prevNDC = in.prevClipXY / in.prevClipW;
  let curNDC = in.curClipXY / in.curClipW;
  let motion = vec2<f32>(
    (curNDC.x - prevNDC.x) * 0.5,
    (prevNDC.y - curNDC.y) * 0.5,
  );
  out.motion = motion;

  out.depthCopy = vec4<f32>(in.position.z, 0.0, 0.0, 0.0);

  return out;
}
`;

// Toon variant of the GBuffer pass. Writes a flat base color and stamps
// materialID = TOON (1); bandCount is parked in material.g so the deferred
// lighting pass can read it straight from the GBuffer without extra plumbing.
const gbufferToonFSBody = `
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

struct GBufferOutput {
  @location(0) albedo: vec4<f32>,
  @location(1) normal: vec4<f32>,
  @location(2) material: vec4<f32>,
  @location(3) motion: vec2<f32>,
  @location(4) depthCopy: vec4<f32>,
};

@fragment
fn fs_main(in: VSOut) -> GBufferOutput {
  var out: GBufferOutput;

  var baseColor = vec3<f32>(mat_toon.r, mat_toon.g, mat_toon.b);

  out.albedo = vec4<f32>(baseColor, 1.0);
  // normal.w carries the object ID (2 = toon sphere) for the screen-space outline pass.
  out.normal = vec4<f32>(normalize(in.worldNormal), 2.0);
  // material = (metallic=0, bandCount, emissive=0, shadingModelId=TOON)
  out.material = vec4<f32>(0.0, mat_toon.bandCount, 0.0, ${ShadingModel.TOON}.0);

  let prevNDC = in.prevClipXY / in.prevClipW;
  let curNDC = in.curClipXY / in.curClipW;
  let motion = vec2<f32>(
    (curNDC.x - prevNDC.x) * 0.5,
    (prevNDC.y - curNDC.y) * 0.5,
  );
  out.motion = motion;

  out.depthCopy = vec4<f32>(in.position.z, 0.0, 0.0, 0.0);

  return out;
}
`;

// ---- Step 2 character materials (ShadingModelID 2/3/4) -------------------
// Each packs its own parameters into the GBuffer material channel; the
// lighting pass reads them back to pick the right BxDF.
const SKIN_MATERIAL_BLUEPRINT: MaterialBlueprint = {
  name: "skin",
  group: 1,
  fields: [
    { name: "time", type: "f32", value: 0 },
    { name: "sssStrength", type: "f32", value: 0.8 },
    { name: "roughness", type: "f32", value: 0.4 },
  ],
};

const HAIR_MATERIAL_BLUEPRINT: MaterialBlueprint = {
  name: "hair",
  group: 1,
  fields: [
    { name: "time", type: "f32", value: 0 },
    { name: "roughness", type: "f32", value: 0.5 },
    { name: "aniso", type: "f32", value: 0.5 },
  ],
};

const EYE_MATERIAL_BLUEPRINT: MaterialBlueprint = {
  name: "eye",
  group: 1,
  fields: [
    { name: "time", type: "f32", value: 0 },
    { name: "cornea", type: "f32", value: 0.9 },
    { name: "irisDark", type: "f32", value: 0.25 },
  ],
};


/** One character part in the Step-2 demo scene. */
interface CharPart {
  material: MaterialInstance;
  fsCode: string;
  pipeline: GPURenderPipeline | null;
  materialBG: GPUBindGroup | null;
  gbufferBG: GPUBindGroup | null;
  globalsUBO: GPUBuffer | null;
  uboData: Float32Array;
  model: Float32Array;
  prevModel: Float32Array;
  position: [number, number, number];
  scale: [number, number, number];
  rotSpeed: number;
  objectId: number;
  selectedId: number;
  geometry: "sphere" | "cube";
}

const shadowVS = `
struct Uniforms {
  lightVP: mat4x4<f32>,
  model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
) -> @builtin(position) vec4<f32> {
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  return u.lightVP * worldPos;
}
`;

// Screen-space outline machine (Blender-style). One parameterized shader draws
// a rim on background pixels around any fragment whose id == cfg.targetId.
// Two independent layers reuse this same machine:
//   • cel outline  → reads the GBuffer material channel, targetId = ShadingModelID.TOON
//   • selection     → reads the GBuffer normal.w (object id), targetId = selectedId
// Winding- and depth-independent, so it can never "fill" the object the way an
// inverted hull can.

const gizmoVS = `
struct GizmoOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec4<f32>,
  @location(1) color: vec4<f32>,
) -> GizmoOut {
  var out: GizmoOut;
  out.position = pos;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: GizmoOut) -> @location(0) vec4<f32> {
  return in.color;
}
`;

export class DeferredDemo implements Demo {
  label = "Deferred PBR";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private engine!: EngineContext;
  private camera!: Camera;
  private passManager!: PassManager;

  private gbuffer!: GBuffer;
  private deferredLighting!: DeferredLightingPass;
  private csm!: CascadedShadowMap;
  private gtao!: GTAOPass;
  private bloomPass!: BloomPass;
  private postProcessPass!: PostProcessPass;
  private taaPass!: TAAPass;
  private lightScene!: LightScene;
  private envMap!: EnvironmentMap;
  private brdfLut!: BrdfLut;

  private gbufferStdPipeline!: GPURenderPipeline;
  private gbufferToonPipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;

  private cubeVB!: GPUBuffer;
  private cubeIB!: GPUBuffer;
  private cubeIndexCount = 0;
  private sphereVB!: GPUBuffer;
  private sphereIB!: GPUBuffer;
  private sphereIndexCount = 0;

  private globalsCubeUBO!: GPUBuffer;
  private globalsSphereUBO!: GPUBuffer;
  private shadowUBO!: GPUBuffer;
  private stdMaterialBG: GPUBindGroup | null = null;
  private toonMaterialBG: GPUBindGroup | null = null;
  private standardMaterial!: MaterialInstance;
  private toonMaterial!: MaterialInstance;
  private dummyDepthTexture!: GPUTexture;

  private prevViewProj: Float32Array = new Float32Array(16);
  private prevModel: Float32Array = new Float32Array(16);
  private prevBallModel: Float32Array = new Float32Array(16);

  metallic = 0.1;
  roughness = 0.5;
  bloomIntensity = 0.3;
  useCSM = true;
  useSSAO = true;
  useTAA = true;

  private vsCode = "";
  private fsCode = "";
  private toonFsCode = "";

  private cubePos: [number, number, number] = [0, 0, 0];
  private ballPos: [number, number, number] = [2.6, 0, 0];
  private ballModel = new Float32Array(16);
  private selectedId = 0;
  private editMode = false;
  private dragAxis = 0;
  private dragging = false;
  private dragLastX = 0;
  private dragLastY = 0;

  private gizmoVB!: GPUBuffer;
  private gizmoPipeline!: GPURenderPipeline;
  private gizmoData = new Float32Array(6 * 8);
  private gizmoNDC = new Float32Array(6 * 2);
  private outlineScreenPipeline!: GPURenderPipeline;
  private outlineSampler!: GPUSampler;

  // Layer A — cel (toon) outline: driven by ShadingModelID in the GBuffer
  // material channel, so it belongs to the artwork and is included in exports.
  private celOutlineUBO!: GPUBuffer;
  private celOutlineUboData = new Float32Array(8);
  private celOutlineBG: GPUBindGroup | null = null;

  // Layer B — selection highlight: driven by the editor's selected object id
  // (GBuffer normal.w). Editor-only overlay, EXCLUDED from exports.
  private selOutlineUBO!: GPUBuffer;
  private selOutlineUboData = new Float32Array(8);
  private selOutlineBG: GPUBindGroup | null = null;

  /** Editor-only selection outline (Blender-style orange rim). Off for exports. */
  outlineEnabled = true;
  /** When false, the selection overlay is suppressed (e.g. when rendering to a file). */
  editorOverlay = true;
  /** Selection outline width in pixels (screen-space). */
  selectionWidth = 2;
  /** Always draw a cel-shading outline around TOON-shaded geometry. */
  toonOutline = true;
  /** Cel outline width in pixels (screen-space). */
  celOutlineWidth = 1;
  /** Number of diffuse bands for the toon sphere's cel shading. */
  toonBands = 3;
  /** Toon base color (also pushed into the toon MaterialInstance). */
  toonR = 0.26;
  toonG = 0.56;
  toonB = 0.95;

  // --- Step 2 character shading GUI params (ShadingModelID 2/3/4) -------
  /** Skin subsurface-scattering strength (0-1). */
  skinSss = 0.8;
  /** Skin roughness. */
  skinRoughness = 0.4;
  /** Hair roughness. */
  hairRoughness = 0.5;
  /** Hair anisotropy / strand shift. */
  hairAniso = 0.5;
  /** Eye cornea highlight strength. */
  eyeCornea = 0.9;
  /** Eye iris darkness (lower = darker iris). */
  eyeIrisDark = 0.25;

  // Step 2 character parts (skin head / hair tuft / eye). Each is a full
  // GBuffer pipeline + MaterialInstance + per-object globals UBO.
  private skinMaterial!: MaterialInstance;
  private hairMaterial!: MaterialInstance;
  private eyeMaterial!: MaterialInstance;
  private gbufferSkinPipeline!: GPURenderPipeline;
  private gbufferHairPipeline!: GPURenderPipeline;
  private gbufferEyePipeline!: GPURenderPipeline;
  private skinFsCode = "";
  private hairFsCode = "";
  private eyeFsCode = "";
  private charParts: CharPart[] = [];

  init(ctx: GPUContext, camera: Camera, engine?: EngineContext) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.engine = engine!;

    this.passManager = new PassManager(ctx.device, ctx.format);
    this.gbuffer = new GBuffer(ctx.device);
    this.lightScene = new LightScene();
    this.lightScene.ambientColor = [0.15, 0.15, 0.18];
    this.lightScene.ambientIntensity = 1.0;

    const dirLight = createDirectionalLight([-0.5, -1.0, -0.3], [1, 1, 1], 3.0);
    this.lightScene.addLight(dirLight);
    this.lightScene.addLight(createPointLight([3, 2, 3], [1, 0.8, 0.5], 8.0, 15.0));
    this.lightScene.addLight(createPointLight([-3, 2, -2], [0.3, 0.5, 1.0], 6.0, 12.0));
    this.lightScene.addLight(createSpotLight([0, 5, 0], [0, -1, 0], [1, 0.9, 0.7], 12.0, 20.0));

    this.deferredLighting = new DeferredLightingPass(ctx.device, this.lightScene, "rgba16float");
    this.csm = new CascadedShadowMap(ctx.device, 2048);
    this.gtao = new GTAOPass(ctx.device);
    this.envMap = new EnvironmentMap();
    this.envMap.bake(ctx.device);
    this.brdfLut = new BrdfLut();
    this.brdfLut.bake(ctx.device);
    this.bloomPass = new BloomPass(ctx.device, "rgba16float");
    this.bloomPass.threshold = 0.3;
    this.bloomPass.knee = 0.7;
    this.bloomPass.radius = 2.5;
    this.postProcessPass = new PostProcessPass(ctx.device, ctx.format);
    this.taaPass = new TAAPass(ctx.device, "rgba16float");

    this.standardMaterial = new MaterialInstance(STANDARD_MATERIAL_BLUEPRINT);
    this.toonMaterial = new MaterialInstance(TOON_MATERIAL_BLUEPRINT);
    this.vsCode = GLOBALS_STRUCT + gbufferVSBody;
    this.fsCode = this.standardMaterial.generateWGSL() + "\n" + GLOBALS_STRUCT + gbufferFSBody;
    this.toonFsCode = this.toonMaterial.generateWGSL() + "\n" + GLOBALS_STRUCT + gbufferToonFSBody;

    // Step 2 character materials + their GBuffer FS (stamp ShadingModelID).
    this.skinMaterial = new MaterialInstance(SKIN_MATERIAL_BLUEPRINT);
    this.hairMaterial = new MaterialInstance(HAIR_MATERIAL_BLUEPRINT);
    this.eyeMaterial = new MaterialInstance(EYE_MATERIAL_BLUEPRINT);
    this.skinFsCode = this.skinMaterial.generateWGSL() + "\n" + GLOBALS_STRUCT
      + makeCharGbufferFS({
        albedoExpr: "vec3<f32>(0.95, 0.72, 0.62)",
        packExpr: "vec3<f32>(mat_skin.sssStrength, mat_skin.roughness, 0.0)",
        idExpr: `${ShadingModel.SKIN}.0`,
        objectIdExpr: "3.0",
      });
    this.hairFsCode = this.hairMaterial.generateWGSL() + "\n" + GLOBALS_STRUCT
      + makeCharGbufferFS({
        albedoExpr: "vec3<f32>(0.28, 0.18, 0.13)",
        packExpr: "vec3<f32>(mat_hair.roughness, mat_hair.aniso, 0.0)",
        idExpr: `${ShadingModel.HAIR}.0`,
        objectIdExpr: "4.0",
      });
    this.eyeFsCode = this.eyeMaterial.generateWGSL() + "\n" + GLOBALS_STRUCT
      + makeCharGbufferFS({
        albedoExpr: "vec3<f32>(0.85, 0.9, 0.95)",
        packExpr: "vec3<f32>(mat_eye.cornea, mat_eye.irisDark, 0.0)",
        idExpr: `${ShadingModel.EYE}.0`,
        objectIdExpr: "5.0",
      });

    const mkCharUbo = (label: string) => this.device.createBuffer({
      label, size: 352, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Step 2 character parts: a small "head" = skin sphere + hair tuft
    // (stretched cube) + eye sphere. Placed to the left of the standard cube
    // / toon sphere so all five objects share ONE scene, ONE lighting pass.
    // NOTE: must be populated BEFORE buildPipelines() so the per-part pipeline
    // references are valid.
    this.charParts = [
      {
        material: this.skinMaterial, fsCode: this.skinFsCode,
        pipeline: null, materialBG: null, gbufferBG: null, globalsUBO: mkCharUbo("deferred-globals-skin-ubo"),
        uboData: new Float32Array(88), model: new Float32Array(16), prevModel: new Float32Array(16),
        position: [-2.6, 0, 0.5], scale: [1.1, 1.1, 1.1], rotSpeed: 0.3, objectId: 3, selectedId: 3, geometry: "sphere",
      },
      {
        material: this.hairMaterial, fsCode: this.hairFsCode,
        pipeline: null, materialBG: null, gbufferBG: null, globalsUBO: mkCharUbo("deferred-globals-hair-ubo"),
        uboData: new Float32Array(88), model: new Float32Array(16), prevModel: new Float32Array(16),
        position: [-2.6, 1.15, 0.5], scale: [0.22, 0.9, 0.22], rotSpeed: 0.0, objectId: 4, selectedId: 4, geometry: "cube",
      },
      {
        material: this.eyeMaterial, fsCode: this.eyeFsCode,
        pipeline: null, materialBG: null, gbufferBG: null, globalsUBO: mkCharUbo("deferred-globals-eye-ubo"),
        uboData: new Float32Array(88), model: new Float32Array(16), prevModel: new Float32Array(16),
        position: [-2.15, 0.2, 1.25], scale: [0.35, 0.35, 0.35], rotSpeed: 0.3, objectId: 5, selectedId: 5, geometry: "sphere",
      },
    ];

    this.createGeometry();
    this.buildPipelines();

    this.globalsCubeUBO = this.device.createBuffer({
      label: "deferred-globals-cube-ubo",
      size: 352,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.globalsSphereUBO = this.device.createBuffer({
      label: "deferred-globals-sphere-ubo",
      size: 352,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.shadowUBO = this.device.createBuffer({
      label: "deferred-shadow-ubo",
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    mat4.identity(this.prevViewProj);
    mat4.identity(this.prevModel);
    mat4.identity(this.prevBallModel);

    this.dummyDepthTexture = this.device.createTexture({
      label: "deferred-dummy-depth",
      size: [1, 1],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.gizmoVB = this.device.createBuffer({
      label: "gizmo-vb",
      size: 6 * 8 * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.celOutlineUBO = this.device.createBuffer({
      label: "deferred-cel-outline-ubo",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.selOutlineUBO = this.device.createBuffer({
      label: "deferred-sel-outline-ubo",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.outlineSampler = this.device.createSampler({
      label: "deferred-outline-sampler",
      magFilter: "nearest",
      minFilter: "nearest",
    });

    const cv = ctx.canvas;
    cv.addEventListener("pointerdown", this.onPointerDown, true);
    cv.addEventListener("pointermove", this.onPointerMove, true);
    cv.addEventListener("pointerup", this.onPointerUp, true);
    cv.addEventListener("pointercancel", this.onPointerUp, true);
  }

  private createGeometry(): void {
    const cube = createCubeGeometry();
    this.cubeIndexCount = cube.indices.length;
    this.cubeVB = this.device.createBuffer({
      label: "deferred-cube-vb",
      size: cube.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.cubeVB.getMappedRange()).set(cube.vertices);
    this.cubeVB.unmap();
    this.cubeIB = this.device.createBuffer({
      label: "deferred-cube-ib",
      size: cube.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.cubeIB.getMappedRange()).set(cube.indices);
    this.cubeIB.unmap();

    const sphere = createSphereGeometry(1.0, 32, 32);
    this.sphereIndexCount = sphere.indices.length;
    this.sphereVB = this.device.createBuffer({
      label: "deferred-sphere-vb",
      size: sphere.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.sphereVB.getMappedRange()).set(sphere.vertices);
    this.sphereVB.unmap();
    this.sphereIB = this.device.createBuffer({
      label: "deferred-sphere-ib",
      size: sphere.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.sphereIB.getMappedRange()).set(sphere.indices);
    this.sphereIB.unmap();
  }

  private buildPipelines(): void {
    // Reset bind groups so they are recreated against the new pipeline layout
    // (also happens on shader hot-reload).
    this.gbufferCubeBG = null;
    this.gbufferSphereBG = null;
    this.stdMaterialBG = null;
    this.toonMaterialBG = null;
    for (const p of this.charParts) { p.gbufferBG = null; p.materialBG = null; }

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const vsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-vs", this.vsCode);
    const stdFsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-fs", this.fsCode);
    const toonFsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-toon-fs", this.toonFsCode);
    const skinFsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-skin-fs", this.skinFsCode);
    const hairFsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-hair-fs", this.hairFsCode);
    const eyeFsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-eye-fs", this.eyeFsCode);

    const makeGBufferPipeline = (label: string, fsModule: GPUShaderModule): GPURenderPipeline => {
      return this.device.createRenderPipeline({
        label,
        layout: "auto",
        vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
        fragment: {
          module: fsModule,
          entryPoint: "fs_main",
          targets: [
            { format: GBuffer.ALBEDO_FORMAT },
            { format: GBuffer.NORMAL_FORMAT },
            { format: GBuffer.MATERIAL_FORMAT },
            { format: GBuffer.MOTION_FORMAT },
            { format: GBuffer.DEPTH_COPY_FORMAT },
          ],
        },
        primitive: { topology: "triangle-list", cullMode: "back" },
        depthStencil: {
          format: GBuffer.DEPTH_FORMAT,
          depthWriteEnabled: true,
          depthCompare: "less",
        },
      });
    };

    // Two GBuffer pipelines sharing the same vertex stage but writing
    // different ShadingModelIDs into the material channel.
    this.gbufferStdPipeline = makeGBufferPipeline("deferred-gbuffer-std", stdFsModule);
    this.gbufferToonPipeline = makeGBufferPipeline("deferred-gbuffer-toon", toonFsModule);
    this.gbufferSkinPipeline = makeGBufferPipeline("deferred-gbuffer-skin", skinFsModule);
    this.gbufferHairPipeline = makeGBufferPipeline("deferred-gbuffer-hair", hairFsModule);
    this.gbufferEyePipeline = makeGBufferPipeline("deferred-gbuffer-eye", eyeFsModule);
    this.charParts[0].pipeline = this.gbufferSkinPipeline;
    this.charParts[1].pipeline = this.gbufferHairPipeline;
    this.charParts[2].pipeline = this.gbufferEyePipeline;

    // The material's uniform block (group 1) is per-pipeline.
    this.stdMaterialBG = this.standardMaterial.createBindGroup(this.device, this.gbufferStdPipeline);
    this.toonMaterialBG = this.toonMaterial.createBindGroup(this.device, this.gbufferToonPipeline);
    for (const p of this.charParts) {
      p.materialBG = p.material.createBindGroup(this.device, p.pipeline!);
    }

    const shadowVsModule = this.device.createShaderModule({ code: shadowVS });
    this.shadowPipeline = this.device.createRenderPipeline({
      label: "deferred-shadow",
      layout: "auto",
      vertex: { module: shadowVsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const gizmoModule = this.device.createShaderModule({ code: gizmoVS });
    this.gizmoPipeline = this.device.createRenderPipeline({
      label: "gizmo",
      layout: "auto",
      vertex: {
        module: gizmoModule,
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 8 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x4" },
            { shaderLocation: 1, offset: 16, format: "float32x4" },
          ],
        }],
      },
      fragment: { module: gizmoModule, entryPoint: "fs_main", targets: [{ format: this.ctx.format }] },
      primitive: { topology: "line-list" },
    });

    const outlineScreenModule = this.device.createShaderModule({ code: OUTLINE_SCREEN_SHADER });
    this.outlineScreenPipeline = this.device.createRenderPipeline({
      label: "deferred-outline-screen",
      layout: "auto",
      vertex: { module: outlineScreenModule, entryPoint: "vs_main" },
      fragment: {
        module: outlineScreenModule,
        entryPoint: "fs_main",
        targets: [{
          format: this.ctx.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  getShaderStages(): ShaderStageDesc[] {
    return [
      { label: "Deferred / GBuffer VS", type: "vertex", code: this.vsCode },
      { label: "Deferred / GBuffer FS", type: "fragment", code: this.fsCode },
    ];
  }

  onShaderReload(stageLabel: string, code: string): boolean {
    if (stageLabel === "Deferred / GBuffer VS") this.vsCode = code;
    else if (stageLabel === "Deferred / GBuffer FS") this.fsCode = code;
    try { this.buildPipelines(); return true; } catch { return false; }
  }

  private uboData = new Float32Array(88);
  private ballUboData = new Float32Array(88);
  private shadowUboData = new Float32Array(32);
  private jitteredVP = new Float32Array(16);
  private cubeModel = new Float32Array(16);
  private shadowCubeBG: GPUBindGroup | null = null;
  private shadowSphereBG: GPUBindGroup | null = null;
  private gbufferCubeBG: GPUBindGroup | null = null;
  private gbufferSphereBG: GPUBindGroup | null = null;
  private frameTime = 0;
  private taaFrameIndex = 0;
  private taaJitter = true;
  private pauseAnimation = false;
  private frameViewProj = new Float32Array(16);
  private frameInvViewProj = new Float32Array(16);

  private halton(index: number, base: number): number {
    let result = 0;
    let f = 1 / base;
    let i = index;
    while (i > 0) {
      result += f * (i % base);
      i = Math.floor(i / base);
      f /= base;
    }
    return result;
  }

  private applyJitter(viewProj: Float32Array, index: number, w: number, h: number): Float32Array {
    // 0.5px jitter in NDC space
    const jx = (this.halton(index, 2) - 0.5) * 2.0 / w;
    const jy = (this.halton(index, 3) - 0.5) * 2.0 / h;
    this.jitteredVP.set(viewProj as unknown as ArrayLike<number>);
    // this assumes column-major wgpu-matrix: proj[2][0]=8, proj[2][1]=9
    this.jitteredVP[8] += jx;
    this.jitteredVP[9] += jy;
    return this.jitteredVP;
  }

  update(time: number) {
    if (this.pauseAnimation) time = 0;
    this.frameTime = time;
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const baseVP = this.camera.getViewProjectionMatrix(aspect) as unknown as Float32Array;
    const viewProj = this.useTAA && this.taaJitter ? this.applyJitter(baseVP, this.taaFrameIndex, w, h) : baseVP;

    const ubo = this.uboData;
    ubo.set(viewProj as unknown as ArrayLike<number>, 0);
    ubo.set(this.prevViewProj as unknown as ArrayLike<number>, 16);
    const model = mat4.mul(
      mat4.translation(this.cubePos),
      mat4.mul(mat4.rotationY(time * 0.5), mat4.scaling([1.5, 1.5, 1.5])),
    );
    const invTransModel = mat4.transpose(mat4.inverse(model));
    this.cubeModel.set(model as unknown as ArrayLike<number>);
    ubo.set(model as unknown as ArrayLike<number>, 32);
    ubo.set(invTransModel as unknown as ArrayLike<number>, 48);
    ubo.set(this.prevModel as unknown as ArrayLike<number>, 64);
    ubo[80] = this.camera.position[0];
    ubo[81] = this.camera.position[1];
    ubo[82] = this.camera.position[2];
    ubo[83] = 1.0;
    ubo[84] = this.selectedId === 1 ? 1 : 0;

    this.device.queue.writeBuffer(this.globalsCubeUBO, 0, ubo as unknown as GPUAllowSharedBufferSource);

    // Material parameters are owned by the MaterialInstance, not the globals UBO.
    this.standardMaterial.setField("time", time);
    this.standardMaterial.setField("metallic", this.metallic);
    this.standardMaterial.setField("roughness", this.roughness);
    this.standardMaterial.upload(this.device);

    // Toon material: tint animates slowly, band count + base color from GUI.
    this.toonMaterial.setField("time", time);
    this.toonMaterial.setField("bandCount", this.toonBands);
    this.toonMaterial.setField("r", this.toonR);
    this.toonMaterial.setField("g", this.toonG);
    this.toonMaterial.setField("b", this.toonB);
    this.toonMaterial.upload(this.device);
    this.prevViewProj.set(viewProj as unknown as ArrayLike<number>);
    this.prevModel.set(model as unknown as ArrayLike<number>);

    // Ball model: offset to the side, slowly rotating
    const ballModel = mat4.mul(
      mat4.translation(this.ballPos),
      mat4.mul(mat4.rotationY(time * 0.4), mat4.scaling([1.0, 1.0, 1.0])),
    );
    const invTransBall = mat4.transpose(mat4.inverse(ballModel));
    this.ballModel.set(ballModel as unknown as ArrayLike<number>);
    const ballUbo = this.ballUboData;
    ballUbo.set(viewProj as unknown as ArrayLike<number>, 0);
    ballUbo.set(this.prevViewProj as unknown as ArrayLike<number>, 16);
    ballUbo.set(ballModel as unknown as ArrayLike<number>, 32);
    ballUbo.set(invTransBall as unknown as ArrayLike<number>, 48);
    ballUbo.set(this.prevBallModel as unknown as ArrayLike<number>, 64);
    ballUbo[80] = this.camera.position[0];
    ballUbo[81] = this.camera.position[1];
    ballUbo[82] = this.camera.position[2];
    ballUbo[83] = 1.0;
    ballUbo[84] = this.selectedId === 2 ? 1 : 0;

    this.device.queue.writeBuffer(this.globalsSphereUBO, 0, ballUbo as unknown as GPUAllowSharedBufferSource);
    this.prevBallModel.set(ballModel as unknown as ArrayLike<number>);

    // Step 2 character parts: per-object globals + material params.
    for (const p of this.charParts) {
      const m = mat4.mul(
        mat4.translation(p.position),
        mat4.mul(mat4.rotationY(time * p.rotSpeed), mat4.scaling(p.scale)),
      );
      const invT = mat4.transpose(mat4.inverse(m));
      p.uboData.set(viewProj as unknown as ArrayLike<number>, 0);
      p.uboData.set(this.prevViewProj as unknown as ArrayLike<number>, 16);
      p.uboData.set(m as unknown as ArrayLike<number>, 32);
      p.uboData.set(invT as unknown as ArrayLike<number>, 48);
      p.uboData.set(p.prevModel as unknown as ArrayLike<number>, 64);
      p.uboData[80] = this.camera.position[0];
      p.uboData[81] = this.camera.position[1];
      p.uboData[82] = this.camera.position[2];
      p.uboData[83] = 1.0;
      p.uboData[84] = this.selectedId === p.selectedId ? 1 : 0;
      this.device.queue.writeBuffer(p.globalsUBO!, 0, p.uboData as unknown as GPUAllowSharedBufferSource);
      p.prevModel.set(m as unknown as ArrayLike<number>);

      p.material.setField("time", time);
      if (p.material.name === "skin") {
        p.material.setField("sssStrength", this.skinSss);
        p.material.setField("roughness", this.skinRoughness);
      } else if (p.material.name === "hair") {
        p.material.setField("roughness", this.hairRoughness);
        p.material.setField("aniso", this.hairAniso);
      } else if (p.material.name === "eye") {
        p.material.setField("cornea", this.eyeCornea);
        p.material.setField("irisDark", this.eyeIrisDark);
      }
      p.material.upload(this.device);
    }

    this.bloomPass.bloomIntensity = this.bloomIntensity;

    const invViewProj = mat4.inverse(viewProj);
    this.frameViewProj.set(viewProj as unknown as ArrayLike<number>);
    this.frameInvViewProj.set(invViewProj as unknown as ArrayLike<number>);
    this.postProcessPass.cameraPos = [this.camera.position[0], this.camera.position[1], this.camera.position[2]];
    this.postProcessPass.invVP.set(invViewProj as unknown as ArrayLike<number>);

    if (this.useCSM) {
      this.csm.updateCascadeVPs(
        viewProj as Mat4,
        invViewProj as Mat4,
        0.1, 50.0,
      );
    }

    this.taaFrameIndex++;
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, screenView: GPUTextureView) => {
        const w = this.ctx.canvas.width;
        const h = this.ctx.canvas.height;
        this.gbuffer.resize(w, h);
        // Outline bind groups reference GBuffer textures, which are recreated on
        // resize — drop the cached groups so they rebuild against the new views.
        this.celOutlineBG = null;
        this.selOutlineBG = null;
        this.passManager.resize(w, h);

        // Step 1: CSM shadow passes
        if (this.useCSM) {
          for (let i = 0; i < CSM_CASCADE_COUNT; i++) {
            const shadowPass = this.csm.beginCascadePass(encoder, i);
            shadowPass.setPipeline(this.shadowPipeline);

            const shadowUbo = this.shadowUboData;
            shadowUbo.set(this.csm.cascadeVPs[i] as unknown as ArrayLike<number>, 0);
            shadowUbo.set(this.cubeModel as unknown as ArrayLike<number>, 16);
            this.device.queue.writeBuffer(this.shadowUBO, 0, shadowUbo as unknown as GPUAllowSharedBufferSource);

            if (!this.shadowCubeBG) {
              this.shadowCubeBG = this.device.createBindGroup({
                layout: this.shadowPipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
              });
            }
            shadowPass.setBindGroup(0, this.shadowCubeBG);
            shadowPass.setVertexBuffer(0, this.cubeVB);
            shadowPass.setIndexBuffer(this.cubeIB, "uint16");
            shadowPass.drawIndexed(this.cubeIndexCount);

            shadowUbo.set(this.csm.cascadeVPs[i] as unknown as ArrayLike<number>, 0);
            shadowUbo.set(this.ballUboData.subarray(32, 48) as unknown as ArrayLike<number>, 16);
            this.device.queue.writeBuffer(this.shadowUBO, 0, shadowUbo as unknown as GPUAllowSharedBufferSource);

            if (!this.shadowSphereBG) {
              this.shadowSphereBG = this.device.createBindGroup({
                layout: this.shadowPipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
              });
            }
            shadowPass.setBindGroup(0, this.shadowSphereBG);
            shadowPass.setVertexBuffer(0, this.sphereVB);
            shadowPass.setIndexBuffer(this.sphereIB, "uint16");
            shadowPass.drawIndexed(this.sphereIndexCount);

            // Step 2 character parts also cast shadows.
            for (const p of this.charParts) {
              shadowUbo.set(this.csm.cascadeVPs[i] as unknown as ArrayLike<number>, 0);
              shadowUbo.set(p.model as unknown as ArrayLike<number>, 16);
              this.device.queue.writeBuffer(this.shadowUBO, 0, shadowUbo as unknown as GPUAllowSharedBufferSource);
              shadowPass.setBindGroup(0, this.shadowCubeBG!);
              const vb = p.geometry === "sphere" ? this.sphereVB : this.cubeVB;
              const ib = p.geometry === "sphere" ? this.sphereIB : this.cubeIB;
              const idx = p.geometry === "sphere" ? this.sphereIndexCount : this.cubeIndexCount;
              shadowPass.setVertexBuffer(0, vb);
              shadowPass.setIndexBuffer(ib, "uint16");
              shadowPass.drawIndexed(idx);
            }
            shadowPass.end();
          }
        }

        // Step 2: GBuffer pass. Cube = STANDARD PBR, Sphere = TOON.
        // Both write the same GBuffer layout; the lighting pass reads the
        // materialID channel to pick the right BxDF — one scene, one light.
        {
          const gbufferPass = this.gbuffer.beginGBufferPass(encoder);

          // --- Cube: Standard PBR ---
          gbufferPass.setPipeline(this.gbufferStdPipeline);
          if (!this.gbufferCubeBG) {
            this.gbufferCubeBG = this.device.createBindGroup({
              layout: this.gbufferStdPipeline.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: { buffer: this.globalsCubeUBO } }],
            });
          }
          gbufferPass.setBindGroup(0, this.gbufferCubeBG);
          gbufferPass.setBindGroup(1, this.stdMaterialBG!);
          gbufferPass.setVertexBuffer(0, this.cubeVB);
          gbufferPass.setIndexBuffer(this.cubeIB, "uint16");
          gbufferPass.drawIndexed(this.cubeIndexCount);

          // --- Sphere: Toon / cel ---
          gbufferPass.setPipeline(this.gbufferToonPipeline);
          if (!this.gbufferSphereBG) {
            this.gbufferSphereBG = this.device.createBindGroup({
              layout: this.gbufferToonPipeline.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: { buffer: this.globalsSphereUBO } }],
            });
          }
          gbufferPass.setBindGroup(0, this.gbufferSphereBG);
          gbufferPass.setBindGroup(1, this.toonMaterialBG!);
          gbufferPass.setVertexBuffer(0, this.sphereVB);
          gbufferPass.setIndexBuffer(this.sphereIB, "uint16");
          gbufferPass.drawIndexed(this.sphereIndexCount);

          // --- Step 2 character parts: SKIN / HAIR / EYE ---
          // Each writes its own ShadingModelID into the GBuffer material
          // channel; the lighting pass picks the matching BxDF. Same scene.
          for (const p of this.charParts) {
            gbufferPass.setPipeline(p.pipeline!);
            if (!p.gbufferBG) {
              p.gbufferBG = this.device.createBindGroup({
                layout: p.pipeline!.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: p.globalsUBO! } }],
              });
            }
            gbufferPass.setBindGroup(0, p.gbufferBG);
            gbufferPass.setBindGroup(1, p.materialBG!);
            const vb = p.geometry === "sphere" ? this.sphereVB : this.cubeVB;
            const ib = p.geometry === "sphere" ? this.sphereIB : this.cubeIB;
            const idx = p.geometry === "sphere" ? this.sphereIndexCount : this.cubeIndexCount;
            gbufferPass.setVertexBuffer(0, vb);
            gbufferPass.setIndexBuffer(ib, "uint16");
            gbufferPass.drawIndexed(idx);
          }

          gbufferPass.end();
        }

        // Step 3: SSAO pass
        if (this.useSSAO) {
          this.gtao.execute(encoder, this.gbuffer, this.frameViewProj as unknown as Float32Array);
        }

        // Step 4: Lighting pass
        const lightingRT = this.passManager.getOrCreateTarget("deferred-lighting", "rgba16float");
        {
          this.deferredLighting.update(
            this.frameViewProj as Mat4,
            [this.camera.position[0], this.camera.position[1], this.camera.position[2]],
            this.frameInvViewProj as Mat4,
            w, h, 0.1, 50.0,
          );
          this.deferredLighting.execute(
            encoder,
            this.gbuffer,
            lightingRT.view,
            this.useCSM ? this.csm.views[0] : undefined,
            this.useCSM ? this.csm.sampler : undefined,
            this.useCSM ? this.csm.ubo : undefined,
            this.useSSAO ? this.gtao.view : undefined,
            {
              irradiance: this.envMap.irradianceView,
              prefilter: this.envMap.prefilterView,
              brdfLut: this.brdfLut.view,
            },
          );
        }

        // Step 5: TAA (temporal accumulation; skip when disabled or when history is stale)
        let taaResultTex = lightingRT.texture;
        if (this.useTAA) {
          const taaResult = this.taaPass.execute(encoder, lightingRT.texture, this.gbuffer.motionView, w, h);
          taaResultTex = taaResult.texture;
        } else {
          this.taaPass.reset();
        }

        // Step 6: Bloom (extract bright pixels, downsample/upsample pyramid)
        const bloomResult = this.bloomPass.execute(encoder, taaResultTex);
        const bloomCombineRT = this.passManager.getOrCreateTarget("bloom-combine", "rgba16float");
        this.bloomPass.combine(
          encoder,
          taaResultTex,
          bloomResult.view,
          bloomCombineRT.view,
          this.bloomIntensity,
        );

        // Step 7: Post-process (tonemap, saturation, vignette, chromatic) → screen
        this.postProcessPass.execute(
          encoder,
          bloomCombineRT.texture,
          this.gbuffer.depthSampledView,
          screenView,
          [w, h],
          this.frameTime,
        );

        // Step 8: Screen-space outlines — two independent layers sharing one
        // edge-detection machine.
        //
        // Layer A (cel/toon outline): driven by ShadingModelID in the GBuffer
        // material channel. Belongs to the artwork → part of the beauty pass and
        // included when rendering to a file.
        if (this.toonOutline) {
          if (!this.celOutlineBG) {
            this.celOutlineBG = this.device.createBindGroup({
              layout: this.outlineScreenPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: this.gbuffer.materialView },
                { binding: 1, resource: this.outlineSampler },
                { binding: 2, resource: { buffer: this.celOutlineUBO } },
              ],
            });
          }
          const c = this.celOutlineUboData;
          c[0] = w; c[1] = h;
          c[2] = ShadingModel.TOON;          // target id = TOON shading model
          c[3] = this.celOutlineWidth;        // radius (px)
          c[4] = 0.03; c[5] = 0.03; c[6] = 0.04; // near-black cel color
          c[7] = 1.0;                          // enabled
          this.device.queue.writeBuffer(this.celOutlineUBO, 0, c as unknown as GPUAllowSharedBufferSource);
          const celPass = encoder.beginRenderPass({
            label: "outline-cel",
            colorAttachments: [{
              view: screenView,
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "load",
              storeOp: "store",
            }],
          });
          celPass.setPipeline(this.outlineScreenPipeline);
          celPass.setBindGroup(0, this.celOutlineBG);
          celPass.draw(3);
          celPass.end();
        }

        // Layer B (selection highlight): driven by the editor's selected object
        // id (GBuffer normal.w). Editor-only overlay → suppressed for exports by
        // clearing editorOverlay.
        if (this.outlineEnabled && this.editorOverlay && this.selectedId > 0) {
          if (!this.selOutlineBG) {
            this.selOutlineBG = this.device.createBindGroup({
              layout: this.outlineScreenPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: this.gbuffer.normalView },
                { binding: 1, resource: this.outlineSampler },
                { binding: 2, resource: { buffer: this.selOutlineUBO } },
              ],
            });
          }
          const s = this.selOutlineUboData;
          s[0] = w; s[1] = h;
          s[2] = this.selectedId;            // target id = selected object
          s[3] = this.selectionWidth;         // radius (px)
          s[4] = 0.97; s[5] = 0.66; s[6] = 0.11; // Blender-style orange
          s[7] = 1.0;                          // enabled
          this.device.queue.writeBuffer(this.selOutlineUBO, 0, s as unknown as GPUAllowSharedBufferSource);
          const selPass = encoder.beginRenderPass({
            label: "outline-selection",
            colorAttachments: [{
              view: screenView,
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "load",
              storeOp: "store",
            }],
          });
          selPass.setPipeline(this.outlineScreenPipeline);
          selPass.setBindGroup(0, this.selOutlineBG);
          selPass.draw(3);
          selPass.end();
        }

        // Step 9: Gizmo (selection axes) drawn over the final image
        if (this.selectedId > 0) {
          this.buildGizmoData();
          this.device.queue.writeBuffer(this.gizmoVB, 0, this.gizmoData as unknown as GPUAllowSharedBufferSource);
          const gizmoPass = encoder.beginRenderPass({
            label: "gizmo",
            colorAttachments: [{
              view: screenView,
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "load",
              storeOp: "store",
            }],
          });
          gizmoPass.setPipeline(this.gizmoPipeline);
          gizmoPass.setVertexBuffer(0, this.gizmoVB);
          gizmoPass.draw(6);
          gizmoPass.end();
        }
      },
    }];
  }

  stats() {
    return {
      drawCalls: 2 + CSM_CASCADE_COUNT,
      triangles: this.cubeIndexCount / 3,
      custom: {
        "Pipeline": "Deferred (G-Buffer MRT4)",
        "Lights": `${this.lightScene.count} (${this.lightScene.directionalLights.length}D + ${this.lightScene.pointLights.length}P + ${this.lightScene.spotLights.length}S)`,
        "CSM": this.useCSM ? `${CSM_CASCADE_COUNT} cascades` : "Off",
        "SSAO": this.useSSAO ? "GTAO" : "Off",
      },
    };
  }

  private gizmoAxes: [number, number, number][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  private gizmoColors: [number, number, number, number][] = [
    [1, 0.25, 0.25, 1],
    [0.3, 1, 0.3, 1],
    [0.35, 0.45, 1, 1],
  ];
  private GIZMO_LEN = 0.6;

  private buildGizmoData(): void {
    const part = this.charParts.find((p) => p.selectedId === this.selectedId);
    const pos = this.selectedId === 1 ? this.cubePos : this.selectedId === 2 ? this.ballPos : (part ? part.position : this.cubePos);
    const d = this.gizmoData;
    const ndc = this.gizmoNDC;
    for (let a = 0; a < 3; a++) {
      const o = [pos[0], pos[1], pos[2], 1] as const;
      const tip = [
        pos[0] + this.gizmoAxes[a][0] * this.GIZMO_LEN,
        pos[1] + this.gizmoAxes[a][1] * this.GIZMO_LEN,
        pos[2] + this.gizmoAxes[a][2] * this.GIZMO_LEN,
        1,
      ] as const;
      const c0 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(o[0], o[1], o[2], o[3]));
      const c1 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(tip[0], tip[1], tip[2], tip[3]));
      const base = a * 16;
      d[base] = c0[0] / c0[3]; d[base + 1] = c0[1] / c0[3]; d[base + 2] = c0[2] / c0[3]; d[base + 3] = 1;
      d[base + 4] = this.gizmoColors[a][0];
      d[base + 5] = this.gizmoColors[a][1];
      d[base + 6] = this.gizmoColors[a][2];
      d[base + 7] = this.gizmoColors[a][3];
      d[base + 8] = c1[0] / c1[3]; d[base + 9] = c1[1] / c1[3]; d[base + 10] = c1[2] / c1[3]; d[base + 11] = 1;
      d[base + 12] = this.gizmoColors[a][0];
      d[base + 13] = this.gizmoColors[a][1];
      d[base + 14] = this.gizmoColors[a][2];
      d[base + 15] = this.gizmoColors[a][3];
      ndc[a * 4] = c0[0] / c0[3]; ndc[a * 4 + 1] = c0[1] / c0[3];
      ndc[a * 4 + 2] = c1[0] / c1[3]; ndc[a * 4 + 3] = c1[1] / c1[3];
    }
  }

  private eventToNDC(e: PointerEvent): [number, number] {
    const rect = this.ctx.canvas.getBoundingClientRect();
    return [
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -(((e.clientY - rect.top) / rect.height) * 2 - 1),
    ];
  }

  private distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
    const abx = bx - ax, aby = by - ay;
    const len2 = abx * abx + aby * aby;
    let t = len2 > 0 ? ((px - ax) * abx + (py - ay) * aby) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const dx = px - (ax + abx * t);
    const dy = py - (ay + aby * t);
    return Math.hypot(dx, dy);
  }

  private hitGizmoAxis(ndc: [number, number]): number {
    if (this.selectedId <= 0) return 0;
    const thr = 10 / (this.ctx.canvas.height * 0.5);
    let best = 0;
    let bestD = Infinity;
    for (let a = 0; a < 3; a++) {
      const p = a * 4;
      const d = this.distToSegment(ndc[0], ndc[1], this.gizmoNDC[p], this.gizmoNDC[p + 1], this.gizmoNDC[p + 2], this.gizmoNDC[p + 3]);
      if (d < bestD) { bestD = d; best = a + 1; }
    }
    return bestD <= thr ? best : 0;
  }

  private applyGizmoDrag(dxPx: number, dyPx: number): void {
    if (this.dragAxis < 1 || this.dragAxis > 3) return;
    const axis = this.gizmoAxes[this.dragAxis - 1];
    const part = this.selectedId > 2 ? this.charParts.find((p) => p.selectedId === this.selectedId) : null;
    const pos = this.selectedId === 1 ? this.cubePos : this.selectedId === 2 ? this.ballPos : (part ? part.position : null);
    if (!pos) return;
    const p0 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(pos[0], pos[1], pos[2], 1));
    const p1 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(pos[0] + axis[0], pos[1] + axis[1], pos[2] + axis[2], 1));
    let sx = p1[0] / p1[3] - p0[0] / p0[3];
    let sy = p1[1] / p1[3] - p0[1] / p0[3];
    const len = Math.hypot(sx, sy);
    if (len < 1e-6) return;
    sx /= len; sy /= len;
    // screen pixels moved along the axis (NDC y is flipped vs mouse y)
    const sPx = dxPx * sx - dyPx * sy;
    const scale = (2 * this.camera.distance * Math.tan(this.camera.fov * Math.PI / 360)) / this.ctx.canvas.height;
    const t = sPx * scale;
    pos[0] += axis[0] * t;
    pos[1] += axis[1] * t;
    pos[2] += axis[2] * t;
  }

  private onPointerDown = (e: PointerEvent) => {
    if (!this.editMode) return;
    const ndc = this.eventToNDC(e);
    const axis = this.hitGizmoAxis(ndc);
    if (axis > 0) {
      e.stopPropagation();
      e.preventDefault();
      this.ctx.canvas.setPointerCapture(e.pointerId);
      this.dragAxis = axis;
      this.dragging = true;
      this.dragLastX = e.clientX;
      this.dragLastY = e.clientY;
      return;
    }
    this.selectedId = this.pickAtNDC(ndc);
  };

  private onPointerMove = (e: PointerEvent) => {
    if (!this.dragging) return;
    e.stopPropagation();
    e.preventDefault();
    const dx = e.clientX - this.dragLastX;
    const dy = e.clientY - this.dragLastY;
    this.dragLastX = e.clientX;
    this.dragLastY = e.clientY;
    this.applyGizmoDrag(dx, dy);
  };

  private onPointerUp = (e: PointerEvent) => {
    if (!this.dragging) return;
    e.stopPropagation();
    this.dragging = false;
    this.dragAxis = 0;
    this.ctx.canvas.releasePointerCapture(e.pointerId);
  };

  private pickAtNDC(ndc: [number, number]): number {
    // Unproject a ray through the clicked pixel using the last frame's invVP.
    const cam = this.camera.position;
    const near = mat4.mul(this.frameInvViewProj as unknown as Mat4, vec4.create(ndc[0], ndc[1], 0.0, 1.0));
    const rd = vec3.normalize(vec3.create(
      near[0] / near[3] - cam[0],
      near[1] / near[3] - cam[1],
      near[2] / near[3] - cam[2],
    ));

    let bestT = Infinity;
    let bestId = 0;

    // Ball: sphere at ballPos with radius 1
    {
      const ocx = cam[0] - this.ballPos[0];
      const ocy = cam[1] - this.ballPos[1];
      const ocz = cam[2] - this.ballPos[2];
      const b = ocx * rd[0] + ocy * rd[1] + ocz * rd[2];
      const c = ocx * ocx + ocy * ocy + ocz * ocz - 1;
      const disc = b * b - c;
      if (disc >= 0) {
        const t = -b - Math.sqrt(disc);
        if (t > 0 && t < bestT) { bestT = t; bestId = 2; }
      }
    }

    // Cube: OBB via inverse model; local AABB is [-1.5, 1.5] (scale 1.5).
    // Ray param t is preserved under the affine transform.
    {
      const inv = mat4.inverse(this.cubeModel as unknown as Mat4) as unknown as Float32Array;
      const om = mat4.mul(inv, vec4.create(cam[0], cam[1], cam[2], 1));
      const dm = mat4.mul(inv, vec4.create(cam[0] + rd[0], cam[1] + rd[1], cam[2] + rd[2], 1));
      const ox = om[0] / om[3], oy = om[1] / om[3], oz = om[2] / om[3];
      const dx = dm[0] / dm[3] - ox, dy = dm[1] / dm[3] - oy, dz = dm[2] / dm[3] - oz;
      let tmin = -Infinity;
      let tmax = Infinity;
      const slabs = [[dx, ox, -1.5, 1.5], [dy, oy, -1.5, 1.5], [dz, oz, -1.5, 1.5]] as const;
      for (let i = 0; i < 3; i++) {
        const d = slabs[i][0], o = slabs[i][1], lo = slabs[i][2], hi = slabs[i][3];
        if (Math.abs(d) < 1e-8) {
          if (o < lo || o > hi) return bestId;
        } else {
          let t1 = (lo - o) / d;
          let t2 = (hi - o) / d;
          if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
          tmin = Math.max(tmin, t1);
          tmax = Math.min(tmax, t2);
          if (tmin > tmax) return bestId;
        }
      }
      const t = Math.max(tmin, 0);
      if (t < bestT) { bestT = t; bestId = 1; }
    }

    // Character parts (SKIN/HAIR/EYE): OBB via inverse model; sphere geometry
    // is a unit sphere (AABB [-1, 1]), cube geometry is [-1.5, 1.5].
    for (const p of this.charParts) {
      const half = p.geometry === "sphere" ? 1 : 1.5;
      const t = this.rayHitOBB(cam, rd, p.model, half);
      if (t < bestT) { bestT = t; bestId = p.objectId; }
    }

    return bestId;
  }

  /** Slab test against an oriented box: model transforms the unit sphere /
   *  cube geometry; returns the ray parameter t (> 0) or Infinity on miss. */
  private rayHitOBB(ro: ArrayLike<number>, rd: ArrayLike<number>, model: Float32Array, half: number): number {
    const inv = mat4.inverse(model as unknown as Mat4) as unknown as Float32Array;
    const om = mat4.mul(inv, vec4.create(ro[0], ro[1], ro[2], 1));
    const dm = mat4.mul(inv, vec4.create(ro[0] + rd[0], ro[1] + rd[1], ro[2] + rd[2], 1));
    const ox = om[0] / om[3], oy = om[1] / om[3], oz = om[2] / om[3];
    const dx = dm[0] / dm[3] - ox, dy = dm[1] / dm[3] - oy, dz = dm[2] / dm[3] - oz;
    let tmin = -Infinity;
    let tmax = Infinity;
    for (const s of [[dx, ox], [dy, oy], [dz, oz]] as const) {
      const d = s[0], o = s[1];
      if (Math.abs(d) < 1e-8) {
        if (o < -half || o > half) return Infinity;
      } else {
        let t1 = (-half - o) / d;
        let t2 = (half - o) / d;
        if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
        tmin = Math.max(tmin, t1);
        tmax = Math.min(tmax, t2);
        if (tmin > tmax) return Infinity;
      }
    }
    return Math.max(tmin, 0);
  }

  registerGUI(gui: any) {
    gui.add(this, "metallic", 0, 1, 0.01).name("Metallic");
    gui.add(this, "roughness", 0.01, 1, 0.01).name("Roughness");
    gui.add(this, "bloomIntensity", 0, 1, 0.01).name("Bloom Intensity");
    gui.add(this, "useCSM").name("CSM Shadows");
    gui.add(this, "useSSAO").name("GTAO");
    gui.add(this, "useTAA").name("TAA");
    gui.add(this, "taaJitter").name("TAA Jitter");
    gui.add(this, "pauseAnimation").name("Pause");
    const editFolder = gui.addFolder("Edit");
    editFolder.add(this, "editMode").name("Edit Mode (pick & move)");
    editFolder.add(this, "selectedId", 0, 5, 1).name("Selection (1=cube 2=sphere 3=skin 4=hair 5=eye)").listen();
    editFolder.add(this, "outlineEnabled").name("Selection Outline");
    editFolder.add(this, "editorOverlay").name("Editor Overlay (off for export)");
    editFolder.add(this, "selectionWidth", 1, 6, 1).name("Selection Width (px)");
    gui.add(this.taaPass, "alpha", 0.02, 0.5, 0.01).name("TAA Alpha");
    const toonFolder = gui.addFolder("Toon Sphere (ShadingModelID=TOON)");
    toonFolder.add(this, "toonBands", 1, 6, 1).name("Cel Bands");
    toonFolder.add(this, "toonR", 0, 1, 0.01).name("Base R");
    toonFolder.add(this, "toonG", 0, 1, 0.01).name("Base G");
    toonFolder.add(this, "toonB", 0, 1, 0.01).name("Base B");
    toonFolder.add(this, "toonOutline").name("Cel Outline");
    toonFolder.add(this, "celOutlineWidth", 1, 4, 0.5).name("Cel Outline Width (px)");
    const charFolder = gui.addFolder("Character (Step 2: SKIN/HAIR/EYE)");
    charFolder.add(this, "skinSss", 0, 1, 0.01).name("Skin SSS Strength");
    charFolder.add(this, "skinRoughness", 0.01, 1, 0.01).name("Skin Roughness");
    charFolder.add(this, "hairRoughness", 0.01, 1, 0.01).name("Hair Roughness");
    charFolder.add(this, "hairAniso", 0, 1, 0.01).name("Hair Anisotropy");
    charFolder.add(this, "eyeCornea", 0, 1, 0.01).name("Eye Cornea Highlight");
    charFolder.add(this, "eyeIrisDark", 0.05, 1, 0.01).name("Eye Iris Darkness");
    const taaFolder = gui.addFolder("TAA Debug");
    taaFolder.add(this.taaPass, "debugMode", {
      "OFF": 0,
      "Motion": 1,
      "No-Reproj": 2,
      "No-History": 3,
    }).name("Mode");
    gui.add(this.deferredLighting, "envIntensity", 0, 3, 0.05).name("IBL Intensity");
    gui.add(this.bloomPass, "threshold", 0, 2, 0.01).name("Bloom Threshold");
    const pp = this.postProcessPass.params;
    const fxFolder = gui.addFolder("Post Process");
    fxFolder.add(pp, "exposure", 0.1, 3, 0.01).name("Exposure");
    fxFolder.add(pp, "chromaticStrength", 0, 0.05, 0.001).name("Chromatic Aberr.");
    fxFolder.add(pp, "vignetteStrength", 0, 1, 0.01).name("Vignette");
    fxFolder.add(pp, "saturation", 0, 2, 0.01).name("Saturation");
  }

  destroy() {
    this.ctx.canvas.removeEventListener("pointerdown", this.onPointerDown, true);
    this.ctx.canvas.removeEventListener("pointermove", this.onPointerMove, true);
    this.ctx.canvas.removeEventListener("pointerup", this.onPointerUp, true);
    this.ctx.canvas.removeEventListener("pointercancel", this.onPointerUp, true);
    this.cubeVB.destroy();
    this.cubeIB.destroy();
    this.sphereVB.destroy();
    this.sphereIB.destroy();
    this.globalsCubeUBO.destroy();
    this.globalsSphereUBO.destroy();
    this.standardMaterial.destroy();
    this.toonMaterial.destroy();
    for (const p of this.charParts) {
      p.globalsUBO?.destroy();
      p.material.destroy();
    }
    this.shadowUBO.destroy();
    this.celOutlineUBO.destroy();
    this.selOutlineUBO.destroy();
    this.gizmoVB.destroy();
    this.dummyDepthTexture.destroy();
    this.gbuffer.destroy();
    this.passManager.destroy();
    this.deferredLighting.destroy();
    this.csm.destroy();
    this.gtao.destroy();
    this.envMap.destroy();
    this.brdfLut.destroy();
    this.bloomPass.destroy();
    this.taaPass.destroy();
    this.postProcessPass.destroy();
  }
}