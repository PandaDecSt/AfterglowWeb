import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";
import { Skeleton, type BoneDesc } from "../scene/skeleton";
import { Skinning } from "../scene/skinning";
import { AnimationPlayer } from "../scene/animation-player";
import { loadPMX, type PMXModel, type PMXMaterial } from "../utils/pmx-loader";
import { ShadowMap, SHADOW_WGSL } from "../passes/shadow";
import { BloomPass } from "../passes/bloom";
import { HDRRenderTarget } from "../passes/hdr";
import { mat4, quat, vec3 } from "wgpu-matrix";
import { decodeTGA } from "../utils/tga-loader";
import { BrdfLut, BRDF_LUT_SIZE } from "../passes/brdf-lut";
import { loadVMD, vmdDuration } from "../utils/vmd-loader";
import { buildIKChains, solveIK, type IKChain } from "../scene/ik-solver";

const HDR_FORMAT = "rgba16float";


type RenderClass = "auto" | "eye" | "hair";

interface PresetConfig {
  metallic: number;
  roughness: number;
  emissionStrength: number;
  nprMix: number;
  rimColor: [number, number, number];
  rimStrength: number;
  rimPower: number;
  alphaMode?: number;
  renderClass: RenderClass;
}

const PRESETS: Record<string, PresetConfig> = {
  default:      { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.0, rimColor: [1, 1, 1], rimStrength: 0.0, rimPower: 3.0, renderClass: "auto" },
  body:         { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.5, rimColor: [1, 0.85, 0.7], rimStrength: 0.3, rimPower: 3.0, renderClass: "auto" },
  face:         { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.5, rimColor: [1, 0.9, 0.8], rimStrength: 0.2, rimPower: 4.0, renderClass: "auto" },
  hair:         { metallic: 0.0, roughness: 0.3, emissionStrength: 0.0, nprMix: 0.2, rimColor: [1, 1, 1], rimStrength: 0.4, rimPower: 2.5, renderClass: "hair" },
  eye:          { metallic: 0.0, roughness: 0.1, emissionStrength: 1.5, nprMix: 0.0, rimColor: [1, 1, 1], rimStrength: 0.0, rimPower: 3.0, renderClass: "eye" },
  eyelash:      { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.3, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 3.0, renderClass: "eye" },
  metal:        { metallic: 1.0, roughness: 0.3, emissionStrength: 0.0, nprMix: 0.3, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 5.0, renderClass: "auto" },
  stockings:    { metallic: 0.0, roughness: 0.8, emissionStrength: 0.0, nprMix: 0.0, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 3.0, alphaMode: 1, renderClass: "auto" },
  cloth_smooth: { metallic: 0.0, roughness: 0.6, emissionStrength: 0.0, nprMix: 0.1, rimColor: [1, 1, 1], rimStrength: 0.15, rimPower: 3.0, renderClass: "auto" },
  cloth_rough:  { metallic: 0.0, roughness: 0.82, emissionStrength: 0.0, nprMix: 0.1, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 3.5, renderClass: "auto" },
};

function detectPreset(name: string, isTransparent: boolean): PresetConfig {
  const n = name.toLowerCase();
  if (n.includes("顔") || n.includes("面") || n.includes("face")) return PRESETS.face;
  if (n.includes("睫") || n.includes("まつげ") || n.includes("まつ毛") || n.includes("eyelash")) return PRESETS.eyelash;
  if (n.includes("髪") || n.includes("毛") || n.includes("hair")) return PRESETS.hair;
  if (n.includes("目") || n.includes("眼") || n.includes("eye") || n.includes("瞳")) return PRESETS.eye;
  if (n.includes("金属") || n.includes("metal") || n.includes("メタル")) return PRESETS.metal;
  if (n.includes("ストッキング") || n.includes("靴下") || n.includes("stocking") || n.includes("ニーソ")) return PRESETS.stockings;
  if (n.includes("服") || n.includes("衣") || n.includes("cloth") || n.includes("シャツ") || n.includes("スカート")) return PRESETS.cloth_smooth;
  if (n.includes("肌") || n.includes("体") || n.includes("body") || n.includes("skin")) return PRESETS.body;
  if (isTransparent) return PRESETS.stockings;
  return PRESETS.default;
}

const SCENE_VS = `
struct Scene {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> scene: Scene;

@group(2) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
};

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

fn safe_normal(n: vec3<f32>) -> vec3<f32> {
  let l2 = dot(n, n);
  if (l2 < 1e-12) { return vec3<f32>(0.0, 1.0, 0.0); }
  return n * inverseSqrt(l2);
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
  var out: VSOut;

  let weightSum = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
  let invW = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let w = select(vec4<f32>(1.0, 0.0, 0.0, 0.0), in.weights * invW, weightSum > 0.0001);

  var skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var skinNrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let pos4 = vec4<f32>(in.position, 1.0);
  let nrm4 = vec4<f32>(in.normal, 0.0);
  for (var i = 0u; i < 4u; i++) {
    let j = in.joints[i];
    skinPos += skinMatrices[j] * pos4 * w[i];
    skinNrm += skinMatrices[j] * nrm4 * w[i];
  }

  let worldPos = (scene.model * skinPos).xyz;
  out.position = scene.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldNormal = safe_normal((scene.model * skinNrm).xyz);
  out.worldPos = worldPos;
  out.uv = in.uv;
  return out;
}
`;

const MAIN_FS = `
override IS_OVER_EYES: bool = false;
override IS_EYE: bool = false;

struct Scene {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> scene: Scene;

struct Mat {
  diffuseColor_alpha: vec4<f32>,
  ambient_shininess: vec4<f32>,
  specular_sphereMode: vec4<f32>,
  pbrParams: vec4<f32>,
  rimColor_strength: vec4<f32>,
  rimPower_pad: vec4<f32>,
};
@group(0) @binding(1) var<uniform> mat: Mat;

@group(0) @binding(2) var diffuseTex: texture_2d<f32>;
@group(0) @binding(3) var sphereTex: texture_2d<f32>;
@group(0) @binding(4) var toonTex: texture_2d<f32>;
@group(0) @binding(5) var texSampler: sampler;
@group(0) @binding(6) var brdfLut: texture_2d<f32>;

@group(1) @binding(0) var shadowTex: texture_depth_2d;
@group(1) @binding(1) var shadowSampler: sampler_comparison;
@group(1) @binding(2) var<uniform> lightVP: mat4x4<f32>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

${SHADOW_WGSL}

fn ggxNDF(NH: f32, a2: f32) -> f32 {
  let d = NH * NH * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d);
}

fn smithG2(NL: f32, NV: f32, k: f32) -> f32 {
  return 1.0 / (4.0 * (NL * (1.0 - k) + k) * (NV * (1.0 - k) + k));
}

fn fresnelSchlick(VH: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(1.0 - VH, 5.0);
}

fn brdf_lut_sample(NV: f32, roughness: f32) -> vec4<f32> {
  let LUT_SIZE: f32 = ${BRDF_LUT_SIZE}.0;
  var uv = vec2f(clamp(roughness, 0.0, 1.0), sqrt(clamp(1.0 - NV, 0.0, 1.0)));
  uv = uv * ((LUT_SIZE - 1.0) / LUT_SIZE) + 0.5 / LUT_SIZE;
  return textureSampleLevel(brdfLut, texSampler, uv, 0.0);
}

fn F_brdf_multi_scatter(f0: vec3<f32>, f90: vec3<f32>, lut: vec2<f32>) -> vec3<f32> {
  let FssEss = lut.y * f90 + lut.x * f0;
  let Ess = lut.x + lut.y;
  let Ems = 1.0 - Ess;
  let Favg = f0 + (1.0 - f0) / 21.0;
  let Fms = FssEss * Favg / (1.0 - (1.0 - Ess) * Favg);
  return FssEss + Fms * Ems;
}

fn ltc_brdf_scale(lut: vec4<f32>) -> f32 {
  return (lut.z + lut.w) / max(lut.x + lut.y, 1e-6);
}

fn hash3(p3: vec3<f32>) -> f32 {
  var p = fract(p3 * 0.1031);
  p += dot(p, p.yzx + 33.33);
  return fract((p.x + p.y) * p.z);
}

fn hashedAlphaThreshold(co: vec3<f32>, alphaHashScale: f32) -> f32 {
  let dx = dpdx(co);
  let dy = dpdy(co);
  let maxDeriv = max(length(dx), length(dy));
  let pixScale = 1.0 / max(alphaHashScale * maxDeriv, 1e-6);
  let pixScaleFloor = floor(pixScale);
  let baseHash = hash3(floor(co * pixScaleFloor));
  let nextHash = hash3(floor(co * pixScaleFloor) + vec3<f32>(1.0));
  let fracPart = fract(pixScale);
  return mix(baseHash, nextHash, fracPart);
}

struct FSOut {
  @location(0) color: vec4<f32>,
  @location(1) mask: vec4<f32>,
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
  if (IS_EYE && scene.cameraPos.z < in.worldPos.z) { discard; }
  let tex_s = textureSample(diffuseTex, texSampler, in.uv);
  let alpha = mat.diffuseColor_alpha.w * tex_s.a;
  if (mat.rimPower_pad.y > 0.5) {
    let threshold = hashedAlphaThreshold(in.worldPos, 1.0);
    if (alpha < threshold) { discard; }
  } else {
    if (alpha < 0.001) { discard; }
  }

  var n = normalize(in.worldNormal);
  let v = normalize(scene.cameraPos.xyz - in.worldPos);
  n = select(-n, n, dot(n, v) >= 0.0);

  let l = normalize(scene.lightDir.xyz);
  let h = normalize(v + l);
  let NL = max(dot(n, l), 0.0);
  let NV = max(dot(n, v), 0.0);
  let NH = max(dot(n, h), 0.0);
  let VH = max(dot(v, h), 0.0);

  let baseColor = tex_s.rgb * mat.diffuseColor_alpha.xyz;
  let shadow = sampleShadowPCF(shadowTex, shadowSampler, lightVP, in.worldPos, n, l);

  let toonT = clamp(NL * 0.5 + 0.5, 0.0, 1.0);
  let toonShade = textureSample(toonTex, texSampler, vec2<f32>(toonT, 0.5)).r;

  let nprDiffuse = baseColor * toonShade * shadow;
  let nprSpecular = pow(NH, max(mat.ambient_shininess.w, 1.0)) * mat.specular_sphereMode.xyz * shadow;

  let metallic = mat.pbrParams.x;
  let roughness = max(mat.pbrParams.y, 0.04);
  let a2 = roughness * roughness * roughness * roughness;
  let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
  let f90 = vec3<f32>(1.0);

  let lut = brdf_lut_sample(NV, roughness);
  let F_ms = F_brdf_multi_scatter(F0, f90, lut.xy);

  let pbrDiffuse = baseColor * (1.0 - metallic) / 3.14159265 * NL * shadow;
  let D = ggxNDF(NH, a2);
  let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  let G = smithG2(NL, NV, k);
  let pbrSpecular = D * G * F_ms * shadow;

  let nprMix = mat.pbrParams.w;
  let diffuse = mix(pbrDiffuse, nprDiffuse, nprMix);
  let specular = mix(pbrSpecular, nprSpecular, nprMix);

  let rimFactor = pow(1.0 - NV, mat.rimPower_pad.x) * mat.rimColor_strength.w;
  let rim = mat.rimColor_strength.xyz * rimFactor; // * shadow; 边缘光本质是视角效应，不需要被阴影遮挡。

  let emission = baseColor * mat.pbrParams.z;

  var sphereAdd = vec3<f32>(0.0);
  if (mat.specular_sphereMode.w > 0.5 && mat.specular_sphereMode.w < 1.5) {
    let viewN = (scene.viewProj * vec4<f32>(n, 0.0)).xy;
    let sphereUV = viewN * 0.5 + vec2<f32>(0.5, 0.5);
    sphereAdd = textureSample(sphereTex, texSampler, sphereUV).rgb * baseColor;
  } else if (mat.specular_sphereMode.w > 1.5) {
    sphereAdd = textureSample(sphereTex, texSampler, in.uv).rgb * baseColor;
  }

  let ambient = baseColor * mat.ambient_shininess.xyz;
  let color = ambient + (diffuse + specular) * scene.lightColor.rgb + rim + emission + sphereAdd;
  var outAlpha = alpha;
  if (IS_OVER_EYES) { outAlpha = alpha * 0.25; }
  var out: FSOut;
  out.color = vec4<f32>(color, outAlpha);
  out.mask = vec4<f32>(1.0, outAlpha, 0.0, 0.0);
  return out;
}
`;


const SHADOW_VS = `
struct ShadowScene {
  lightVP: mat4x4<f32>,
  model: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> shadowScene: ShadowScene;

@group(1) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

@vertex
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
  var skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let pos4 = vec4<f32>(position, 1.0);
  for (var i = 0u; i < 4u; i++) {
    let j = joints[i];
    let w = weights[i];
    skinPos += skinMatrices[j] * pos4 * w;
  }
  let worldPos = (shadowScene.model * skinPos).xyz;
  return shadowScene.lightVP * vec4<f32>(worldPos, 1.0);
}
`;

const OUTLINE_VS = `
struct Scene {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> scene: Scene;

struct OutlineMat {
  edgeColor: vec4<f32>,
  edgeSize: f32,
  _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(1) var<uniform> omat: OutlineMat;

@group(2) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
) -> VSOut {
  var out: VSOut;

  var skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var skinNrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let pos4 = vec4<f32>(position, 1.0);
  let nrm4 = vec4<f32>(normal, 0.0);
  for (var i = 0u; i < 4u; i++) {
    let j = joints[i];
    let w = weights[i];
    skinPos += skinMatrices[j] * pos4 * w;
    skinNrm += skinMatrices[j] * nrm4 * w;
  }

  let worldPos = (scene.model * skinPos).xyz;
  let worldNrm = normalize((scene.model * skinNrm).xyz);
  let clipPos = scene.viewProj * vec4<f32>(worldPos, 1.0);
  let viewNrm = (scene.viewProj * vec4<f32>(worldNrm, 0.0)).xyz;
  let screenNrm = normalize(viewNrm.xy);
  let offset = screenNrm * (omat.edgeSize * 0.003) * clipPos.w;
  out.position = vec4<f32>(clipPos.xy + offset, clipPos.z, clipPos.w);
  out.uv = uv;
  return out;
}
`;

const OUTLINE_FS = `
struct OutlineMat {
  edgeColor: vec4<f32>,
  edgeSize: f32,
  _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(1) var<uniform> omat: OutlineMat;
@group(0) @binding(2) var diffuseTex: texture_2d<f32>;
@group(0) @binding(5) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct OutlineFSOut {
  @location(0) color: vec4<f32>,
  @location(1) mask: vec4<f32>,
};

@fragment
fn fs_main(in: VSOut) -> OutlineFSOut {
  let texA = textureSample(diffuseTex, texSampler, in.uv).a;
  if (texA < 0.05) { discard; }
  var out: OutlineFSOut;
  out.color = vec4<f32>(omat.edgeColor.rgb, omat.edgeColor.a * texA);
  out.mask = vec4<f32>(1.0, omat.edgeColor.a * texA, 0.0, 0.0);
  return out;
}
`;

interface MatRenderData {
  indexOffset: number;
  indexCount: number;
  mainBG: GPUBindGroup;
  shadowBG: GPUBindGroup;
  outlineBG: GPUBindGroup | null;
  isTransparent: boolean;
  hasEdge: boolean;
  renderClass: RenderClass;
  castsShadow: boolean;
}

function create1x1Texture(device: GPUDevice, r: number, g: number, b: number, a: number, label: string): GPUTexture {
  const tex = device.createTexture({ label, size: [1, 1], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, new Uint8Array([r, g, b, a]), { bytesPerRow: 4 }, [1, 1]);
  return tex;
}

function createToonRampTexture(device: GPUDevice): GPUTexture {
  const h = 64;
  const data = new Uint8Array(h * 4);
  for (let y = 0; y < h; y++) {
    const v = y / (h - 1);
    const t = Math.min(1, Math.max(0, (v - 0.5) / 0.1));
    const s = t * t * (3 - 2 * t);
    data[y * 4 + 0] = Math.round(255 - s * (255 - 196));
    data[y * 4 + 1] = Math.round(255 - s * (255 - 186));
    data[y * 4 + 2] = Math.round(255 - s * (255 - 205));
    data[y * 4 + 3] = 255;
  }
  const tex = device.createTexture({ label: "toon-ramp", size: [1, h], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: 4 }, [1, h]);
  return tex;
}

async function loadTextureImage(device: GPUDevice, url: string, label: string): Promise<GPUTexture | null> {
  try {
    if (url.toLowerCase().endsWith(".tga")) {
      const resp = await fetch(url);
      if (!resp.ok) return null;
      const buffer = await resp.arrayBuffer();
      const tga = decodeTGA(buffer);
      const tex = device.createTexture({
        label,
        size: [tga.width, tga.height],
        format: "rgba8unorm-srgb",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const tgaBytes = new Uint8Array(tga.data.buffer.slice(tga.data.byteOffset, tga.data.byteOffset + tga.data.byteLength));
      device.queue.writeTexture({ texture: tex }, tgaBytes as unknown as GPUAllowSharedBufferSource, { bytesPerRow: tga.width * 4 }, [tga.width, tga.height]);
      return tex;
    }
    const resp = await fetch(url);
    if (!resp.ok) return null;
    const blob = await resp.blob();
    const bitmap = await createImageBitmap(blob);
    const tex = device.createTexture({ label, size: [bitmap.width, bitmap.height], format: "rgba8unorm-srgb", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, [bitmap.width, bitmap.height]);
    bitmap.close();
    return tex;
  } catch { return null; }
}

export class PMXDemo implements Demo {
  label = "PMX Viewer";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;

  private mainPipeline!: GPURenderPipeline;
  private eyePipeline!: GPURenderPipeline;
  private hairPipeline!: GPURenderPipeline;
  private hairOverEyesPipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;
  private outlinePipeline!: GPURenderPipeline;
  private sceneBuffer!: GPUBuffer;
  private sceneData = new Float32Array(80);
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private totalIndexCount = 0;
  private use32bit = false;

  private shadowMap!: ShadowMap;
  private shadowBGLayout!: GPUBindGroupLayout;
  private hdrTarget!: HDRRenderTarget;
  private bloom!: BloomPass;
  private brdfLut!: BrdfLut;

  private resolvedHDR: GPUTexture | null = null;
  private resolvedHDRView: GPUTextureView | null = null;
  private bloomMaskTex: GPUTexture | null = null;
  private bloomMaskView: GPUTextureView | null = null;
  private bloomOutput: GPUTexture | null = null;
  private bloomOutputView: GPUTextureView | null = null;
  private tonePipeline: GPURenderPipeline | null = null;
  private toneBindGroup: GPUBindGroup | null = null;
  private toneUBO!: GPUBuffer;
  private gradeUBO!: GPUBuffer;
  private grade2UBO!: GPUBuffer;
  private toneSampler!: GPUSampler;

  private matRenders: MatRenderData[] = [];
  private opaqueOrder: MatRenderData[] = [];
  private gpuTextures: GPUTexture[] = [];

  private skeleton: Skeleton | null = null;
  private skinning: Skinning | null = null;
  private animPlayer: AnimationPlayer | null = null;
  private ikChains: IKChain[] = [];

  private skinMatrixStorageBuffer!: GPUBuffer;
  private skinBGL!: GPUBindGroupLayout;
  private skinBG!: GPUBindGroup;

  private shadowSceneBuffer!: GPUBuffer;
  private shadowSceneData = new Float32Array(32);
  private shadowSceneBG!: GPUBindGroup;

  bloomEnabled = true;
  tonemapEnabled = true;
  stencilEnabled = true;
  gradeEnabled = true;
  debugIK = false;
  animPaused = false;
  lightX = 5;
  lightY = 10;
  lightZ = 8;
  shadowRes = 2048;

  private loaded = false;
  private _renderLogOnce = true;
  private _skinDebugOnce = false;
  private _ikDbgTimer = 0;
  private _ikDbgTimer2 = 0;
  private _legVertCount = 0;
  private _skinDebugOnce2 = false;
  private _gpuSkinDebug = false;
  private _depthTex: GPUTexture | null = null;
  private _depthW = 0;
  private _depthH = 0;

  async init(ctx: GPUContext, camera: Camera, engine?: EngineContext): Promise<void> {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;

    try {
      const pmx = await loadPMX("/model.pmx");
      console.log(`[PMXDemo] Loaded: ${pmx.name}, V:${pmx.vertices.length} I:${pmx.indices.length} M:${pmx.materials.length} B:${pmx.bones.length} T:${pmx.textures.length}`);
      await this.setupFromPMX(pmx);
      this.loaded = true;
    } catch (e) {
      console.error("[PMXDemo] Load failed:", e);
    }
  }

  private async setupFromPMX(pmx: PMXModel): Promise<void> {
    const vertexCount = pmx.vertices.length;
    const vertexStride = 56;
    const vertexBuf = new ArrayBuffer(vertexCount * vertexStride);
    const dv = new DataView(vertexBuf);
    let minPos = [Infinity, Infinity, Infinity];
    let maxPos = [-Infinity, -Infinity, -Infinity];

    for (let i = 0; i < vertexCount; i++) {
      const v = pmx.vertices[i];
      const off = i * vertexStride;
      dv.setFloat32(off, v.position[0], true);
      dv.setFloat32(off + 4, v.position[1], true);
      dv.setFloat32(off + 8, v.position[2], true);
      dv.setFloat32(off + 12, v.normal[0], true);
      dv.setFloat32(off + 16, v.normal[1], true);
      dv.setFloat32(off + 20, v.normal[2], true);
      dv.setFloat32(off + 24, v.uv[0], true);
      dv.setFloat32(off + 28, v.uv[1], true);
      const bj = v.boneIndices;
      const bw = v.boneWeights;
      dv.setUint16(off + 32, bj.length > 0 ? Math.max(0, Math.min(65535, bj[0])) : 0, true);
      dv.setUint16(off + 34, bj.length > 1 ? Math.max(0, Math.min(65535, bj[1])) : 0, true);
      dv.setUint16(off + 36, bj.length > 2 ? Math.max(0, Math.min(65535, bj[2])) : 0, true);
      dv.setUint16(off + 38, bj.length > 3 ? Math.max(0, Math.min(65535, bj[3])) : 0, true);
      dv.setFloat32(off + 40, bw.length > 0 ? bw[0] : 0, true);
      dv.setFloat32(off + 44, bw.length > 1 ? bw[1] : 0, true);
      dv.setFloat32(off + 48, bw.length > 2 ? bw[2] : 0, true);
      dv.setFloat32(off + 52, bw.length > 3 ? bw[3] : 0, true);
      for (let k = 0; k < 3; k++) { if (v.position[k] < minPos[k]) minPos[k] = v.position[k]; if (v.position[k] > maxPos[k]) maxPos[k] = v.position[k]; }
      if (i < 5) {
        console.log(`[VERT ${i}] boneType=${v.boneType} boneIndices=[${Array.from(v.boneIndices)}] boneWeights=[${Array.from(v.boneWeights).map(w=>w.toFixed(3)).join(",")}]`);
      }
      if (i === 1708) {
        for (const bIdx of v.boneIndices) {
          if (bIdx < pmx.bones.length) console.log(`  bone ${bIdx} = ${pmx.bones[bIdx].name} parent=${pmx.bones[bIdx].parentIndex}`);
        }
      }
      if ((bj[0] >= 18 && bj[0] <= 21) || (bj[1] >= 18 && bj[1] <= 21) || (bj[2] >= 18 && bj[2] <= 21) || (bj[3] >= 18 && bj[3] <= 21)) {
        if (this._legVertCount < 10) {
          const dj0 = dv.getUint16(off + 32, true);
          const dj1 = dv.getUint16(off + 34, true);
          const dw0 = dv.getFloat32(off + 40, true);
          const dw1 = dv.getFloat32(off + 44, true);
          console.log(`[LEGVERT ${i}] pmx_joints=[${Array.from(bj)}] pmx_weights=[${Array.from(bw).map(w=>w.toFixed(3)).join(",")}] vb_joints=[${dj0},${dj1}] vb_weights=[${dw0.toFixed(3)},${dw1.toFixed(3)}]`);
          this._legVertCount++;
        }
      }
    }

    const center = [(minPos[0] + maxPos[0]) / 2, (minPos[1] + maxPos[1]) / 2, (minPos[2] + maxPos[2]) / 2];
    const extent = [maxPos[0] - minPos[0], maxPos[1] - minPos[1], maxPos[2] - minPos[2]];
    const radius = Math.sqrt(extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2) / 2;
    this.camera.orbit(vec3.create(center[0], center[1], center[2]), radius * 2.5, radius * 0.01, radius * 20, Math.PI / 2, 0);
    console.log(`[PMXDemo] Bounds: center=[${center.map(v => v.toFixed(2))}] radius=${radius.toFixed(2)} orbitDist=${(radius * 2.5).toFixed(2)}`);

    this.vertexBuffer = this.device.createBuffer({ label: "pmx-vb", size: vertexBuf.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(this.vertexBuffer.getMappedRange()).set(new Uint8Array(vertexBuf));
    this.vertexBuffer.unmap();

    this.use32bit = vertexCount > 65535;
    this.totalIndexCount = pmx.indices.length;
    const indexSize = this.use32bit ? 4 : 2;
    this.indexBuffer = this.device.createBuffer({ label: "pmx-ib", size: this.totalIndexCount * indexSize, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    if (this.use32bit) { new Int32Array(this.indexBuffer.getMappedRange()).set(pmx.indices); }
    else { new Uint16Array(this.indexBuffer.getMappedRange()).set(pmx.indices); }
    this.indexBuffer.unmap();

    this.shadowMap = new ShadowMap(this.device, 2048);
    this.shadowMap.orthoSize = 64;
    this.shadowMap.near = 1;
    this.shadowMap.far = 140;

    this.buildPipelines();

    this.sceneBuffer = this.device.createBuffer({ label: "pmx-scene-ubo", size: this.sceneData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.shadowSceneBuffer = this.device.createBuffer({ label: "pmx-shadow-scene-ubo", size: this.shadowSceneData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const shadowGroup0Layout = this.shadowPipeline.getBindGroupLayout(0);
    this.shadowSceneBG = this.device.createBindGroup({
      label: "pmx-shadow-scene-bg",
      layout: shadowGroup0Layout,
      entries: [{ binding: 0, resource: { buffer: this.shadowSceneBuffer } }],
    });

    const skinBoneCount = Math.max(1, pmx.bones.length);
    this.skinMatrixStorageBuffer = this.device.createBuffer({
      label: "pmx-skin-storage",
      size: skinBoneCount * 16 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (pmx.bones.length === 0) {
      const identity = new Float32Array(16);
      identity[0] = 1; identity[5] = 1; identity[10] = 1; identity[15] = 1;
      this.device.queue.writeBuffer(this.skinMatrixStorageBuffer, 0, identity as unknown as GPUAllowSharedBufferSource);
    }
    this.skinBG = this.device.createBindGroup({
      label: "pmx-skin-bg",
      layout: this.skinBGL,
      entries: [{ binding: 0, resource: { buffer: this.skinMatrixStorageBuffer } }],
    });

    this.brdfLut = new BrdfLut();
    this.brdfLut.bake(this.device);

    const defaultTex = create1x1Texture(this.device, 255, 255, 255, 255, "default-white");
    const toonRampTex = createToonRampTexture(this.device);
    this.gpuTextures.push(defaultTex, toonRampTex);

    const loadedTextures: (GPUTexture | null)[] = [defaultTex];
    for (let i = 0; i < pmx.textures.length; i++) {
      let texPath = pmx.textures[i].path.replace(/\\/g, "/");
      if (!texPath.startsWith("/")) texPath = "/" + texPath;
      if (texPath.startsWith("//")) texPath = texPath.slice(1);
      const tex = await loadTextureImage(this.device, texPath, `pmx-tex-${i}`);
      if (i < 3) console.log(`[PMXDemo] tex ${i}: path="${pmx.textures[i].path}" url="${texPath}" loaded=${tex !== null}`);
      if (tex) this.gpuTextures.push(tex);
      loadedTextures.push(tex);
    }

    const sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" });
    const mainBGL = this.mainPipeline.getBindGroupLayout(0);
    const outlineBGL = this.outlinePipeline.getBindGroupLayout(0);

    let indexOffset = 0;
    for (let mi = 0; mi < pmx.materials.length; mi++) {
      const m = pmx.materials[mi];
      const matIndexCount = m.faceCount;
      const isTransparent = m.diffuse[3] < 1.0 - 0.001;
      const hasEdge = (m.flag & 0x10) !== 0 && m.edgeScale > 0;

      const matBuf = this.device.createBuffer({ label: `pmx-mat-${mi}`, size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const preset = detectPreset(m.name, isTransparent);
      const matData = new Float32Array(24);
      matData[0] = m.diffuse[0]; matData[1] = m.diffuse[1]; matData[2] = m.diffuse[2]; matData[3] = m.diffuse[3];
      matData[4] = m.ambient[0]; matData[5] = m.ambient[1]; matData[6] = m.ambient[2]; matData[7] = m.specularPower;
      matData[8] = m.specular[0]; matData[9] = m.specular[1]; matData[10] = m.specular[2]; matData[11] = m.sphereMode;
      matData[12] = preset.metallic; matData[13] = preset.roughness; matData[14] = preset.emissionStrength; matData[15] = preset.nprMix;
      matData[16] = preset.rimColor[0]; matData[17] = preset.rimColor[1]; matData[18] = preset.rimColor[2]; matData[19] = preset.rimStrength;
      matData[20] = preset.rimPower;
      matData[21] = preset.alphaMode ?? 0;
      this.device.queue.writeBuffer(matBuf, 0, matData as unknown as GPUAllowSharedBufferSource);

      const diffuseTex = (m.textureIndex >= 0 && m.textureIndex + 1 < loadedTextures.length && loadedTextures[m.textureIndex + 1]) ? loadedTextures[m.textureIndex + 1]! : defaultTex;
      const sphereTex = (m.sphereTextureIndex >= 0 && m.sphereTextureIndex + 1 < loadedTextures.length && loadedTextures[m.sphereTextureIndex + 1]) ? loadedTextures[m.sphereTextureIndex + 1]! : defaultTex;
      const toonTex = (m.toonSharing === 0 && m.toonTextureIndex >= 0 && m.toonTextureIndex + 1 < loadedTextures.length && loadedTextures[m.toonTextureIndex + 1])
        ? loadedTextures[m.toonTextureIndex + 1]! : toonRampTex;

      const mainBG = this.device.createBindGroup({
        label: `pmx-main-bg-${mi}`, layout: mainBGL,
        entries: [
          { binding: 0, resource: { buffer: this.sceneBuffer } },
          { binding: 1, resource: { buffer: matBuf } },
          { binding: 2, resource: diffuseTex.createView() },
          { binding: 3, resource: sphereTex.createView() },
          { binding: 4, resource: toonTex.createView() },
          { binding: 5, resource: sampler },
          { binding: 6, resource: this.brdfLut.view },
        ],
      });

      const shadowBG = this.device.createBindGroup({
        label: `pmx-shadow-bg-${mi}`, layout: this.shadowBGLayout,
        entries: [
          { binding: 0, resource: this.shadowMap.view },
          { binding: 1, resource: this.shadowMap.sampler },
          { binding: 2, resource: { buffer: this.shadowMap.getVPBuffer() } },
        ],
      });

      let outlineBG: GPUBindGroup | null = null;
      if (hasEdge) {
        const edgeBuf = this.device.createBuffer({ label: `pmx-edge-${mi}`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const edgeData = new Float32Array(8);
        edgeData[0] = m.edgeColor[0]; edgeData[1] = m.edgeColor[1]; edgeData[2] = m.edgeColor[2]; edgeData[3] = m.edgeColor[3];
        edgeData[4] = m.edgeScale;
        this.device.queue.writeBuffer(edgeBuf, 0, edgeData as unknown as GPUAllowSharedBufferSource);
        outlineBG = this.device.createBindGroup({
          label: `pmx-outline-bg-${mi}`, layout: outlineBGL,
          entries: [
            { binding: 0, resource: { buffer: this.sceneBuffer } },
            { binding: 1, resource: { buffer: edgeBuf } },
            { binding: 2, resource: diffuseTex.createView() },
            { binding: 5, resource: sampler },
          ],
        });
      }

      const castsShadow = (m.flag & 0x04) !== 0;
      this.matRenders.push({ indexOffset, indexCount: matIndexCount, mainBG, shadowBG, outlineBG, isTransparent, hasEdge, renderClass: preset.renderClass, castsShadow });
      indexOffset += matIndexCount;
    }

    const rcRank = (rc: RenderClass) => rc === "eye" ? 1 : rc === "hair" ? 2 : 0;
    this.opaqueOrder = this.matRenders
      .filter(mr => !mr.isTransparent)
      .sort((a, b) => rcRank(a.renderClass) - rcRank(b.renderClass));

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    this.hdrTarget = new HDRRenderTarget(this.device, HDR_FORMAT, "depth24plus-stencil8");
    this.hdrTarget.toneMapping = "filmic";
    this.hdrTarget.resize(w, h);
    this.bloom = new BloomPass(this.device, this.ctx.supportsRG11B10 ? "rg11b10ufloat" as GPUTextureFormat : "rgba16float" as GPUTextureFormat);



    if (pmx.bones.length > 0) {
      const boneDescs: BoneDesc[] = pmx.bones.map((b, i) => {
        let px = b.position[0], py = b.position[1], pz = b.position[2];
        if (b.parentIndex >= 0 && b.parentIndex < pmx.bones.length) {
          const parent = pmx.bones[b.parentIndex];
          px -= parent.position[0];
          py -= parent.position[1];
          pz -= parent.position[2];
        }
        return {
          name: b.name,
          parentIndex: b.parentIndex,
          position: vec3.create(px, py, pz),
          rotation: quat.identity(quat.create()),
          scale: vec3.create(1, 1, 1),
          appendParentIndex: b.appendParentIndex,
          appendRatio: b.appendRatio,
          appendRotate: b.appendRotate,
          appendMove: b.appendMove,
        };
      });
      this.skeleton = new Skeleton(boneDescs);
      const joints = new Uint16Array(vertexCount * 4);
      const weights = new Float32Array(vertexCount * 4);
      for (let i = 0; i < vertexCount; i++) { const v = pmx.vertices[i]; for (let j = 0; j < 4; j++) { joints[i * 4 + j] = v.boneIndices.length > j ? v.boneIndices[j] : 0; weights[i * 4 + j] = v.boneWeights.length > j ? v.boneWeights[j] : 0; } }
      this.skinning = new Skinning(vertexCount, 4, joints, weights, pmx.bones.length);
      this.skeleton.updateWorldMatrices();
      this.skeleton.computeSkinMatrices(this.skinning.skinMatrixData);
      this.device.queue.writeBuffer(this.skinMatrixStorageBuffer, 0, this.skinning.skinMatrixData as unknown as GPUAllowSharedBufferSource);
      this.animPlayer = new AnimationPlayer(this.skeleton, pmx.morphs.length);
      this.ikChains = buildIKChains(pmx.bones);
      console.log(`[PMXDemo] IK chains: ${this.ikChains.length}`);
      for (const c of this.ikChains) {
        const linkNames = c.links.map(l => `${pmx.bones[l.index].name}${l.hasLimit ? `[${l.limitMin[0].toFixed(1)},${l.limitMax[0].toFixed(1)}]x[${l.limitMin[1].toFixed(1)},${l.limitMax[1].toFixed(1)}]y[${l.limitMin[2].toFixed(1)},${l.limitMax[2].toFixed(1)}]z` : ""}`).join(" <- ");
        console.log(`  IK: ${pmx.bones[c.targetIndex].name} -> effector=${pmx.bones[c.effectorIndex].name} iter=${c.iterations} maxAngle=${c.maxAngle.toFixed(3)} links: ${linkNames}`);
      }

      try {
        const vmd = await loadVMD("/motion.vmd");
        const boneNames = pmx.bones.map(b => b.name);
        const morphNames = pmx.morphs.map(m => m.name);
        this.animPlayer.playVMD(vmd, boneNames, morphNames, { loop: true });
        console.log(`[PMXDemo] VMD loaded: "${vmd.name}", duration=${vmdDuration(vmd).toFixed(2)}s, bones=${vmd.boneFrames.size}, morphs=${vmd.morphFrames.size}`);
      } catch (e) {
        console.warn("[PMXDemo] VMD load failed:", e);
      }
    }
  }

  private buildPipelines(): void {
    const vsModule = this.device.createShaderModule({ code: SCENE_VS });
    const fsModule = this.device.createShaderModule({ code: MAIN_FS });
    const shadowVSModule = this.device.createShaderModule({ code: SHADOW_VS });
    const outVSModule = this.device.createShaderModule({ code: OUTLINE_VS });

    vsModule.getCompilationInfo().then(info => {
      for (const msg of info.messages) console.log(`[PMXDemo] VS compile: ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}`);
    });
    fsModule.getCompilationInfo().then(info => {
      for (const msg of info.messages) console.log(`[PMXDemo] FS compile: ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}`);
    });
    const outFSModule = this.device.createShaderModule({ code: OUTLINE_FS });

    const vertexLayout = {
      arrayStride: 56,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" as const },
        { shaderLocation: 1, offset: 12, format: "float32x3" as const },
        { shaderLocation: 2, offset: 24, format: "float32x2" as const },
        { shaderLocation: 3, offset: 32, format: "uint16x4" as const },
        { shaderLocation: 4, offset: 40, format: "float32x4" as const },
      ],
    };

    const blendState = {
      color: { srcFactor: "src-alpha" as const, dstFactor: "one-minus-src-alpha" as const, operation: "add" as const },
      alpha: { srcFactor: "one" as const, dstFactor: "one-minus-src-alpha" as const, operation: "add" as const },
    };

    this.shadowBGLayout = this.device.createBindGroupLayout({
      label: "shadow-bg-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });

    const mainGroup0 = this.device.createBindGroupLayout({
      label: "main-group0",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    this.skinBGL = this.device.createBindGroupLayout({
      label: "skin-bg-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    const DS_FORMAT: GPUTextureFormat = "depth24plus-stencil8";
    const mainLayout = this.device.createPipelineLayout({ bindGroupLayouts: [mainGroup0, this.shadowBGLayout, this.skinBGL] });
    const mainTargets = [{ format: HDR_FORMAT as GPUTextureFormat, blend: blendState }, { format: "rg8unorm" as GPUTextureFormat }];

    this.mainPipeline = this.device.createRenderPipeline({
      label: "pmx-main",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: { format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });

    this.eyePipeline = this.device.createRenderPipeline({
      label: "pmx-eye",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets, constants: { IS_EYE: 1 } },
      primitive: { topology: "triangle-list", cullMode: "back", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less", depthBias: -1, depthBiasSlopeScale: 0.0,
        stencilFront: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilBack: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilReadMask: 0xff, stencilWriteMask: 0xff,
      },
    });

    this.hairPipeline = this.device.createRenderPipeline({
      label: "pmx-hair",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less",
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff, stencilWriteMask: 0,
      },
    });

    this.hairOverEyesPipeline = this.device.createRenderPipeline({
      label: "pmx-hair-over-eyes",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets, constants: { IS_OVER_EYES: 1 } },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: false, depthCompare: "less-equal",
        stencilFront: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff, stencilWriteMask: 0,
      },
    });
    console.log(`[PMXDemo] mainPipeline valid=${this.mainPipeline !== null}`);

    const shadowGroup0 = this.device.createBindGroupLayout({
      label: "shadow-group0",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });

    this.shadowPipeline = this.device.createRenderPipeline({
      label: "pmx-shadow",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [shadowGroup0, this.skinBGL] }),
      vertex: { module: shadowVSModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: { format: this.shadowMap.format, depthWriteEnabled: true, depthCompare: "less", depthBias: 2, depthBiasSlopeScale: 1.5 },
    });

    const outlineGroup0 = this.device.createBindGroupLayout({
      label: "outline-group0",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.outlinePipeline = this.device.createRenderPipeline({
      label: "pmx-outline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [outlineGroup0, this.shadowBGLayout, this.skinBGL] }),
      vertex: { module: outVSModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outFSModule, entryPoint: "fs_main", targets: [{ format: HDR_FORMAT, blend: blendState }, { format: "rg8unorm" }] },
      primitive: { topology: "triangle-list", cullMode: "front", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less-equal", depthBias: 4, depthBiasSlopeScale: 1,
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff, stencilWriteMask: 0,
      },
    });
  }


  update(time: number, deltaTime: number): void {
    if (!this.loaded) return;

    if (this.animPlayer && !this.animPaused) {
      this.animPlayer.update(deltaTime);
      this.skeleton!.updateWorldMatrices();
      if (this.ikChains.length > 0) {
        solveIK(this.skeleton!, this.ikChains);

        this._ikDbgTimer += deltaTime;
        if (this._ikDbgTimer > 2.0) {
          this._ikDbgTimer = 0;
          const wm = this.skeleton!.worldMatrices;
          const sk = this.skeleton!;
          for (const chain of this.ikChains) {
            const tOff = chain.targetIndex * 16;
            const eOff = chain.effectorIndex * 16;
            const dx = wm[tOff+12]-wm[eOff+12], dy = wm[tOff+13]-wm[eOff+13], dz = wm[tOff+14]-wm[eOff+14];
            const dist = Math.sqrt(dx*dx+dy*dy+dz*dz);
            const linkRots = chain.links.map(l => {
              const o4 = l.index * 4;
              const rw = sk.localRotations[o4+3];
              const angle = 2 * Math.acos(Math.min(1, Math.abs(rw))) * 180 / Math.PI;
              return `${sk.boneNames[l.index]}:${angle.toFixed(1)}°`;
            }).join(" ");
            console.log(`[IK] ${sk.boneNames[chain.targetIndex]} dist=${dist.toFixed(3)} [${linkRots}]`);
          }
        }
      }
      this.skeleton!.computeSkinMatrices(this.skinning!.skinMatrixData);

      if (!this._skinDebugOnce2) {
        this._skinDebugOnce2 = true;
        const sk = this.skeleton!;
        const d = this.skinning!.skinMatrixData;
        const kneeIdx = sk.getBoneIndex("右ひざ");
        if (kneeIdx >= 0) {
          const o16 = kneeIdx * 16;
          console.log(`[SKIN-CHECK] 右ひざ skinMatrix row0=[${d[o16].toFixed(4)},${d[o16+1].toFixed(4)},${d[o16+2].toFixed(4)},${d[o16+3].toFixed(4)}]`);
          console.log(`[SKIN-CHECK] 右ひざ skinMatrix row1=[${d[o16+4].toFixed(4)},${d[o16+5].toFixed(4)},${d[o16+6].toFixed(4)},${d[o16+7].toFixed(4)}]`);
          console.log(`[SKIN-CHECK] 右ひざ skinMatrix row2=[${d[o16+8].toFixed(4)},${d[o16+9].toFixed(4)},${d[o16+10].toFixed(4)},${d[o16+11].toFixed(4)}]`);
          console.log(`[SKIN-CHECK] 右ひざ skinMatrix row3=[${d[o16+12].toFixed(4)},${d[o16+13].toFixed(4)},${d[o16+14].toFixed(4)},${d[o16+15].toFixed(4)}]`);
        }
      }
    }

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (w !== this.hdrTarget.w || h !== this.hdrTarget.h) {
      this.hdrTarget.resize(w, h);

    }
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const model = mat4.scaling(vec3.create(1, 1, -1));

    this.sceneData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.sceneData.set(model as unknown as ArrayLike<number>, 16);
    const lx = this.lightX, ly = this.lightY, lz = this.lightZ;
    const len = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
    this.sceneData[32] = lx / len; this.sceneData[33] = ly / len; this.sceneData[34] = lz / len; this.sceneData[35] = 0;
    this.sceneData[36] = 2.0; this.sceneData[37] = 2.0; this.sceneData[38] = 2.0; this.sceneData[39] = 0;
    this.sceneData[40] = this.camera.position[0]; this.sceneData[41] = this.camera.position[1]; this.sceneData[42] = this.camera.position[2]; this.sceneData[43] = 0;
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData as unknown as GPUAllowSharedBufferSource);

    if (!this._skinDebugOnce && this.skinning) {
      this._skinDebugOnce = true;
      const d = this.skinning.skinMatrixData;
      const sk = this.skeleton!;
      for (const name of ["右足", "右ひざ", "右足首", "右つま先"]) {
        const bi = sk.getBoneIndex(name);
        if (bi < 0) { console.log(`[SKIN] ${name}: NOT FOUND`); continue; }
        const o = bi * 16;
        console.log(`[SKIN] ${name} idx=${bi} R00=${d[o].toFixed(4)} R11=${d[o+5].toFixed(4)} T=(${d[o+12].toFixed(3)},${d[o+13].toFixed(3)},${d[o+14].toFixed(3)})`);
      }
    }

    const slx = this.lightX, sly = this.lightY, slz = this.lightZ;
    const slen = Math.sqrt(slx * slx + sly * sly + slz * slz) || 1;
    const sdx = slx / slen, sdy = sly / slen, sdz = slz / slen;
    const shadowTarget = vec3.create(0, 11, 0);
    const shadowEye = vec3.create(shadowTarget[0] + sdx * 72, shadowTarget[1] + sdy * 72, shadowTarget[2] + sdz * 72);
    this.shadowMap.lightPosition = shadowEye;
    this.shadowMap.lightTarget = shadowTarget;
    this.shadowMap.updateLightVP();

    this.shadowSceneData.set(this.shadowMap.lightVP as unknown as ArrayLike<number>, 0);
    this.shadowSceneData.set(model as unknown as ArrayLike<number>, 16);
    this.device.queue.writeBuffer(this.shadowSceneBuffer, 0, this.shadowSceneData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        if (!this.loaded) return;
        const w = this.ctx.canvas.width;
        const h = this.ctx.canvas.height;
        if (this.hdrTarget.w !== w || this.hdrTarget.h !== h) { this.hdrTarget.resize(w, h); }
        if (!this.bloomMaskTex || this.bloomMaskTex.width !== w || this.bloomMaskTex.height !== h) {
          this.bloomMaskTex?.destroy();
          this.bloomMaskTex = this.device.createTexture({ label: "bloom-mask", size: [w, h], format: "rg8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
          this.bloomMaskView = this.bloomMaskTex.createView();
        }

        if (this.skinning) {
          if (!this._gpuSkinDebug) {
            this._gpuSkinDebug = true;
            const d = this.skinning.skinMatrixData;
            const kneeIdx = this.skeleton!.getBoneIndex("右ひざ");
            if (kneeIdx >= 0) {
              const o = kneeIdx * 16;
              console.log(`[GPU-UPLOAD] 右ひざ skinMatrix row0=[${d[o].toFixed(4)},${d[o+1].toFixed(4)},${d[o+2].toFixed(4)},${d[o+3].toFixed(4)}]`);
              console.log(`[GPU-UPLOAD] 右ひざ skinMatrix row3=[${d[o+12].toFixed(4)},${d[o+13].toFixed(4)},${d[o+14].toFixed(4)},${d[o+15].toFixed(4)}]`);
              console.log(`[GPU-UPLOAD] buffer size=${this.skinMatrixStorageBuffer.size} data bytes=${d.byteLength} kneeIdx=${kneeIdx} offset=${o*4}`);
            }
          }
          this.device.queue.writeBuffer(this.skinMatrixStorageBuffer, 0, this.skinning.skinMatrixData as unknown as GPUAllowSharedBufferSource);
        }

        const shadowPass = this.shadowMap.beginShadowPass(encoder);
        shadowPass.setPipeline(this.shadowPipeline);
        shadowPass.setBindGroup(0, this.shadowSceneBG);
        shadowPass.setBindGroup(1, this.skinBG);
        shadowPass.setVertexBuffer(0, this.vertexBuffer);
        shadowPass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");
        for (const mr of this.matRenders) {
          if (!mr.castsShadow) continue;
          shadowPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
        }
        shadowPass.end();

        const mainPass = encoder.beginRenderPass({
          colorAttachments: [
            { view: this.hdrTarget.colorTarget.view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
            { view: this.bloomMaskView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          ],
          depthStencilAttachment: {
            view: this.hdrTarget.depthTarget.view,
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            stencilClearValue: 0,
            stencilLoadOp: "clear",
            stencilStoreOp: "store",
          },
        });
        mainPass.setVertexBuffer(0, this.vertexBuffer);
        mainPass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");
        mainPass.setStencilReference(1);

        for (const mr of this.opaqueOrder) {
          if (mr.hasEdge && mr.outlineBG && mr.renderClass !== "eye") {
            mainPass.setPipeline(this.outlinePipeline);
            mainPass.setBindGroup(0, mr.outlineBG);
            mainPass.setBindGroup(1, mr.shadowBG);
            mainPass.setBindGroup(2, this.skinBG);
            mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
          }
          const pipeline = this.stencilEnabled
            ? (mr.renderClass === "eye" ? this.eyePipeline
              : mr.renderClass === "hair" ? this.hairPipeline
              : this.mainPipeline)
            : this.mainPipeline;
          mainPass.setPipeline(pipeline);
          mainPass.setBindGroup(0, mr.mainBG);
          mainPass.setBindGroup(1, mr.shadowBG);
          mainPass.setBindGroup(2, this.skinBG);
          mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
        }

        for (const mr of this.matRenders) {
          if (!mr.isTransparent) continue;
          if (mr.hasEdge && mr.outlineBG) {
            mainPass.setPipeline(this.outlinePipeline);
            mainPass.setBindGroup(0, mr.outlineBG);
            mainPass.setBindGroup(1, mr.shadowBG);
            mainPass.setBindGroup(2, this.skinBG);
            mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
          }
          mainPass.setPipeline(this.mainPipeline);
          mainPass.setBindGroup(0, mr.mainBG);
          mainPass.setBindGroup(1, mr.shadowBG);
          mainPass.setBindGroup(2, this.skinBG);
          mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
        }
        mainPass.end();

        if (!this.resolvedHDR || this.resolvedHDR.width !== w || this.resolvedHDR.height !== h) {
          this.resolvedHDR?.destroy();
          this.resolvedHDR = this.device.createTexture({ label: "resolved-hdr", size: [w, h], format: HDR_FORMAT, usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING });
          this.resolvedHDRView = this.resolvedHDR.createView();
        }
        encoder.copyTextureToTexture({ texture: this.hdrTarget.colorTarget.texture }, { texture: this.resolvedHDR }, [w, h]);

        if (this.bloomEnabled) {
          const bloomView = this.bloom.execute(encoder, this.resolvedHDR, this.bloomMaskView!);
          this.applyTonemap(encoder, view, this.ctx.format, this.resolvedHDRView!, bloomView, this.bloom.bloomIntensity);
        } else {
          this.applyTonemap(encoder, view, this.ctx.format, this.resolvedHDRView!, null, 0);
        }

        if (this.debugIK && this.ikChains.length > 0) {
          this.drawIKDebug(encoder, view);
        }
      },
    }];
  }

  private _ikDebugPipeline: GPURenderPipeline | null = null;
  private _boneLinePipeline: GPURenderPipeline | null = null;
  private _ikDebugBuf: GPUBuffer | null = null;
  private _boneLineBuf: GPUBuffer | null = null;
  private _ikDebugUBO: GPUBuffer | null = null;

  private drawIKDebug(encoder: GPUCommandEncoder, view: GPUTextureView): void {
    if (!this._ikDebugPipeline) {
      const code = `
struct U { viewProj: mat4x4<f32>, screenSize: vec2<f32>, pad: vec2<f32> };
@group(0) @binding(0) var<uniform> u: U;
struct V { @location(0) pos: vec3<f32>, @location(1) color: vec3<f32>, @location(2) size: f32 };
struct O { @builtin(position) position: vec4<f32>, @location(0) color: vec3<f32> };
@vertex fn vs(v: V) -> O {
  let clip = u.viewProj * vec4<f32>(v.pos, 1.0);
  var o: O;
  o.position = clip;
  o.color = v.color;
  return o;
}
@fragment fn fs(o: O) -> @location(0) vec4<f32> {
  return vec4<f32>(o.color, 1.0);
}`;
      const mod = this.device.createShaderModule({ code });
      const vbLayout: GPUVertexBufferLayout = { arrayStride: 28, attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32" },
      ]};
      const dsState: GPUDepthStencilState = { format: "depth24plus-stencil8", depthWriteEnabled: false, depthCompare: "less" };
      const sharedBGL = this.device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ]});
      const sharedLayout = this.device.createPipelineLayout({ bindGroupLayouts: [sharedBGL] });
      this._ikDebugPipeline = this.device.createRenderPipeline({
        label: "ik-debug-points",
        layout: sharedLayout,
        vertex: { module: mod, entryPoint: "vs", buffers: [vbLayout] },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: this.ctx.format }] },
        primitive: { topology: "point-list" },
        depthStencil: dsState,
      });
      this._boneLinePipeline = this.device.createRenderPipeline({
        label: "ik-debug-lines",
        layout: sharedLayout,
        vertex: { module: mod, entryPoint: "vs", buffers: [vbLayout] },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: this.ctx.format }] },
        primitive: { topology: "line-list" },
        depthStencil: dsState,
      });
    }

    const wm = this.skeleton!.worldMatrices;
    const sk = this.skeleton!;

    // Bone lines: parent→child as cyan lines (Z negated to match model matrix Z-flip)
    const lines: number[] = [];
    for (let i = 0; i < sk.boneCount; i++) {
      const pi = sk.parentIndices[i];
      if (pi < 0) continue;
      const cOff = i * 16, pOff = pi * 16;
      lines.push(wm[pOff+12], wm[pOff+13], -wm[pOff+14], 0, 0.8, 0.8, 1);
      lines.push(wm[cOff+12], wm[cOff+13], -wm[cOff+14], 0, 0.8, 0.8, 1);
    }

    // IK points: target=red, effector=green, chain=yellow
    const pts: number[] = [];
    for (const chain of this.ikChains) {
      const tOff = chain.targetIndex * 16;
      pts.push(wm[tOff+12], wm[tOff+13], -wm[tOff+14], 1, 0, 0, 8);
      const eOff = chain.effectorIndex * 16;
      pts.push(wm[eOff+12], wm[eOff+13], -wm[eOff+14], 0, 1, 0, 8);
      for (const link of chain.links) {
        const lOff = link.index * 16;
        pts.push(wm[lOff+12], wm[lOff+13], -wm[lOff+14], 1, 1, 0, 5);
      }
    }

    const lineVertCount = lines.length / 7;
    const lineByteSize = lineVertCount * 28;
    if (!this._boneLineBuf || this._boneLineBuf.size < lineByteSize) {
      this._boneLineBuf?.destroy();
      this._boneLineBuf = this.device.createBuffer({ label: "bone-line-vb", size: Math.max(lineByteSize, 1024), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this._boneLineBuf, 0, new Float32Array(lines) as unknown as GPUAllowSharedBufferSource);

    const ptVertCount = pts.length / 7;
    const ptByteSize = ptVertCount * 28;
    if (!this._ikDebugBuf || this._ikDebugBuf.size < ptByteSize) {
      this._ikDebugBuf?.destroy();
      this._ikDebugBuf = this.device.createBuffer({ label: "ik-debug-vb", size: Math.max(ptByteSize, 256), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this._ikDebugBuf, 0, new Float32Array(pts) as unknown as GPUAllowSharedBufferSource);

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const uboData = new Float32Array(20);
    uboData.set(viewProj as unknown as ArrayLike<number>, 0);
    uboData[16] = w; uboData[17] = h;

    if (!this._ikDebugUBO) {
      this._ikDebugUBO = this.device.createBuffer({ label: "ik-debug-ubo", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    this.device.queue.writeBuffer(this._ikDebugUBO, 0, uboData as unknown as GPUAllowSharedBufferSource);

    const bg = this.device.createBindGroup({
      layout: this._ikDebugPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this._ikDebugUBO } }],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: this.hdrTarget.depthTarget.view, depthLoadOp: "load", depthStoreOp: "store", depthClearValue: 1.0, stencilLoadOp: "load", stencilStoreOp: "store", stencilClearValue: 0 },
    });
    // Draw bone lines
    pass.setPipeline(this._boneLinePipeline!);
    pass.setBindGroup(0, bg);
    pass.setVertexBuffer(0, this._boneLineBuf);
    pass.draw(lineVertCount);
    // Draw IK points
    pass.setPipeline(this._ikDebugPipeline);
    pass.setVertexBuffer(0, this._ikDebugBuf);
    pass.draw(ptVertCount);
    pass.end();
  }

  exposure = 1.0;
  gamma = 2.2;
  slope = 1.0;
  offset = 0.0;
  power = 1.0;
  saturation = 1.0;
  contrast = 1.0;
  private filmicLUT: GPUTexture | null = null;
  private filmicLUTView: GPUTextureView | null = null;
  private toneBGSceneView: GPUTextureView | null = null;
  private toneBGBloomView: GPUTextureView | null = null;
  private blackTex: GPUTexture | null = null;
  private blackTexView: GPUTextureView | null = null;
  private bloomParamsUBO!: GPUBuffer;

  private buildFilmicLUT(): GPUTexture {
    if (this.filmicLUT) return this.filmicLUT;
    const LUT_W = 256;
    const data = new Float32Array(LUT_W * 4);
    const A = 0.22, B = 0.30, C = 0.10, D = 0.20, E = 0.01, F = 0.30;
    const filmicWhite = ((11.2 * (A * 11.2 + C * B) + D * E) / (11.2 * (A * 11.2 + B) + D * F)) - E / F;
    const whiteScale = 1.0 / filmicWhite;
    for (let i = 0; i < LUT_W; i++) {
      const logX = (i / (LUT_W - 1)) * 13.0 - 10.0;
      const x = Math.pow(2, logX);
      const filmic = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
      const v = Math.max(0, filmic * whiteScale);
      data[i * 4 + 0] = v;
      data[i * 4 + 1] = v;
      data[i * 4 + 2] = v;
      data[i * 4 + 3] = 1.0;
    }
    this.filmicLUT = this.device.createTexture({
      label: "filmic-lut",
      size: [LUT_W, 1],
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture({ texture: this.filmicLUT }, data as unknown as GPUAllowSharedBufferSource, { bytesPerRow: LUT_W * 16 }, [LUT_W, 1]);
    this.filmicLUTView = this.filmicLUT.createView();
    return this.filmicLUT;
  }

  private applyTonemap(encoder: GPUCommandEncoder, screenView: GPUTextureView, screenFormat: GPUTextureFormat, sceneView: GPUTextureView, bloomView: GPUTextureView | null, bloomIntensity: number): void {
    if (!this.tonePipeline) {
      const lut = this.buildFilmicLUT();
      const code = `
struct Params { exposure: f32, gamma: f32, contrast: f32, flags: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSampler: sampler;
@group(0) @binding(3) var filmicLut: texture_2d<f32>;
@group(0) @binding(4) var<uniform> grade: vec4<f32>;
@group(0) @binding(5) var<uniform> grade2: vec4<f32>;
@group(0) @binding(6) var bloomTex: texture_2d<f32>;
@group(0) @binding(7) var<uniform> bloomParams: vec4<f32>;

fn filmicLUT(x: f32) -> f32 {
  let t = clamp(log2(max(x, 1e-10)) + 10.0, 0.0, 13.0);
  let idx = u32(t * 255.0 / 13.0 + 0.5);
  return textureLoad(filmicLut, vec2u(min(idx, 255u), 0u), 0).r;
}

fn gradeColor(c: vec3f) -> vec3f {
  let slope = grade.xyz;
  let offset = vec3f(grade.w);
  let power = grade2.xyz;
  let sat = grade2.w;
  var x = pow(max(c * slope + offset, vec3f(0.0)), power);
  let luma = dot(x, vec3f(0.2126, 0.7152, 0.0722));
  return max(mix(vec3f(luma), x, sat), vec3f(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3));
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(srcTex));
  let uv = pos.xy / dims;
  let scene = textureSample(srcTex, srcSampler, uv).rgb;
  let bloom = textureSample(bloomTex, srcSampler, uv).rgb;
  var color = (scene + bloom * bloomParams.x) * p.exposure;
  let doTonemap = (p.flags & 1u) != 0u;
  let doGrade = (p.flags & 2u) != 0u;
  if (doTonemap) {
    color = vec3f(filmicLUT(color.r), filmicLUT(color.g), filmicLUT(color.b));
  } else {
    color = clamp(color, vec3f(0.0), vec3f(1.0));
  }
  if (doGrade) {
    color = gradeColor(color);
    color = (color - vec3f(0.5)) * p.contrast + vec3f(0.5);
  }
  color = pow(max(color, vec3f(0.0)), vec3f(1.0 / p.gamma));
  return vec4<f32>(color, 1.0);
}`;
      const module = this.device.createShaderModule({ code });
      this.tonePipeline = this.device.createRenderPipeline({
        label: "pmx-tonemap",
        layout: "auto",
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint: "fs_main", targets: [{ format: screenFormat }] },
        primitive: { topology: "triangle-list" },
      });
    }

    if (!this.toneUBO) {
      this.toneUBO = this.device.createBuffer({ label: "tone-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!this.gradeUBO) {
      this.gradeUBO = this.device.createBuffer({ label: "grade-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.grade2UBO = this.device.createBuffer({ label: "grade2-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!this.bloomParamsUBO) {
      this.bloomParamsUBO = this.device.createBuffer({ label: "bloom-params-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!this.blackTex) {
      this.blackTex = this.device.createTexture({ label: "tonemap-black", size: [1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.device.queue.writeTexture({ texture: this.blackTex }, new Uint8Array([0, 0, 0, 0]), { bytesPerRow: 4 }, [1, 1]);
      this.blackTexView = this.blackTex.createView();
    }
    const flags = (this.tonemapEnabled ? 1 : 0) | (this.gradeEnabled ? 2 : 0);
    const data = new ArrayBuffer(16);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);
    f32[0] = this.exposure; f32[1] = this.gamma; f32[2] = this.contrast; u32[3] = flags;
    this.device.queue.writeBuffer(this.toneUBO, 0, data);
    const gd = new Float32Array([this.slope, this.slope, this.slope, this.offset]);
    this.device.queue.writeBuffer(this.gradeUBO, 0, gd as unknown as GPUAllowSharedBufferSource);
    const gd2 = new Float32Array([this.power, this.power, this.power, this.saturation]);
    this.device.queue.writeBuffer(this.grade2UBO, 0, gd2 as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.bloomParamsUBO, 0, new Float32Array([bloomIntensity, 0, 0, 0]) as unknown as GPUAllowSharedBufferSource);

    if (!this.toneSampler) {
      this.toneSampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear" });
    }
    const effectiveBloomView = bloomView ?? this.blackTexView!;
    if (!this.toneBindGroup || sceneView !== this.toneBGSceneView || effectiveBloomView !== this.toneBGBloomView) {
      this.toneBindGroup = this.device.createBindGroup({
        layout: this.tonePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.toneUBO } },
          { binding: 1, resource: sceneView },
          { binding: 2, resource: this.toneSampler },
          { binding: 3, resource: this.filmicLUTView! },
          { binding: 4, resource: { buffer: this.gradeUBO } },
          { binding: 5, resource: { buffer: this.grade2UBO } },
          { binding: 6, resource: effectiveBloomView },
          { binding: 7, resource: { buffer: this.bloomParamsUBO } },
        ],
      });
      this.toneBGSceneView = sceneView;
      this.toneBGBloomView = effectiveBloomView;
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: screenView, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(this.tonePipeline);
    pass.setBindGroup(0, this.toneBindGroup);
    pass.draw(3);
    pass.end();
  }


  private setShadowResolution(size: number): void {
    if (size === this.shadowMap.size) return;
    this.shadowMap.resize(size);
    for (const mr of this.matRenders) {
      mr.shadowBG = this.device.createBindGroup({
        layout: this.shadowBGLayout,
        entries: [
          { binding: 0, resource: this.shadowMap.view },
          { binding: 1, resource: this.shadowMap.sampler },
          { binding: 2, resource: { buffer: this.shadowMap.getVPBuffer() } },
        ],
      });
    }
  }

  registerGUI(gui: any) {
    gui.add(this, "animPaused").name("Pause Animation");
    gui.add(this, "debugIK").name("Debug Skeleton");
    gui.add(this, "bloomEnabled").name("Bloom");
    gui.add(this, "tonemapEnabled").name("Tone Mapping");
    gui.add(this, "gradeEnabled").name("Color Grading");
    gui.add(this, "stencilEnabled").name("Eye Stencil");
    gui.add(this, "shadowRes", [1024, 2048, 4096]).name("Shadow Res").onChange((v: number) => this.setShadowResolution(v));
    const camFolder = gui.addFolder("Camera");
    camFolder.add(this.camera, "distance", 1, 80, 0.5).name("Distance");
    camFolder.add(this.camera, "fov", 20, 120, 1).name("FOV");
    camFolder.add(this.camera.target, "0", -20, 20, 0.1).name("Target X");
    camFolder.add(this.camera.target, "1", -5, 30, 0.1).name("Target Y");
    camFolder.add(this.camera.target, "2", -20, 20, 0.1).name("Target Z");
    gui.add(this.bloom, "bloomIntensity", 0, 1, 0.01).name("Bloom Intensity");
    gui.add(this.bloom, "threshold", 0, 2, 0.01).name("Bloom Threshold");
    gui.add(this.bloom, "knee", 0, 1, 0.01).name("Bloom Knee");
    gui.add(this.bloom, "radius", 0.5, 10, 0.1).name("Bloom Radius");
    gui.add(this.bloom, "maxMips", [2, 3, 4, 5]).name("Bloom Mips");
    const toneFolder = gui.addFolder("Tone Mapping");
    toneFolder.add(this, "exposure", 0.1, 3, 0.01).name("Exposure");
    toneFolder.add(this, "gamma", 1.0, 3.0, 0.01).name("Gamma");
    const gradeFolder = gui.addFolder("Color Grading");
    gradeFolder.add(this, "slope", 0.5, 2.0, 0.01).name("Slope");
    gradeFolder.add(this, "offset", -0.5, 0.5, 0.01).name("Offset");
    gradeFolder.add(this, "power", 0.5, 2.0, 0.01).name("Power");
    gradeFolder.add(this, "saturation", 0, 2, 0.01).name("Saturation");
    gradeFolder.add(this, "contrast", 0.5, 2.0, 0.01).name("Contrast");
    const lightFolder = gui.addFolder("Light");
    lightFolder.add(this, "lightX", -30, 30, 0.5).name("X");
    lightFolder.add(this, "lightY", -30, 30, 0.5).name("Y");
    lightFolder.add(this, "lightZ", -30, 30, 0.5).name("Z");
  }

  destroy(): void {
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.sceneBuffer?.destroy();
    this.shadowSceneBuffer?.destroy();
    this.skinMatrixStorageBuffer?.destroy();
    this.skinning?.destroy();
    this.shadowMap?.destroy();
    this.hdrTarget?.destroy();
    this.bloom?.destroy();
    this.resolvedHDR?.destroy();
    this.bloomMaskTex?.destroy();
    this.bloomOutput?.destroy();
    this.toneUBO?.destroy();
    this.gradeUBO?.destroy();
    this.grade2UBO?.destroy();
    this.filmicLUT?.destroy();
    this._depthTex?.destroy();

    for (const t of this.gpuTextures) t.destroy();
    this.gpuTextures = [];
    this.matRenders = [];
  }
}
