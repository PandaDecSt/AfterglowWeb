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

const HDR_FORMAT = "rgba16float";

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

@vertex
fn vs_main(in: VSIn) -> VSOut {
  var out: VSOut;

  var skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var skinNrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  let pos4 = vec4<f32>(in.position, 1.0);
  let nrm4 = vec4<f32>(in.normal, 0.0);
  for (var i = 0u; i < 4u; i++) {
    let j = in.joints[i];
    let w = in.weights[i];
    skinPos += skinMatrices[j] * pos4 * w;
    skinNrm += skinMatrices[j] * nrm4 * w;
  }

  let worldPos = (scene.model * skinPos).xyz;
  out.position = scene.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldNormal = normalize((scene.model * skinNrm).xyz);
  out.worldPos = worldPos;
  out.uv = in.uv;
  return out;
}
`;

const MAIN_FS = `
struct Scene {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> scene: Scene;

struct Mat {
  diffuseColor: vec3<f32>,
  alpha: f32,
  ambient: vec3<f32>,
  shininess: f32,
  specular: vec3<f32>,
  sphereMode: f32,
  _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(1) var<uniform> mat: Mat;

@group(0) @binding(2) var diffuseTex: texture_2d<f32>;
@group(0) @binding(3) var sphereTex: texture_2d<f32>;
@group(0) @binding(4) var toonTex: texture_2d<f32>;
@group(0) @binding(5) var texSampler: sampler;

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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let tex_s = textureSample(diffuseTex, texSampler, in.uv);
  let alpha = mat.alpha * tex_s.a;
  if (alpha < 0.001) { discard; }

  var n = normalize(in.worldNormal);
  let v = normalize(scene.cameraPos.xyz - in.worldPos);
  n = select(-n, n, dot(n, v) >= 0.0);

  let l = normalize(scene.lightDir.xyz);
  let h = normalize(v + l);
  let NL = max(dot(n, l), 0.0);
  let NH = max(dot(n, h), 0.0);

  let baseColor = tex_s.rgb * mat.diffuseColor;
  let shadow = sampleShadowPCF(shadowTex, shadowSampler, lightVP, in.worldPos, 0.002);

  let toonT = clamp(NL * 0.5 + 0.5, 0.0, 1.0);
  let toonShade = textureSample(toonTex, texSampler, vec2<f32>(toonT, 0.5)).r;

  let ambient = baseColor * mat.ambient;
  let diffuse = baseColor * toonShade * shadow;
  let spec = pow(NH, max(mat.shininess, 1.0)) * mat.specular * shadow;
  let specular = spec * scene.lightColor.rgb;

  var sphereAdd = vec3<f32>(0.0);
  if (mat.sphereMode > 0.5 && mat.sphereMode < 1.5) {
    let viewN = (scene.viewProj * vec4<f32>(n, 0.0)).xy;
    let sphereUV = viewN * 0.5 + vec2<f32>(0.5, 0.5);
    sphereAdd = textureSample(sphereTex, texSampler, sphereUV).rgb * baseColor;
  } else if (mat.sphereMode > 1.5) {
    sphereAdd = textureSample(sphereTex, texSampler, in.uv).rgb * baseColor;
  }

  let color = ambient + diffuse * scene.lightColor.rgb + specular + sphereAdd;
  return vec4<f32>(color, alpha);
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

@group(1) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let texA = textureSample(diffuseTex, texSampler, in.uv).a;
  if (texA < 0.05) { discard; }
  return vec4<f32>(omat.edgeColor.rgb, omat.edgeColor.a * texA);
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
  private bloomMips: GPUTexture[] = [];
  private resolvedHDR: GPUTexture | null = null;
  private bloomOutput: GPUTexture | null = null;
  private bloomOutputView: GPUTextureView | null = null;
  private tonePipeline: GPURenderPipeline | null = null;
  private toneBindGroup: GPUBindGroup | null = null;
  private toneUBO!: GPUBuffer;

  private matRenders: MatRenderData[] = [];
  private gpuTextures: GPUTexture[] = [];

  private skeleton: Skeleton | null = null;
  private skinning: Skinning | null = null;
  private animPlayer: AnimationPlayer | null = null;

  private skinMatrixStorageBuffer!: GPUBuffer;
  private skinBGL!: GPUBindGroupLayout;
  private skinBG!: GPUBindGroup;

  private shadowSceneBuffer!: GPUBuffer;
  private shadowSceneData = new Float32Array(32);
  private shadowSceneBG!: GPUBindGroup;

  bloomEnabled = true;

  private loaded = false;
  private _renderLogOnce = true;
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
    }

    const center = [(minPos[0] + maxPos[0]) / 2, (minPos[1] + maxPos[1]) / 2, (minPos[2] + maxPos[2]) / 2];
    const extent = [maxPos[0] - minPos[0], maxPos[1] - minPos[1], maxPos[2] - minPos[2]];
    const radius = Math.sqrt(extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2) / 2;
    this.camera.orbit(vec3.create(center[0], center[1], center[2]), radius * 2.5, radius * 0.01, radius * 20);
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

    this.shadowMap = new ShadowMap(this.device, 4096);
    this.shadowMap.orthoSize = radius * 3;
    this.shadowMap.far = radius * 6;

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

      const matBuf = this.device.createBuffer({ label: `pmx-mat-${mi}`, size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const matData = new Float32Array(16);
      matData[0] = m.diffuse[0]; matData[1] = m.diffuse[1]; matData[2] = m.diffuse[2]; matData[3] = m.diffuse[3];
      matData[4] = m.ambient[0]; matData[5] = m.ambient[1]; matData[6] = m.ambient[2]; matData[7] = m.specularPower;
      matData[8] = m.specular[0]; matData[9] = m.specular[1]; matData[10] = m.specular[2]; matData[11] = m.sphereMode;
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

      this.matRenders.push({ indexOffset, indexCount: matIndexCount, mainBG, shadowBG, outlineBG, isTransparent, hasEdge });
      indexOffset += matIndexCount;
    }

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    this.hdrTarget = new HDRRenderTarget(this.device, HDR_FORMAT);
    this.hdrTarget.toneMapping = "filmic";
    this.hdrTarget.resize(w, h);
    this.bloom = new BloomPass(this.device, HDR_FORMAT);
    this.ensureBloomMips(w, h);

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
      ],
    });

    this.skinBGL = this.device.createBindGroupLayout({
      label: "skin-bg-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.mainPipeline = this.device.createRenderPipeline({
      label: "pmx-main",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [mainGroup0, this.shadowBGLayout, this.skinBGL] }),
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: [{ format: HDR_FORMAT, blend: blendState }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
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
      primitive: { topology: "triangle-list", cullMode: "none" },
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
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [outlineGroup0, this.skinBGL] }),
      vertex: { module: outVSModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outFSModule, entryPoint: "fs_main", targets: [{ format: HDR_FORMAT, blend: blendState }] },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less-equal", depthBias: 4, depthBiasSlopeScale: 1 },
    });
  }

  private ensureBloomMips(w: number, h: number): void {
    const mipCount = 5;
    for (let i = 0; i < mipCount; i++) {
      const mw = Math.max(1, Math.floor(w / (2 ** (i + 1))));
      const mh = Math.max(1, Math.floor(h / (2 ** (i + 1))));
      const existing = this.bloomMips[i];
      if (existing && existing.width === mw && existing.height === mh) continue;
      existing?.destroy();
      this.bloomMips[i] = this.device.createTexture({ label: `bloom-mip-${i}`, size: [mw, mh], format: HDR_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    }
  }

  update(time: number, deltaTime: number): void {
    if (!this.loaded) return;

    if (this.animPlayer) {
      this.animPlayer.update(deltaTime);
      this.skeleton!.updateWorldMatrices();
      this.skeleton!.computeSkinMatrices(this.skinning!.skinMatrixData);
      this.device.queue.writeBuffer(
        this.skinMatrixStorageBuffer, 0,
        this.skinning!.skinMatrixData as unknown as GPUAllowSharedBufferSource
      );
    }

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (w !== this.hdrTarget.w || h !== this.hdrTarget.h) {
      this.hdrTarget.resize(w, h);
      this.ensureBloomMips(w, h);
    }
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const model = mat4.identity(mat4.create());

    this.sceneData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.sceneData.set(model as unknown as ArrayLike<number>, 16);
    this.sceneData[32] = 0.5; this.sceneData[33] = 1.0; this.sceneData[34] = 0.8; this.sceneData[35] = 0;
    this.sceneData[36] = 2.0; this.sceneData[37] = 2.0; this.sceneData[38] = 2.0; this.sceneData[39] = 0;
    this.sceneData[40] = this.camera.position[0]; this.sceneData[41] = this.camera.position[1]; this.sceneData[42] = this.camera.position[2]; this.sceneData[43] = 0;
    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData as unknown as GPUAllowSharedBufferSource);

    if ((time as unknown as number) < 0.001) {
      const vp = this.sceneData.slice(0, 16);
      const cam = this.sceneData.slice(40, 44);
      console.log(`[PMXDemo] VP row0=[${Array.from(vp.slice(0, 4)).map(v => v.toFixed(3))}]`);
      console.log(`[PMXDemo] Camera pos=[${cam[0].toFixed(2)},${cam[1].toFixed(2)},${cam[2].toFixed(2)}]`);
      if (this.skinning) {
        const d = this.skinning.skinMatrixData;
        console.log(`[PMXDemo] skin[0]: ${d[0]} ${d[1]} ${d[2]} ${d[3]} | ${d[4]} ${d[5]} ${d[6]} ${d[7]} | ${d[8]} ${d[9]} ${d[10]} ${d[11]} | ${d[12]} ${d[13]} ${d[14]} ${d[15]}`);
        console.log(`[PMXDemo] skin[1]: ${d[16]} ${d[17]} ${d[18]} ${d[19]} | ${d[20]} ${d[21]} ${d[22]} ${d[23]} | ${d[24]} ${d[25]} ${d[26]} ${d[27]} | ${d[28]} ${d[29]} ${d[30]} ${d[31]}`);
      }
    }

    this.shadowMap.lightPosition = vec3.create(5, 10, 8);
    this.shadowMap.lightTarget = vec3.create(0, 10, 0);
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
        if (this.hdrTarget.w !== w || this.hdrTarget.h !== h) { this.hdrTarget.resize(w, h); this.ensureBloomMips(w, h); }

        const shadowPass = this.shadowMap.beginShadowPass(encoder);
        shadowPass.setPipeline(this.shadowPipeline);
        shadowPass.setBindGroup(0, this.shadowSceneBG);
        shadowPass.setBindGroup(1, this.skinBG);
        shadowPass.setVertexBuffer(0, this.vertexBuffer);
        shadowPass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");
        shadowPass.drawIndexed(this.totalIndexCount);
        shadowPass.end();

        const mainPass = this.hdrTarget.beginRenderPass(encoder);
        mainPass.setVertexBuffer(0, this.vertexBuffer);
        mainPass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");

        for (const mr of this.matRenders) {
          if (mr.isTransparent) continue;
          if (mr.hasEdge && mr.outlineBG) {
            mainPass.setPipeline(this.outlinePipeline);
            mainPass.setBindGroup(0, mr.outlineBG);
            mainPass.setBindGroup(1, this.skinBG);
            mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
          }
          mainPass.setPipeline(this.mainPipeline);
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
            mainPass.setBindGroup(1, this.skinBG);
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
        }
        if (!this.bloomOutput || this.bloomOutput.width !== w || this.bloomOutput.height !== h) {
          this.bloomOutput?.destroy();
          this.bloomOutput = this.device.createTexture({ label: "bloom-output", size: [w, h], format: HDR_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
          this.bloomOutputView = this.bloomOutput.createView();
          this.tonePipeline = null;
          this.toneBindGroup = null;
        }
        encoder.copyTextureToTexture({ texture: this.hdrTarget.colorTarget.texture }, { texture: this.resolvedHDR }, [w, h]);

        if (this.bloomEnabled) {
          this.bloom.execute(encoder, this.resolvedHDR, this.bloomMips, this.bloomOutputView!);
          this.applyTonemap(encoder, view, this.ctx.format, this.bloomOutputView!);
        } else {
          const hdrView = this.resolvedHDR.createView();
          this.applyTonemap(encoder, view, this.ctx.format, hdrView);
        }
      },
    }];
  }

  private applyTonemap(encoder: GPUCommandEncoder, screenView: GPUTextureView, screenFormat: GPUTextureFormat, srcView: GPUTextureView): void {
    if (!this.tonePipeline) {
      const code = `
struct Params { exposure: f32, gamma: f32, width: f32, height: f32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSampler: sampler;

fn filmic(x: vec3<f32>) -> vec3<f32> {
  let A = 0.22; let B = 0.30; let C = 0.10; let D = 0.20; let E = 0.01; let F = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3));
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / vec2<f32>(p.width, p.height);
  var color = textureSample(srcTex, srcSampler, uv).rgb * p.exposure;
  color = filmic(color);
  color = pow(color, vec3<f32>(1.0 / p.gamma));
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
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const data = new Float32Array([1.0, 2.2, w, h]);
    this.device.queue.writeBuffer(this.toneUBO, 0, data as unknown as GPUAllowSharedBufferSource);

    const bg = this.device.createBindGroup({
      layout: this.tonePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.toneUBO } },
        { binding: 1, resource: srcView },
        { binding: 2, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
      ],
    });

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: screenView, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(this.tonePipeline);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }

  registerGUI(gui: any) {
    gui.add(this, "bloomEnabled").name("Bloom");
    gui.add(this.bloom, "bloomIntensity", 0, 2, 0.01).name("Bloom Intensity");
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
    this.bloomOutput?.destroy();
    this.toneUBO?.destroy();
    this._depthTex?.destroy();
    for (const t of this.bloomMips) t?.destroy();
    for (const t of this.gpuTextures) t.destroy();
    this.gpuTextures = [];
    this.matRenders = [];
  }
}
