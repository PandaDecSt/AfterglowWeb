import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { mat4 } from "wgpu-matrix";
import { loadGLTF, interleaveMesh, type MeshData, type MaterialData } from "../utils/gltf-loader";

// === Material Classification ===
type MaterialCategory = "body" | "face" | "hair" | "eye" | "eyelash" | "other";

function classifyMaterial(name: string): MaterialCategory {
  const n = name.toLowerCase();
  if (/face|脸|面部/.test(n)) return "face";
  if (/hair|头发|发/.test(n)) return "hair";
  if (/eyelash|睫毛/.test(n)) return "eyelash";
  if (/eye|iris|眼|瞳孔|schlera/.test(n)) return "eye";
  if (/body|dress|tights|nude|neck|hat|衣|身|裙|帽|颈/.test(n)) return "body";
  return "other";
}

// Render modes: 0=PBR, 1=ToonBody, 2=ToonFace, 3=ToonHair, 4=ToonEye, 5=ToonEyelash
const RENDER_MODE_NAMES = ["PBR", "Toon Body", "Toon Face", "Toon Hair", "Toon Eye", "Toon Eyelash", "Normal View"];

// === Preset System ===
interface RenderPreset {
  name: string;
  mapping: Record<MaterialCategory, number>;
}

const PRESETS_KEY = "afterglow-glb-presets";

function loadPresets(): RenderPreset[] {
  try {
    const raw = localStorage.getItem(PRESETS_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return [defaultPreset()];
}

function savePresets(presets: RenderPreset[]) {
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
}

function defaultPreset(): RenderPreset {
  return {
    name: "Endfield Default",
    mapping: { body: 1, face: 2, hair: 3, eye: 4, eyelash: 5, other: 1 },
  };
}

// === Shaders ===
const modelShader = `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
};

struct MatUniforms {
  baseColorFactor: vec4<f32>,
  params: vec4<f32>,  // metallic, roughness, hasBaseColorTex, lightmapStrength
  params2: vec4<f32>, // alphaCutoff, renderMode, outlineWidth, pad
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> mat: MatUniforms;
@group(1) @binding(1) var baseColorTex: texture_2d<f32>;
@group(1) @binding(2) var lightmapTex: texture_2d<f32>;
@group(1) @binding(3) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = frame.model * vec4<f32>(pos, 1.0);
  out.position = frame.viewProj * worldPos;
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((frame.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  return out;
}

const PI: f32 = 3.14159265359;

fn distributionGGX(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
  let a2 = roughness * roughness * roughness * roughness;
  let NdotH = max(dot(N, H), 0.0);
  let d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

fn geometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  let nov = max(dot(N, V), 0.0);
  let nol = max(dot(N, L), 0.0);
  return (nov / (nov * (1.0 - k) + k)) * (nol / (nol * (1.0 - k) + k));
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn endfieldRimLighting(nov: f32, vol: f32, nol: f32, width: f32) -> f32 {
  let rimStepMin = clamp(0.9 - width, 0.0, 0.99);
  let rimStepMax = clamp(1.0 - width, 0.01, 1.0);
  var rim = smoothstep(rimStepMin, rimStepMax, 1.0 - nov);
  rim *= max(vol, 0.0) * max(nol + 0.5, 0.0) * 2.0;
  return rim;
}

@fragment
fn fs_main(in: VSOut, @builtin(front_facing) isFront: bool) -> @location(0) vec4<f32> {
  var N = normalize(in.worldNormal);
  if (!isFront) { N = -N; }
  let V = normalize(frame.cameraPosition.xyz - in.worldPos);
  let L = normalize(-frame.lightDir.xyz);
  let H = normalize(V + L);

  var baseColor: vec3<f32>;
  var texAlpha = 1.0;
  if (mat.params.z > 0.5) {
    let texColor = textureSample(baseColorTex, texSampler, in.uv);
    baseColor = texColor.rgb * mat.baseColorFactor.rgb;
    texAlpha = texColor.a * mat.baseColorFactor.a;
  } else {
    baseColor = mat.baseColorFactor.rgb;
  }

  if (mat.params2.x > 0.001 && texAlpha < mat.params2.x) {
    discard;
  }

  let lightmapStrength = mat.params.w;
  var lightmap = vec3<f32>(1.0);
  if (lightmapStrength > 0.001) {
    let lm = textureSample(lightmapTex, texSampler, in.uv);
    lightmap = mix(vec3<f32>(1.0), lm.rgb, lightmapStrength);
  }

  let NdotL = dot(N, L);
  let nov = dot(N, V);
  let vol = dot(-V, L);
  let nol = NdotL;
  var color: vec3<f32>;
  let mode = i32(mat.params2.y + 0.5);

  if (mode == 0) {
    // === PBR ===
    let metallic = mat.params.x;
    let roughness = max(mat.params.y, 0.04);
    let F0 = mix(vec3<f32>(0.04), baseColor, metallic);
    let NDF = distributionGGX(N, H, roughness);
    let G = geometrySmith(N, V, L, roughness);
    let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
    let spec = (NDF * G * F) / (4.0 * max(nov, 0.0) * max(NdotL, 0.0) + 0.0001);
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let Lo = (kD * baseColor / PI + spec) * frame.lightColor.rgb * max(NdotL, 0.0);
    let hemi = mix(vec3<f32>(0.2, 0.15, 0.1), vec3<f32>(0.5, 0.6, 0.8), N.y * 0.5 + 0.5);
    let ambient = hemi * baseColor * 0.35 * lightmap;
    color = acesTonemap((ambient + Lo) * lightmap * 1.3);
    color = pow(color, vec3<f32>(1.0 / 2.2));

  } else if (mode == 1) {
    // === Toon Body (Endfield-style) ===
    // viewAttenuation: darken at grazing angles for volume feel
    // clamp viewFactor to 0.6 min to prevent washout on inconsistent normals
    let viewAttenuation = 0.5;
    var radiance = max(NdotL, 0.0);
    let viewFactor = max(mix(1.0, max(nov, 0.0), viewAttenuation), 0.6);
    radiance *= viewFactor;
    let fadedSSS = 0.4 * (0.5 + viewFactor * 0.5);

    let rampOffset = 0.2;
    let rampCoord = clamp((1.0 - rampOffset) - radiance * (0.5 - rampOffset * 0.5), 0.1, 0.9);
    let rampWarm = vec3<f32>(1.0, 0.6, 0.4);
    let rampCool = vec3<f32>(0.7, 0.75, 0.9);
    let rampColor = mix(rampCool, rampWarm, smoothstep(0.3, 0.7, rampCoord));
    let rampAlpha = smoothstep(0.2, 0.8, rampCoord) * 0.6;

    let finalToon = baseColor * (radiance + (rampColor * rampAlpha) * min(vec3<f32>(fadedSSS), baseColor));
    let rim = endfieldRimLighting(nov, vol, nol, 0.12);
    color = (finalToon * lightmap + rim * baseColor * 0.25) * frame.lightColor.rgb * frame.lightColor.w * 0.5;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

  } else if (mode == 2) {
    // === Toon Face (hard shadow, SDF-like) ===
    // Simulate SDF face shadow: hard boundary based on light angle
    let faceRadiance = smoothstep(0.0, 0.1, NdotL);
    let viewFactor = mix(1.0, max(nov, 0.0), 0.3);
    let finalFace = baseColor * (0.3 + faceRadiance * 0.7) * viewFactor;

    // Subtle warm shadow tint
    let shadowTint = baseColor * vec3<f32>(0.9, 0.7, 0.65);
    let faceResult = mix(shadowTint * 0.5, finalFace, faceRadiance);

    let rim = endfieldRimLighting(nov, vol, nol, 0.08);
    color = (faceResult * lightmap + rim * 0.2) * frame.lightColor.rgb * frame.lightColor.w * 0.5;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

  } else if (mode == 3) {
    // === Toon Hair (anisotropic-like specular) ===
    let radiance = max(NdotL, 0.0);
    let viewFactor = mix(1.0, max(nov, 0.0), 0.4);

    // Anisotropic-like: use shifted half vector along tangent approximation
    // Approximate tangent from UV gradient (use worldNormal cross up)
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let tangent = normalize(cross(up, N) + vec3<f32>(1e-6));
    let TdotH = dot(tangent, H);
    let anisoSpec = pow(sqrt(max(1.0 - TdotH * TdotH, 0.0)), 16.0);
    let specSolid = smoothstep(0.3, 0.5, anisoSpec) * 0.6;

    // Hard diffuse
    let hairDiffuse = smoothstep(0.0, 0.15, radiance);
    let hairColor = baseColor * (0.25 + hairDiffuse * 0.75) * viewFactor;

    let rim = endfieldRimLighting(nov, vol, nol, 0.1);
    color = (hairColor * lightmap + specSolid * frame.lightColor.rgb + rim * 0.25) * frame.lightColor.rgb * frame.lightColor.w * 0.5;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

  } else if (mode == 4) {
    // === Toon Eye/Iris ===
    // Simple: high contrast, sharp specular, dark
    let eyeDiffuse = smoothstep(0.0, 0.05, NdotL);
    let eyeColor = baseColor * (0.4 + eyeDiffuse * 0.6);

    // Sharp specular highlight
    let NdotH = max(dot(N, H), 0.0);
    let eyeSpec = smoothstep(0.85, 0.9, NdotH) * 0.8;

    color = (eyeColor * lightmap + eyeSpec * frame.lightColor.rgb) * frame.lightColor.rgb * frame.lightColor.w * 0.5;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

  } else if (mode == 5) {
    // === Toon Eyelash ===
    // Minimal shading, mostly flat with slight NdotL
    let lashDiffuse = 0.5 + max(NdotL, 0.0) * 0.5;
    color = baseColor * lashDiffuse * lightmap * frame.lightColor.rgb * frame.lightColor.w * 0.5;
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));

  } else {
    // === Normal View (debug) ===
    color = N * 0.5 + 0.5;
  }

  return vec4<f32>(color, 1.0);
}
`;

const outlineShader = `
struct FrameUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
};

struct MatUniforms {
  baseColorFactor: vec4<f32>,
  params: vec4<f32>,
  params2: vec4<f32>, // alphaCutoff, renderMode, outlineWidth, pad
};

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(1) @binding(0) var<uniform> mat: MatUniforms;

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> @builtin(position) vec4<f32> {
  let outlineWidth = mat.params2.z;
  let worldPos = (frame.model * vec4<f32>(pos, 1.0)).xyz;
  let worldNormal = normalize((frame.invTransModel * vec4<f32>(normal, 0.0)).xyz);

  // Endfield-style outline: expand along view dir + normal, scaled by distance
  // This ensures constant screen-space width regardless of viewing angle
  let camToVert = worldPos - frame.cameraPosition.xyz;
  let dist = length(camToVert);
  let viewDir = camToVert / max(dist, 0.001);

  // Distance-based scaling: farther = wider expansion to maintain screen-space size
  let distScale = dist * 0.04;

  // Mix view direction and normal expansion (0.5/0.5 blend)
  let expandDir = normalize(viewDir * 0.5 + worldNormal * 0.5);
  let expandedWorldPos = worldPos + expandDir * outlineWidth * distScale;

  return frame.viewProj * vec4<f32>(expandedWorldPos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.02, 0.02, 0.02, 1.0);
}
`;

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
  matUbo: GPUBuffer;
  name: string;
  category: MaterialCategory;
  baseColorFactor: [number, number, number, number];
  metallic: number;
  roughness: number;
  hasBaseColorTex: number;
  lightmapStrength: number;
  hasLightmap: boolean;
  previewImage: ImageBitmap | null;
  alphaCutoff: number;
  renderMode: number;
  outlineWidth: number;
}

export class GLBViewerDemo implements Demo {
  label = "GLB Viewer";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;
  private pipeline!: GPURenderPipeline;
  private outlinePipeline!: GPURenderPipeline;
  private frameUbo!: GPUBuffer;
  private meshBuffers: MeshBuffers[] = [];
  private materialResources: MaterialResources[] = [];
  private depthTexture: GPUTexture | null = null;
  private cachedDepthView: GPUTextureView | null = null;
  private frameData = new Float32Array(60);
  private loaded = false;
  private totalTriangles = 0;
  private modelName = "";
  private defaultTexture!: GPUTexture;
  private defaultSampler!: GPUSampler;
  private frameBindGroup!: GPUBindGroup;
  private presets: RenderPreset[] = [];

  rotationSpeed = 0.3;
  showOutline = true;
  globalOutlineWidth = 0.015;

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
      size: 240,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const module = this.device.createShaderModule({ code: modelShader });
    const outlineModule = this.device.createShaderModule({ code: outlineShader });

    const frameBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const matBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [frameBGL, matBGL] });

    this.pipeline = this.device.createRenderPipeline({
      label: "glb-render",
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    this.outlinePipeline = this.device.createRenderPipeline({
      label: "glb-outline",
      layout: pipelineLayout,
      vertex: { module: outlineModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outlineModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less", depthBias: 5 },
    });

    this.frameBindGroup = this.device.createBindGroup({
      layout: frameBGL,
      entries: [{ binding: 0, resource: { buffer: this.frameUbo } }],
    });

    try {
      const model = await loadGLTF("/Qin_DL.glb");
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

      // Apply default preset
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

    const matUbo = this.device.createBuffer({
      label: `glb-mat-ubo-${matData.name}`,
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const category = classifyMaterial(matData.name);
    const hasBaseColorTex = matData.baseColorImage ? 1 : 0;
    const lightmapStrength = matData.occlusionImage ? matData.occlusionStrength : 0;

    const matUboData = new Float32Array([
      matData.baseColorFactor[0], matData.baseColorFactor[1], matData.baseColorFactor[2], matData.baseColorFactor[3],
      matData.metallicFactor, matData.roughnessFactor, hasBaseColorTex, lightmapStrength,
      0.0, 1.0, this.globalOutlineWidth, 0.0,
    ]);
    this.device.queue.writeBuffer(matUbo, 0, matUboData as unknown as GPUAllowSharedBufferSource);

    const bindGroup = this.device.createBindGroup({
      layout: matBGL,
      entries: [
        { binding: 0, resource: { buffer: matUbo } },
        { binding: 1, resource: baseColorView },
        { binding: 2, resource: lightmapView },
        { binding: 3, resource: this.defaultSampler },
      ],
    });

    this.materialResources.push({
      bindGroup,
      matUbo,
      name: matData.name,
      category,
      baseColorFactor: matData.baseColorFactor,
      metallic: matData.metallicFactor,
      roughness: matData.roughnessFactor,
      hasBaseColorTex,
      lightmapStrength,
      hasLightmap: !!matData.occlusionImage,
      previewImage: matData.baseColorImage ?? null,
      alphaCutoff: 0.0,
      renderMode: 1,
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

  private ensureDepth() {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (this.depthTexture && this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.cachedDepthView = this.depthTexture.createView();
  }

  update(time: number) {
    if (!this.loaded) return;

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const model = mat4.rotationY(time * this.rotationSpeed);
    const invTrans = mat4.transpose(mat4.inverse(model));

    const d = this.frameData;
    d.set(viewProj as unknown as ArrayLike<number>, 0);
    d.set(model as unknown as ArrayLike<number>, 16);
    d.set(invTrans as unknown as ArrayLike<number>, 32);
    d[48] = this.camera.position[0]; d[49] = this.camera.position[1]; d[50] = this.camera.position[2]; d[51] = 1;
    d[52] = -0.5; d[53] = -1.0; d[54] = -0.3; d[55] = 0;
    d[56] = 1.0; d[57] = 1.0; d[58] = 1.0; d[59] = 2.0;

    this.device.queue.writeBuffer(this.frameUbo, 0, d as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    if (!this.loaded || this.meshBuffers.length === 0) return;
    this.ensureDepth();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.cachedDepthView!,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    // Outline pass (backface expansion, rendered first)
    if (this.showOutline) {
      pass.setPipeline(this.outlinePipeline);
      pass.setBindGroup(0, this.frameBindGroup);
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

    // Main pass
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.frameBindGroup);
    for (const mb of this.meshBuffers) {
      if (!mb.visible) continue;
      const matRes = this.materialResources[Math.min(mb.materialIndex, this.materialResources.length - 1)];
      pass.setBindGroup(1, matRes.bindGroup);
      pass.setVertexBuffer(0, mb.vertexBuffer);
      pass.setIndexBuffer(mb.indexBuffer, mb.use32bit ? "uint32" : "uint16");
      pass.drawIndexed(mb.indexCount);
    }

    pass.end();
  }

  // === Preset System ===
  applyPreset(preset: RenderPreset) {
    for (const matRes of this.materialResources) {
      const mode = preset.mapping[matRes.category] ?? preset.mapping.other ?? 1;
      matRes.renderMode = mode;
      this.updateMatUbo(matRes);
    }
  }

  private updateMatUbo(matRes: MaterialResources) {
    const data = new Float32Array([
      matRes.baseColorFactor[0], matRes.baseColorFactor[1], matRes.baseColorFactor[2], matRes.baseColorFactor[3],
      matRes.metallic, matRes.roughness, matRes.hasBaseColorTex, matRes.lightmapStrength,
      matRes.alphaCutoff, matRes.renderMode, matRes.outlineWidth, 0,
    ]);
    this.device.queue.writeBuffer(matRes.matUbo, 0, data as unknown as GPUAllowSharedBufferSource);
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
      drawCalls: this.meshBuffers.length * (this.showOutline ? 2 : 1),
      triangles: this.totalTriangles,
      custom: {
        "Model": this.modelName || "Loading...",
        "Meshes": this.meshBuffers.length,
        "Materials": this.materialResources.length,
        "Outline": this.showOutline ? "On" : "Off",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "rotationSpeed", 0, 2, 0.05).name("Rotation Speed");
    gui.add(this, "showOutline").name("Show Outline").onChange(() => {});
    gui.add(this, "globalOutlineWidth", 0, 0.05, 0.001).name("Outline Width").onChange((v: number) => {
      for (const matRes of this.materialResources) {
        matRes.outlineWidth = v;
        this.updateMatUbo(matRes);
      }
    });

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
          mapping[matRes.category] = matRes.renderMode;
        }
        const name = prompt("Preset name:", "My Preset") ?? "My Preset";
        const preset: RenderPreset = { name, mapping: mapping as Record<MaterialCategory, number> };
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
        renderMode: RENDER_MODE_NAMES[matRes.renderMode] ?? "PBR",
        metallic: matRes.metallic,
        roughness: matRes.roughness,
        alphaCutoff: matRes.alphaCutoff,
        outlineWidth: matRes.outlineWidth,
      };
      f.add(ctrl, "renderMode", RENDER_MODE_NAMES).name("Render Mode").onChange((v: string) => {
        matRes.renderMode = RENDER_MODE_NAMES.indexOf(v);
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
    });
  }

  destroy() {
    for (const mb of this.meshBuffers) {
      mb.vertexBuffer.destroy();
      mb.indexBuffer.destroy();
    }
    for (const mr of this.materialResources) {
      mr.matUbo.destroy();
      mr.previewImage?.close();
    }
    this.frameUbo.destroy();
    this.depthTexture?.destroy();
    this.defaultTexture.destroy();
  }
}
