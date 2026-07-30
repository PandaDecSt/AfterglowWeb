import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { mat4 } from "wgpu-matrix";
import type { RenderPass } from "../core/renderer";

const arcToonVS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  time: f32,
  outlineWidth: f32,
  materialID: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) vertexColor: f32,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  out.position = u.viewProj * worldPos;
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  // Procedural vertex color based on height (simulates ILM vertex color)
  out.vertexColor = smoothstep(0.0, 1.5, worldPos.y);
  return out;
}
`;

const arcToonFS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  time: f32,
  outlineWidth: f32,
  materialID: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) vertexColor: f32,
};

// Procedural ILM-like map
fn sampleILM(uv: vec2<f32>, materialID: f32) -> vec4<f32> {
  // ILM channels:
  // .x: vertex color (used for diffuse masking)
  // .y: luminance correction
  // .z: specular mask
  // .w: internal lines (for outline effect)

  var ilm = vec4<f32>(1.0, 1.0, 0.5, 1.0);

  // Material-specific ILM patterns
  if (materialID < 0.5) {
    // Face: smooth ILM with cheek highlights
    let cheekL = distance(uv, vec2<f32>(0.3, 0.55));
    let cheekR = distance(uv, vec2<f32>(0.7, 0.55));
    let cheek = max(smoothstep(0.15, 0.05, cheekL), smoothstep(0.15, 0.05, cheekR));
    ilm.x = 1.0 - cheek * 0.3;
    ilm.z = 0.3; // Low specular on skin
    ilm.w = 1.0 - smoothstep(0.02, 0.0, abs(uv.x - 0.5)) * 0.3; // Nose line
  } else if (materialID < 1.5) {
    // Hair: strong specular band
    ilm.z = smoothstep(0.3, 0.7, sin(uv.x * 30.0 + uv.y * 10.0) * 0.5 + 0.5);
    ilm.w = 1.0 - sin(uv.x * 50.0) * 0.1; // Strand lines
  } else if (materialID < 2.5) {
    // Body: fabric pattern
    let pattern = step(0.5, fract(uv.y * 8.0));
    ilm.x = 1.0 - pattern * 0.2;
    ilm.z = 0.6; // Medium specular
    ilm.w = 1.0 - smoothstep(0.48, 0.52, fract(uv.y * 8.0)) * 0.4; // Seam lines
  } else {
    // Eye: high specular, reflective
    let irisDist = distance(uv, vec2<f32>(0.5, 0.5));
    ilm.z = smoothstep(0.3, 0.1, irisDist); // Strong specular on iris
    ilm.w = 1.0;
  }

  return ilm;
}

// Procedural SSS-like color
fn sampleSSS(baseColor: vec3<f32>, materialID: f32, NdotL: f32) -> vec3<f32> {
  var sssColor = baseColor;

  if (materialID < 0.5) {
    // Face: warm subsurface scattering
    let sss = smoothstep(-0.2, 0.3, NdotL);
    sssColor = mix(vec3<f32>(0.85, 0.55, 0.5), baseColor, sss);
  } else if (materialID < 1.5) {
    // Hair: cool subsurface
    let sss = smoothstep(-0.1, 0.2, NdotL);
    sssColor = mix(vec3<f32>(0.15, 0.1, 0.25), baseColor, sss);
  } else if (materialID < 2.5) {
    // Body: neutral subsurface
    let sss = smoothstep(-0.1, 0.2, NdotL);
    sssColor = mix(vec3<f32>(0.15, 0.15, 0.2), baseColor, sss);
  }

  return sssColor;
}

// GGX specular with ILM mask
fn specularGGX(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32, F0: vec3<f32>) -> vec3<f32> {
  let H = normalize(V + L);
  let NdotH = max(dot(N, H), 0.0);
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);

  let a = roughness * roughness;
  let a2 = a * a;
  let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
  let NDF = a2 / (3.14159 * denom * denom);

  let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
  let G1 = NdotV / (NdotV * (1.0 - k) + k);
  let G2 = NdotL / (NdotL * (1.0 - k) + k);
  let G = G1 * G2;

  let F = F0 + (1.0 - F0) * pow(clamp(1.0 - max(dot(H, V), 0.0), 0.0, 1.0), 5.0);

  let numerator = NDF * G * F;
  let denominator = 4.0 * NdotV * NdotL + 0.0001;

  return numerator / denominator;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let L = normalize(-u.lightDir.xyz);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);
  let NdotL = dot(N, L);

  let materialID = u.materialID;

  // Base colors per material
  var baseColor: vec3<f32>;
  if (materialID < 0.5) {
    baseColor = vec3<f32>(0.95, 0.78, 0.68); // Face skin
  } else if (materialID < 1.5) {
    baseColor = vec3<f32>(0.25, 0.2, 0.35); // Hair
  } else if (materialID < 2.5) {
    baseColor = vec3<f32>(0.2, 0.5, 0.85); // Body
  } else {
    baseColor = vec3<f32>(0.1, 0.3, 0.6); // Eye
  }

  // Sample procedural ILM
  let ilm = sampleILM(in.uv, materialID);

  // Toon diffuse with smoothstep (Arc Toon style)
  let diffuse = smoothstep(0.49, 0.50, NdotL * ilm.x);

  // SSS-like shadow color
  let sssColor = sampleSSS(baseColor, materialID, NdotL);

  // Mix diffuse with SSS
  var color = mix(sssColor, baseColor, diffuse);

  // GGX specular with ILM mask
  let specularColor = mix(vec3<f32>(0.04), baseColor, 0.0);
  let specular = specularGGX(N, V, L, 0.5, specularColor);
  let specMask = smoothstep(vec3<f32>(0.04), vec3<f32>(0.05), specular * ilm.z);
  color += specMask * ilm.z;

  // Internal lines (from ILM)
  color *= ilm.w;

  // Lighting
  color *= u.lightColor.rgb * u.lightColor.a;

  // Luminance correction
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  color = color * ilm.y * (1.0 + luma * 0.2);

  return vec4<f32>(color, 1.0);
}
`;

const arcOutlineVS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  time: f32,
  outlineWidth: f32,
  materialID: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> @builtin(position) vec4<f32> {
  let worldPos = (u.model * vec4<f32>(pos, 1.0)).xyz;
  let worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);

  // Distance fade
  let dist = distance(u.cameraPosition.xyz, worldPos);
  let distanceFade = min(dist * 0.15, 0.01);

  // View + normal expansion (Arc Toon style)
  let camToVertex = worldPos - u.cameraPosition.xyz;
  let expansion = u.outlineWidth * distanceFade * (camToVertex * 0.6 + worldNormal * 1.0);
  let expandedWorldPos = worldPos + expansion;

  return u.viewProj * vec4<f32>(expandedWorldPos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  // Arc Toon uses dark outline color based on material
  return vec4<f32>(0.05, 0.02, 0.08, 1.0);
}
`;

interface MaterialPart {
  name: string;
  materialID: number;
  offset: number;
  count: number;
  outlineScale: number;
}

export class ArcToonDemo implements Demo {
  label = "Arc Toon";

  private device!: GPUDevice;
  private format!: GPUTextureFormat;
  private toonPipeline!: GPURenderPipeline;
  private outlinePipeline!: GPURenderPipeline;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private camera!: Camera;
  private ctx!: GPUContext;
  private depthTexture: GPUTexture | null = null;
  private parts: MaterialPart[] = [];
  private static readonly SLOT_SIZE = 256;

  outlineWidth = 1.5;
  showOutline = true;
  showFace = true;
  showHair = true;
  showBody = true;
  showEye = true;

  init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.format = ctx.format;
    this.camera = camera;
    this.ctx = ctx;

    const { vertices, indices, parts } = this.createCharacterGeometry();
    this.parts = parts;

    this.vertexBuffer = this.device.createBuffer({
      label: "arctoon-vb",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: "arctoon-ib",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    const maxSlots = 16;
    this.uniformBuffer = this.device.createBuffer({
      label: "arctoon-ubo",
      size: ArcToonDemo.SLOT_SIZE * maxSlots,
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

    const bindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform", hasDynamicOffset: true },
        },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    });

    const toonModule = this.device.createShaderModule({ code: arcToonFS });
    const toonVSModule = this.device.createShaderModule({ code: arcToonVS });
    this.toonPipeline = this.device.createRenderPipeline({
      label: "arctoon-render",
      layout: pipelineLayout,
      vertex: { module: toonVSModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: toonModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const outlineModule = this.device.createShaderModule({ code: arcOutlineVS });
    this.outlinePipeline = this.device.createRenderPipeline({
      label: "arctoon-outline",
      layout: pipelineLayout,
      vertex: { module: outlineModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outlineModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer, size: ArcToonDemo.SLOT_SIZE } }],
    });
  }

  private createCharacterGeometry(): {
    vertices: Float32Array;
    indices: Uint16Array;
    parts: MaterialPart[];
  } {
    const allVerts: number[] = [];
    const allInds: number[] = [];
    const parts: MaterialPart[] = [];
    let vertOffset = 0;

    const addSphere = (
      cx: number, cy: number, cz: number,
      radius: number, segments: number, rings: number,
      name: string, materialID: number, outlineScale: number,
      uScale = 1, vScale = 1
    ) => {
      const baseVert = vertOffset;
      const indexStart = allInds.length;

      for (let y = 0; y <= rings; y++) {
        const phi = (y / rings) * Math.PI;
        for (let x = 0; x <= segments; x++) {
          const theta = (x / segments) * Math.PI * 2;
          const nx = Math.sin(phi) * Math.cos(theta);
          const ny = Math.cos(phi);
          const nz = Math.sin(phi) * Math.sin(theta);
          allVerts.push(
            cx + nx * radius, cy + ny * radius, cz + nz * radius,
            nx, ny, nz,
            (x / segments) * uScale, (y / rings) * vScale
          );
          vertOffset++;
        }
      }

      for (let y = 0; y < rings; y++) {
        for (let x = 0; x < segments; x++) {
          const a = baseVert + y * (segments + 1) + x;
          const b = a + segments + 1;
          allInds.push(a, a + 1, b, a + 1, b + 1, b);
        }
      }

      parts.push({
        name,
        materialID,
        offset: indexStart,
        count: allInds.length - indexStart,
        outlineScale,
      });
    };

    // Head (face)
    addSphere(0, 1.2, 0, 0.6, 24, 16, "Face", 0, 1.0);
    // Hair
    addSphere(0, 1.45, -0.1, 0.62, 20, 12, "Hair", 1, 1.2, 2, 1);
    // Body
    addSphere(0, 0.1, 0, 0.5, 16, 12, "Body", 2, 0.8, 1, 3);
    // Left eye
    addSphere(-0.2, 1.25, 0.5, 0.12, 12, 8, "EyeL", 3, 0.5);
    // Right eye
    addSphere(0.2, 1.25, 0.5, 0.12, 12, 8, "EyeR", 3, 0.5);

    return {
      vertices: new Float32Array(allVerts),
      indices: new Uint16Array(allInds),
      parts,
    };
  }

  private cachedDepthView: GPUTextureView | null = null;
  private uboScratch = new Float32Array(64);
  private renderBundle: GPURenderBundle | null = null;
  private lastVisibilityKey = "";

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
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);

    const model = mat4.mul(
      mat4.rotationY(time * 0.5),
      mat4.scaling([1.2, 1.2, 1.2])
    );
    const invTransModel = mat4.transpose(mat4.inverse(model));

    const visibility = [this.showFace, this.showHair, this.showBody, this.showEye];
    let slot = 0;
    const ubo = this.uboScratch;

    for (const part of this.parts) {
      if (!visibility[part.materialID]) continue;

      ubo.fill(0);
      ubo.set(viewProj as unknown as ArrayLike<number>, 0);
      ubo.set(model as unknown as ArrayLike<number>, 16);
      ubo.set(invTransModel as unknown as ArrayLike<number>, 32);
      ubo[48] = this.camera.position[0];
      ubo[49] = this.camera.position[1];
      ubo[50] = this.camera.position[2];
      ubo[51] = 1.0;
      // Light direction (normalized)
      const lightLen = Math.sqrt(0.4 * 0.4 + 1.0 * 1.0 + 0.3 * 0.3);
      ubo[52] = -0.4 / lightLen;
      ubo[53] = -1.0 / lightLen;
      ubo[54] = -0.3 / lightLen;
      ubo[55] = 0.0;
      // Light color + intensity
      ubo[56] = 3.0;
      ubo[57] = 3.0;
      ubo[58] = 3.0;
      ubo[59] = 1.0;
      ubo[60] = time;
      ubo[61] = this.outlineWidth * part.outlineScale;
      ubo[62] = part.materialID;
      this.device.queue.writeBuffer(this.uniformBuffer, slot * ArcToonDemo.SLOT_SIZE, ubo as unknown as GPUAllowSharedBufferSource);
      (part as any)._outlineSlot = slot;
      slot++;

      ubo[61] = this.outlineWidth;
      this.device.queue.writeBuffer(this.uniformBuffer, slot * ArcToonDemo.SLOT_SIZE, ubo as unknown as GPUAllowSharedBufferSource);
      (part as any)._toonSlot = slot;
      slot++;
    }
  }

  private buildRenderBundle() {
    const visibility = [this.showFace, this.showHair, this.showBody, this.showEye];
    const key = `${this.showOutline}|${visibility.join("")}`;
    if (key === this.lastVisibilityKey && this.renderBundle) return;
    this.lastVisibilityKey = key;

    const bundleEncoder = this.device.createRenderBundleEncoder({
      colorFormats: [this.format],
      depthStencilFormat: "depth24plus",
    });

    bundleEncoder.setVertexBuffer(0, this.vertexBuffer);
    bundleEncoder.setIndexBuffer(this.indexBuffer, "uint16");

    if (this.showOutline) {
      bundleEncoder.setPipeline(this.outlinePipeline);
      for (const part of this.parts) {
        if (!visibility[part.materialID]) continue;
        bundleEncoder.setBindGroup(0, this.bindGroup, [(part as any)._outlineSlot * ArcToonDemo.SLOT_SIZE]);
        bundleEncoder.drawIndexed(part.count, 1, part.offset);
      }
    }

    bundleEncoder.setPipeline(this.toonPipeline);
    for (const part of this.parts) {
      if (!visibility[part.materialID]) continue;
      bundleEncoder.setBindGroup(0, this.bindGroup, [(part as any)._toonSlot * ArcToonDemo.SLOT_SIZE]);
      bundleEncoder.drawIndexed(part.count, 1, part.offset);
    }

    this.renderBundle = bundleEncoder.finish();
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        this.ensureDepth();
        this.buildRenderBundle();

        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0.9, g: 0.91, b: 0.94, a: 1 },
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

        pass.executeBundles([this.renderBundle!]);
        pass.end();
      },
    }];
  }

  stats() {
    const visibleParts = this.parts.filter((p) =>
      [this.showFace, this.showHair, this.showBody, this.showEye][p.materialID]
    );
    const drawCalls = visibleParts.length * (this.showOutline ? 2 : 1);
    const tris = visibleParts.reduce((sum, p) => sum + p.count / 3, 0) * (this.showOutline ? 2 : 1);
    return {
      drawCalls,
      triangles: tris,
      custom: {
        "Materials": `${visibleParts.length} / ${this.parts.length}`,
        "Outline": this.showOutline ? "Arc Style" : "Off",
        "Technique": "ILM + SSS + GGX",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "outlineWidth", 0.0, 5.0, 0.1).name("Outline Width");
    gui.add(this, "showOutline").name("Show Outline");
    const f = gui.addFolder("Material Visibility");
    f.add(this, "showFace").name("Face (SSS)");
    f.add(this, "showHair").name("Hair (Aniso)");
    f.add(this, "showBody").name("Body (Banded)");
    f.add(this, "showEye").name("Eye (Reflective)");
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.uniformBuffer.destroy();
    this.depthTexture?.destroy();
  }
}
