import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { mat4 } from "wgpu-matrix";

const multiToonShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
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
  return out;
}

// Material 0: Face - SDF-like soft shadow, warm skin tone
fn shadeFace(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let NdotL = dot(N, L);
  let shadowEdge = smoothstep(-0.02, 0.05, NdotL);
  let baseColor = vec3<f32>(0.95, 0.78, 0.68);
  let shadowColor = vec3<f32>(0.85, 0.55, 0.5);
  var color = mix(shadowColor, baseColor, shadowEdge);
  // Cheek blush
  let cheekL = distance(uv, vec2<f32>(0.3, 0.55));
  let cheekR = distance(uv, vec2<f32>(0.7, 0.55));
  let blush = max(smoothstep(0.12, 0.04, cheekL), smoothstep(0.12, 0.04, cheekR));
  color = mix(color, vec3<f32>(0.95, 0.5, 0.45), blush * 0.4);
  // Nose highlight
  let nose = smoothstep(0.06, 0.0, distance(uv, vec2<f32>(0.5, 0.5)));
  color += nose * 0.1;
  return color;
}

// Material 1: Hair - anisotropic-like strand highlight, cool tone
fn shadeHair(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let NdotL = dot(N, L);
  let diffuse = smoothstep(0.0, 0.3, NdotL) * 0.7 + 0.3;
  let baseColor = vec3<f32>(0.25, 0.2, 0.35);
  var color = baseColor * diffuse;
  // Strand highlight (anisotropic approximation)
  let H = normalize(L + V);
  let strandDir = normalize(vec3<f32>(0.0, 1.0, 0.0));
  let TdotH = dot(strandDir, H);
  let aniso = pow(sqrt(max(1.0 - TdotH * TdotH, 0.0)), 8.0);
  let specBand = step(0.6, aniso) * 0.5;
  color += vec3<f32>(0.6, 0.5, 0.8) * specBand;
  // Hair strand lines
  let strands = sin(uv.x * 80.0 + uv.y * 20.0) * 0.5 + 0.5;
  color *= 0.9 + strands * 0.1;
  return color;
}

// Material 2: Body/Clothes - hard banded toon, saturated
fn shadeBody(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let NdotL = dot(N, L);
  var band: f32;
  if (NdotL > 0.7) { band = 1.0; }
  else if (NdotL > 0.3) { band = 0.65; }
  else if (NdotL > 0.0) { band = 0.4; }
  else { band = 0.2; }
  let baseColor = vec3<f32>(0.2, 0.5, 0.85);
  var color = baseColor * band;
  // Rim light
  let rim = 1.0 - max(dot(N, V), 0.0);
  color += baseColor * step(0.7, rim) * 0.5;
  // Fabric pattern
  let pattern = step(0.5, fract(uv.y * 6.0));
  color = mix(color, color * 0.8, pattern * 0.3);
  return color;
}

// Material 3: Eye - reflective, sharp highlight
fn shadeEye(N: vec3<f32>, L: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let irisCenter = vec2<f32>(0.5, 0.5);
  let irisDist = distance(uv, irisCenter);
  let irisMask = smoothstep(0.35, 0.3, irisDist);
  let pupilMask = smoothstep(0.12, 0.1, irisDist);
  // Iris color gradient
  let irisColor = mix(vec3<f32>(0.2, 0.6, 0.8), vec3<f32>(0.1, 0.3, 0.6), irisDist * 3.0);
  var color = mix(vec3<f32>(0.95, 0.95, 0.95), irisColor, irisMask);
  color = mix(color, vec3<f32>(0.02, 0.02, 0.05), pupilMask);
  // Sharp specular
  let H = normalize(L + V);
  let spec = pow(max(dot(N, H), 0.0), 64.0);
  color += step(0.5, spec) * 0.9;
  // Reflection dot
  let reflPos = vec2<f32>(0.38, 0.38);
  let refl = smoothstep(0.06, 0.03, distance(uv, reflPos));
  color += refl * 0.8;
  return color;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let L = normalize(-u.lightDir.xyz);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);

  var color: vec3<f32>;
  let matID = i32(u.materialID);
  if (matID == 0) {
    color = shadeFace(N, L, V, in.uv);
  } else if (matID == 1) {
    color = shadeHair(N, L, V, in.uv);
  } else if (matID == 2) {
    color = shadeBody(N, L, V, in.uv);
  } else {
    color = shadeEye(N, L, V, in.uv);
  }

  return vec4<f32>(color, 1.0);
}
`;

const outlineShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  time: f32,
  outlineWidth: f32,
  materialID: f32,
  distanceFadeFactor: f32,
  fovCompensation: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
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

  // Distance fade: caps expansion growth at far distances
  let dist = distance(u.cameraPosition.xyz, worldPos);
  let distanceFade = min(dist * u.distanceFadeFactor, 0.01);

  // View-direction expansion (unnormalized, scales with distance like original)
  let camToVertex = worldPos - u.cameraPosition.xyz;
  let faceExpand = camToVertex * 0.6;

  // Normal-direction expansion (pushes shell outward)
  let normalExpand = worldNormal * 1.0;

  // Combined expansion with FOV compensation for screen-space consistency
  let expansion = u.outlineWidth * (distanceFade * u.fovCompensation) * (faceExpand + normalExpand);
  let expandedWorldPos = worldPos + expansion;

  return u.viewProj * vec4<f32>(expandedWorldPos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  let matID = i32(u.materialID);
  var outlineColor: vec3<f32>;
  if (matID == 0) {
    outlineColor = vec3<f32>(0.15, 0.05, 0.05);
  } else if (matID == 1) {
    outlineColor = vec3<f32>(0.05, 0.02, 0.1);
  } else if (matID == 2) {
    outlineColor = vec3<f32>(0.02, 0.05, 0.12);
  } else {
    outlineColor = vec3<f32>(0.02, 0.02, 0.02);
  }
  return vec4<f32>(outlineColor, 1.0);
}
`;

interface MaterialPart {
  name: string;
  materialID: number;
  offset: number;
  count: number;
  outlineScale: number;
}

export class ToonDemo implements Demo {
  label = "Toon Multi-Material";

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

  outlineWidth = 2.0;
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
      label: "toon-vb",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: "toon-ib",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    const maxSlots = 16;
    this.uniformBuffer = this.device.createBuffer({
      label: "toon-ubo",
      size: ToonDemo.SLOT_SIZE * maxSlots,
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

    const module = this.device.createShaderModule({ code: multiToonShader });
    this.toonPipeline = this.device.createRenderPipeline({
      label: "toon-render",
      layout: pipelineLayout,
      vertex: { module, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const outlineModule = this.device.createShaderModule({ code: outlineShader });
    this.outlinePipeline = this.device.createRenderPipeline({
      label: "toon-outline",
      layout: pipelineLayout,
      vertex: { module: outlineModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outlineModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer, size: ToonDemo.SLOT_SIZE } }],
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

    // Head (face) - large sphere
    addSphere(0, 1.2, 0, 0.6, 24, 16, "Face", 0, 1.0);
    // Hair - slightly larger sphere on top/back
    addSphere(0, 1.45, -0.1, 0.62, 20, 12, "Hair", 1, 1.2, 2, 1);
    // Body - elongated sphere below
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

    const fovRad = (this.camera.fov * Math.PI) / 180;
    const fovCompensation = Math.tan(fovRad / 2);
    const distanceFadeFactor = 0.15;

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
      ubo[52] = -0.4; ubo[53] = -1.0; ubo[54] = -0.3; ubo[55] = 0.0;
      ubo[56] = time;
      ubo[57] = this.outlineWidth * part.outlineScale;
      ubo[58] = part.materialID;
      ubo[59] = distanceFadeFactor;
      ubo[60] = fovCompensation;
      this.device.queue.writeBuffer(this.uniformBuffer, slot * ToonDemo.SLOT_SIZE, ubo as unknown as GPUAllowSharedBufferSource);
      (part as any)._outlineSlot = slot;
      slot++;

      ubo[57] = this.outlineWidth;
      this.device.queue.writeBuffer(this.uniformBuffer, slot * ToonDemo.SLOT_SIZE, ubo as unknown as GPUAllowSharedBufferSource);
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
        bundleEncoder.setBindGroup(0, this.bindGroup, [(part as any)._outlineSlot * ToonDemo.SLOT_SIZE]);
        bundleEncoder.drawIndexed(part.count, 1, part.offset);
      }
    }

    bundleEncoder.setPipeline(this.toonPipeline);
    for (const part of this.parts) {
      if (!visibility[part.materialID]) continue;
      bundleEncoder.setBindGroup(0, this.bindGroup, [(part as any)._toonSlot * ToonDemo.SLOT_SIZE]);
      bundleEncoder.drawIndexed(part.count, 1, part.offset);
    }

    this.renderBundle = bundleEncoder.finish();
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
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
        "Outline": this.showOutline ? "Per-Material" : "Off",
        "Shading": "Face/Hair/Body/Eye",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "outlineWidth", 0.0, 5.0, 0.1).name("Outline Width");
    gui.add(this, "showOutline").name("Show Outline");
    const f = gui.addFolder("Material Visibility");
    f.add(this, "showFace").name("Face (Skin)");
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
