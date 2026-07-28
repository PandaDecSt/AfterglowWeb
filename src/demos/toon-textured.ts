import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import type { Demo } from "./types";
import { mat4 } from "wgpu-matrix";

const toonTexturedShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  time: f32,
  materialID: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var rampTex: texture_2d<f32>;
@group(1) @binding(1) var rampSampler: sampler;

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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);
  let L = normalize(u.lightDir.xyz);

  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 0.0);

  let rampOffset = 0.2;
  let rampX = clamp((1.0 - rampOffset) - NdotL * (0.5 - rampOffset * 0.5), 0.1, 0.9);
  let rampColor = textureSample(rampTex, rampSampler, vec2<f32>(rampX, 0.5));

  var baseColor: vec3<f32>;
  let matID = i32(u.materialID);
  if (matID == 0) { baseColor = vec3<f32>(0.95, 0.78, 0.68); }
  else if (matID == 1) { baseColor = vec3<f32>(0.35, 0.25, 0.45); }
  else if (matID == 2) { baseColor = vec3<f32>(0.3, 0.6, 0.9); }
  else { baseColor = vec3<f32>(0.2, 0.7, 0.9); }

  let diffuse = NdotL * 0.6 + 0.4;
  let rampContrib = rampColor.rgb * 0.35;
  var color = baseColor * (diffuse + rampContrib);

  let rim = smoothstep(0.6, 1.0, 1.0 - NdotV);
  color += baseColor * rim * 0.25;

  let lightIntensity = u.lightDir.w;
  color *= lightIntensity;

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
  pad: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> @builtin(position) vec4<f32> {
  let expanded = pos + normal * u.outlineWidth;
  return u.viewProj * u.model * vec4<f32>(expanded, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
  return vec4<f32>(0.1, 0.05, 0.15, 1.0);
}
`;

export class ToonTexturedDemo implements Demo {
  label = "Toon Textured (Ramp)";
  private ctx!: GPUContext;
  private camera!: Camera;
  private device!: GPUDevice;
  private format!: GPUTextureFormat;

  private pipeline!: GPURenderPipeline;
  private outlinePipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private mainBindGroup!: GPUBindGroup;
  private outlineBindGroup!: GPUBindGroup;
  private rampBindGroup!: GPUBindGroup;
  private rampTexture!: GPUTexture;
  private depthTexture: GPUTexture | null = null;
  private cachedDepthView: GPUTextureView | null = null;

  private outlineWidth = 0.02;

  async init(ctx: GPUContext, camera: Camera) {
    this.ctx = ctx;
    this.camera = camera;
    this.device = ctx.device;
    this.format = ctx.format;

    this.rampTexture = this.createRampTexture();

    const geo = this.createSphere(1.0, 32, 32);

    this.vertexBuffer = this.device.createBuffer({
      size: geo.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, geo.vertices);

    this.indexBuffer = this.device.createBuffer({
      size: geo.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indexBuffer, 0, geo.indices);

    this.uniformBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const uniformBGLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }],
    });

    const rampBGLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.mainBindGroup = this.device.createBindGroup({
      layout: uniformBGLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    this.outlineBindGroup = this.device.createBindGroup({
      layout: this.device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }],
      }),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    this.rampBindGroup = this.device.createBindGroup({
      layout: rampBGLayout,
      entries: [
        { binding: 0, resource: this.rampTexture.createView() },
        { binding: 1, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [uniformBGLayout, rampBGLayout],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: toonTexturedShader }),
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ],
        }],
      },
      fragment: {
        module: this.device.createShaderModule({ code: toonTexturedShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    this.outlinePipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.device.createBindGroupLayout({
          entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: {} }],
        })],
      }),
      vertex: {
        module: this.device.createShaderModule({ code: outlineShader }),
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ],
        }],
      },
      fragment: {
        module: this.device.createShaderModule({ code: outlineShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "front" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
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

  private createRampTexture(): GPUTexture {
    const size = 256;
    const data = new Uint8Array(size * 4);
    for (let i = 0; i < size; i++) {
      const t = i / (size - 1);
      let r: number, g: number, b: number;
      if (t < 0.33) { r = 0.3; g = 0.25; b = 0.35; }
      else if (t < 0.66) { r = 0.6; g = 0.5; b = 0.7; }
      else { r = 0.9; g = 0.85; b = 1.0; }
      data[i * 4 + 0] = Math.floor(r * 255);
      data[i * 4 + 1] = Math.floor(g * 255);
      data[i * 4 + 2] = Math.floor(b * 255);
      data[i * 4 + 3] = 255;
    }

    const tex = this.device.createTexture({
      size: [size, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: size * 4 }, [size, 1]);
    return tex;
  }

  private createSphere(radius: number, wSeg: number, hSeg: number) {
    const verts: number[] = [];
    const inds: number[] = [];
    for (let y = 0; y <= hSeg; y++) {
      for (let x = 0; x <= wSeg; x++) {
        const u = x / wSeg;
        const v = y / hSeg;
        const theta = u * Math.PI * 2;
        const phi = v * Math.PI;
        const px = radius * Math.sin(phi) * Math.cos(theta);
        const py = radius * Math.cos(phi);
        const pz = radius * Math.sin(phi) * Math.sin(theta);
        verts.push(px, py, pz, px, py, pz, u, v);
      }
    }
    for (let y = 0; y < hSeg; y++) {
      for (let x = 0; x < wSeg; x++) {
        const a = y * (wSeg + 1) + x;
        const b = a + wSeg + 1;
        inds.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    return { vertices: new Float32Array(verts), indices: new Uint16Array(inds) };
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);

    const model = mat4.mul(mat4.rotationY(time * 0.3), mat4.scaling([1.5, 1.5, 1.5]));
    const invTransModel = mat4.transpose(mat4.inverse(model));

    const ubo = new Float32Array(64);
    ubo.set(viewProj as unknown as ArrayLike<number>, 0);
    ubo.set(model as unknown as ArrayLike<number>, 16);
    ubo.set(invTransModel as unknown as ArrayLike<number>, 32);
    ubo[48] = this.camera.position[0];
    ubo[49] = this.camera.position[1];
    ubo[50] = this.camera.position[2];
    ubo[51] = 1.0;
    ubo[52] = -0.4; ubo[53] = -1.0; ubo[54] = -0.3; ubo[55] = 1.0;
    ubo[56] = time;
    ubo[57] = this.outlineWidth;
    ubo[58] = 2.0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    this.ensureDepth();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0.15, g: 0.15, b: 0.2, a: 1 } }],
      depthStencilAttachment: { view: this.cachedDepthView!, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1.0 },
    });

    // Main toon pass first
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.mainBindGroup);
    pass.setBindGroup(1, this.rampBindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint16");
    pass.drawIndexed(1920);

    // Outline pass after (back faces, slightly expanded)
    pass.setPipeline(this.outlinePipeline);
    pass.setBindGroup(0, this.outlineBindGroup);
    pass.drawIndexed(1920);

    pass.end();
  }

  destroy() {
    this.depthTexture?.destroy();
    this.rampTexture?.destroy();
  }

  registerGUI(gui: any) {
    const folder = gui.addFolder("Toon Textured");
    folder.add(this, "outlineWidth", 0, 0.1).name("Outline Width");
  }
}
