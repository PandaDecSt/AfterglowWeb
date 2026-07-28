import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import type { Demo } from "./types";
import { mat4 } from "wgpu-matrix";

const terrainShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  time: f32,
  waterLevel: f32,
  debugMode: f32,
  pad0: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) height: f32,
};

fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2<f32>(1.0, 0.0)), u.x),
    mix(hash21(i + vec2<f32>(0.0, 1.0)), hash21(i + vec2<f32>(1.0, 1.0)), u.x),
    u.y
  );
}

fn fbm(p: vec2<f32>) -> f32 {
  var value = 0.0;
  var amplitude = 0.5;
  var position = p;
  for (var i = 0; i < 5; i++) {
    value += amplitude * noise(position);
    position *= 2.0;
    amplitude *= 0.5;
  }
  return value;
}

fn terrainHeight(uv: vec2<f32>) -> f32 {
  return fbm(uv * 4.0 + 0.5);
}

fn waterHeight(uv: vec2<f32>) -> f32 {
  let th = terrainHeight(uv);
  let waterLevel = 0.4;
  if (th < waterLevel) {
    return (waterLevel - th) * 1.5;
  }
  return 0.0;
}

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let th = terrainHeight(uv);
  let wh = waterHeight(uv);
  let h = max(th, wh * u.waterLevel) * 1.5;
  let displaced = vec3<f32>(pos.x, h, pos.z);
  let worldPos = u.model * vec4<f32>(displaced, 1.0);
  out.position = u.viewProj * worldPos;
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  out.height = th;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let th = terrainHeight(in.uv);
  let wh = waterHeight(in.uv);

  let N = normalize(vec3<f32>(0.0, 1.0, 0.0));
  let L = normalize(u.lightDir.xyz);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);

  var color: vec3<f32>;

  let debugMode = i32(u.debugMode);

  if (debugMode == 1) {
    // Height heatmap
    color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), th);
  } else if (debugMode == 2) {
    // Water only
    color = vec3<f32>(wh * 3.0);
  } else {
    // Normal terrain rendering
    color = mix(vec3<f32>(0.3, 0.5, 0.2), vec3<f32>(0.5, 0.4, 0.3), smoothstep(0.3, 0.6, th));
    color = mix(color, vec3<f32>(0.8, 0.8, 0.85), smoothstep(0.6, 0.85, th));

    let diff = max(dot(N, L), 0.0) * 0.5 + 0.5;
    color *= diff;

    if (wh > 0.05) {
      let waterColor = vec3<f32>(0.15, 0.4, 0.7);
      let H = normalize(V + L);
      let spec = pow(max(dot(N, H), 0.0), 32.0);
      color = mix(color, waterColor + vec3<f32>(spec * 0.4), 0.5);
    }
  }

  color *= u.lightDir.w;
  return vec4<f32>(color, 1.0);
}
`;

export class TerrainHydrologyDemo implements Demo {
  label = "Terrain Hydrology";
  private ctx!: GPUContext;
  private camera!: Camera;
  private device!: GPUDevice;
  private format!: GPUTextureFormat;

  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private depthTexture: GPUTexture | null = null;
  private cachedDepthView: GPUTextureView | null = null;

  private waterLevel = 1.0;
  private debugMode = 0;

  async init(ctx: GPUContext, camera: Camera) {
    this.ctx = ctx;
    this.camera = camera;
    this.device = ctx.device;
    this.format = ctx.format;

    const geo = this.createPlane(4.0, 64);
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

    const bgLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }],
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bgLayout,
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }),
      vertex: {
        module: this.device.createShaderModule({ code: terrainShader }),
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 20,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x2" },
          ],
        }],
      },
      fragment: {
        module: this.device.createShaderModule({ code: terrainShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
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

  private createPlane(size: number, segments: number) {
    const verts: number[] = [];
    const inds: number[] = [];
    const half = size / 2;
    const step = size / segments;

    for (let z = 0; z <= segments; z++) {
      for (let x = 0; x <= segments; x++) {
        const px = -half + x * step;
        const pz = -half + z * step;
        verts.push(px, 0, pz, x / segments, z / segments);
      }
    }

    for (let z = 0; z < segments; z++) {
      for (let x = 0; x < segments; x++) {
        const a = z * (segments + 1) + x;
        const b = a + segments + 1;
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

    const model = mat4.identity();
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
    ubo[57] = this.waterLevel;
    ubo[58] = this.debugMode;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    this.ensureDepth();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0.2, g: 0.25, b: 0.3, a: 1 } }],
      depthStencilAttachment: { view: this.cachedDepthView!, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1.0 },
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, "uint16");
    pass.drawIndexed(6 * 64 * 64);

    pass.end();
  }

  destroy() {
    this.depthTexture?.destroy();
  }

  registerGUI(gui: any) {
    const folder = gui.addFolder("Terrain Hydrology");
    folder.add(this, "waterLevel", 0, 2).name("Water Level");
    folder.add(this, "debugMode", { Normal: 0, Height: 1, Water: 2 }).name("Debug Mode");
  }
}
