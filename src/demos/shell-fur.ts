import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { mat4 } from "wgpu-matrix";

const SHELL_COUNT = 16;
const SHELL_SLOT_SIZE = 256; // must be 256-byte aligned for dynamic offsets

const shellRenderShader = `
struct SharedUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
};

struct ShellParams {
  shellCount: f32,
  shellIndex: f32,
  furLength: f32,
  furDensity: f32,
};

@group(0) @binding(0) var<uniform> u: SharedUniforms;
@group(1) @binding(0) var<uniform> sp: ShellParams;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) shellLayer: f32,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  let layerOffset = (sp.shellIndex / sp.shellCount) * sp.furLength;
  let worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  let worldPos = (u.model * vec4<f32>(pos, 1.0)).xyz + worldNormal * layerOffset;

  var out: VSOut;
  out.position = u.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldNormal = worldNormal;
  out.uv = uv;
  out.shellLayer = sp.shellIndex / sp.shellCount;
  return out;
}

fn furPattern(uv: vec2<f32>, density: f32) -> f32 {
  let scaledUV = uv * density;
  let cell = floor(scaledUV);
  let localUV = fract(scaledUV);
  let rnd = fract(sin(dot(cell, vec2<f32>(127.1, 311.7))) * 43758.5453);
  let rnd2 = fract(rnd * 7.31);
  let strandCenter = vec2<f32>(0.2 + rnd * 0.6, 0.2 + rnd2 * 0.6);
  return distance(localUV, strandCenter);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let shellLayer = in.shellLayer;
  let density = sp.furDensity;

  // Shell 0 = solid base (skin)
  if (shellLayer < 0.01) {
    let L = normalize(-u.lightDir.xyz);
    let NdotL = max(dot(in.worldNormal, L), 0.0);
    let baseColor = vec3<f32>(0.3, 0.18, 0.08);
    return vec4<f32>(baseColor * (0.3 + NdotL * 0.7), 1.0);
  }

  // Strand cross-section shrinks quadratically with height
  let strandDist = furPattern(in.uv, density);
  let strandRadius = 0.5 * (1.0 - shellLayer) * (1.0 - shellLayer);

  if (strandDist > strandRadius) {
    discard;
  }

  let L = normalize(-u.lightDir.xyz);
  let NdotL = max(dot(in.worldNormal, L), 0.0);

  let baseColor = vec3<f32>(0.35, 0.2, 0.08);
  let tipColor = vec3<f32>(0.75, 0.55, 0.3);
  let furColor = mix(baseColor, tipColor, shellLayer);

  let edgeFactor = 1.0 - smoothstep(strandRadius * 0.3, strandRadius, strandDist);
  let color = furColor * (0.3 + NdotL * 0.7) * (0.6 + edgeFactor * 0.4);
  let alpha = 1.0 - smoothstep(0.85, 1.0, shellLayer);

  return vec4<f32>(color, alpha);
}
`;

export class ShellFurDemo implements Demo {
  label = "ShellFur";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private pipeline!: GPURenderPipeline;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private sharedUniformBuffer!: GPUBuffer;
  private shellParamsBuffer!: GPUBuffer;
  private sharedBindGroup!: GPUBindGroup;
  private shellBindGroup!: GPUBindGroup;
  private indexCount = 0;
  private sharedData = new Float32Array(56);

  furLength = 0.15;
  furDensity = 30.0;

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    const { vertices, indices } = this.createSphere(32, 24, 1.5);
    this.indexCount = indices.length;

    this.vertexBuffer = this.device.createBuffer({
      label: "shell-vb",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    const indexBufferSize = Math.ceil(indices.byteLength / 4) * 4;
    this.indexBuffer = this.device.createBuffer({
      label: "shell-ib",
      size: indexBufferSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    // Shared uniforms: viewProj(16) + model(16) + invTransModel(16) + cameraPosition(4) + lightDir(4) = 56 floats = 224 bytes
    this.sharedUniformBuffer = this.device.createBuffer({
      label: "shell-shared-ubo",
      size: 224,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Per-shell params: SHELL_COUNT slots of 16 bytes each (dynamic offset)
    this.shellParamsBuffer = this.device.createBuffer({
      label: "shell-params-ubo",
      size: SHELL_COUNT * SHELL_SLOT_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Pre-write per-shell params (shellCount, shellIndex, furLength, furDensity)
    this.writeShellParams();

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const module = this.device.createShaderModule({ code: shellRenderShader });

    const sharedBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const shellBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      label: "shell-render",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [sharedBGL, shellBGL] }),
      vertex: { module, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });

    this.sharedBindGroup = this.device.createBindGroup({
      layout: sharedBGL,
      entries: [{ binding: 0, resource: { buffer: this.sharedUniformBuffer } }],
    });

    this.shellBindGroup = this.device.createBindGroup({
      layout: shellBGL,
      entries: [{ binding: 0, resource: { buffer: this.shellParamsBuffer, size: SHELL_SLOT_SIZE } }],
    });
  }

  private writeShellParams() {
    const floatsPerSlot = SHELL_SLOT_SIZE / 4; // 64 floats per 256-byte slot
    const data = new Float32Array(SHELL_COUNT * floatsPerSlot);
    for (let i = 0; i < SHELL_COUNT; i++) {
      data[i * floatsPerSlot + 0] = SHELL_COUNT;
      data[i * floatsPerSlot + 1] = i;
      data[i * floatsPerSlot + 2] = this.furLength;
      data[i * floatsPerSlot + 3] = this.furDensity;
    }
    this.device.queue.writeBuffer(this.shellParamsBuffer, 0, data);
  }

  private createSphere(segments: number, rings: number, radius: number) {
    const verts: number[] = [];
    const inds: number[] = [];
    for (let y = 0; y <= rings; y++) {
      const phi = (y / rings) * Math.PI;
      for (let x = 0; x <= segments; x++) {
        const theta = (x / segments) * Math.PI * 2;
        const nx = Math.sin(phi) * Math.cos(theta);
        const ny = Math.cos(phi);
        const nz = Math.sin(phi) * Math.sin(theta);
        verts.push(nx * radius, ny * radius, nz * radius, nx, ny, nz, x / segments, y / rings);
      }
    }
    for (let y = 0; y < rings; y++) {
      for (let x = 0; x < segments; x++) {
        const a = y * (segments + 1) + x;
        const b = a + segments + 1;
        inds.push(a, a + 1, b, a + 1, b + 1, b);
      }
    }
    return { vertices: new Float32Array(verts), indices: new Uint16Array(inds) };
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const model = mat4.rotationY(time * 0.3);
    const invTrans = mat4.transpose(mat4.inverse(model));

    const d = this.sharedData;
    d.set(viewProj as unknown as ArrayLike<number>, 0);
    d.set(model as unknown as ArrayLike<number>, 16);
    d.set(invTrans as unknown as ArrayLike<number>, 32);
    d[48] = this.camera.position[0]; d[49] = this.camera.position[1]; d[50] = this.camera.position[2]; d[51] = 1;
    d[52] = -0.5; d[53] = -1.0; d[54] = -0.3; d[55] = 0;
    this.device.queue.writeBuffer(this.sharedUniformBuffer, 0, d as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });

    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.sharedBindGroup);
    renderPass.setVertexBuffer(0, this.vertexBuffer);
    renderPass.setIndexBuffer(this.indexBuffer, "uint16");

    for (let i = 0; i < SHELL_COUNT; i++) {
      renderPass.setBindGroup(1, this.shellBindGroup, [i * SHELL_SLOT_SIZE]);
      renderPass.drawIndexed(this.indexCount);
    }

    renderPass.end();
  }

  stats() {
    return {
      drawCalls: SHELL_COUNT,
      triangles: (this.indexCount / 3) * SHELL_COUNT,
      custom: {
        "Shell Layers": SHELL_COUNT,
        "Technique": "Shell Texturing",
        "Fur Pattern": "Procedural",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "furLength", 0.05, 0.5, 0.01).name("Fur Length").onChange(() => this.writeShellParams());
    gui.add(this, "furDensity", 10, 80, 1).name("Fur Density").onChange(() => this.writeShellParams());
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.sharedUniformBuffer.destroy();
    this.shellParamsBuffer.destroy();
  }
}
