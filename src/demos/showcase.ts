import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import { PassManager } from "../passes/render-target";
import { BloomPass } from "../passes/bloom";
import { PostProcessPass } from "../passes/post-process";
import { createCubeGeometry } from "../utils/geometry";
import { mat4 } from "wgpu-matrix";
import type { EngineContext } from "../core/engine";

const sceneVS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  time: f32,
  metallic: f32,
  roughness: f32,
  emissive: f32,
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
`;

const sceneFS = `
#include "pbr"

struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  time: f32,
  metallic: f32,
  roughness: f32,
  emissive: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);
  let L = normalize(-u.lightDir.xyz);

  let baseColor = vec3<f32>(
    0.5 + 0.5 * sin(u.time * 0.3 + in.uv.x * 6.28),
    0.5 + 0.5 * cos(u.time * 0.5 + in.uv.y * 6.28),
    0.7
  );

  var Lo = cookTorrance(N, V, L, baseColor, u.metallic, u.roughness) * u.lightColor.rgb;

  let ambient = vec3<f32>(0.03) * baseColor;
  var color = ambient + Lo;

  let emissiveMask = smoothstep(0.7, 0.9, sin(in.uv.x * 12.56 + u.time * 2.0) * sin(in.uv.y * 12.56 - u.time));
  color += baseColor * emissiveMask * u.emissive * 3.0;

  return vec4<f32>(color, 1.0);
}
`;

export class ShowcaseDemo implements Demo {
  label = "Showcase";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private engine!: EngineContext;
  private camera!: Camera;
  private passManager!: PassManager;
  private bloomPass!: BloomPass;
  private postProcessPass!: PostProcessPass;
  private scenePipeline!: GPURenderPipeline;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;
  private indexCount = 0;
  private mipTargets: GPUTexture[] = [];

  private vsCode = sceneVS;
  private fsCode = sceneFS;

  bloomIntensity = 0.6;
  emissive = 2.0;

  init(ctx: GPUContext, camera: Camera, engine?: EngineContext) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.engine = engine!;

    this.passManager = new PassManager(ctx.device, ctx.format);
    this.bloomPass = new BloomPass(ctx.device, "rgba16float");
    this.postProcessPass = new PostProcessPass(ctx.device, ctx.format);

    const { vertices, indices } = createCubeGeometry();
    this.indexCount = indices.length;

    this.vertexBuffer = this.device.createBuffer({
      label: "showcase-vb",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: "showcase-ib",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.uniformBuffer = this.device.createBuffer({
      label: "showcase-ubo",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.buildPipeline();

    this.bindGroup = this.device.createBindGroup({
      layout: this.scenePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });
  }

  private buildPipeline(): boolean {
    try {
      const vertexLayout: GPUVertexBufferLayout = {
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" },
          { shaderLocation: 1, offset: 12, format: "float32x3" },
          { shaderLocation: 2, offset: 24, format: "float32x2" },
        ],
      };

      const vsModule = this.engine.modules.resolveAndCompile(
        this.device,
        "showcase-vs",
        this.vsCode
      );
      const fsModule = this.engine.modules.resolveAndCompile(
        this.device,
        "showcase-fs",
        this.fsCode
      );

      this.scenePipeline = this.device.createRenderPipeline({
        label: "showcase-scene",
        layout: "auto",
        vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
        fragment: {
          module: fsModule,
          entryPoint: "fs_main",
          targets: [{ format: "rgba16float" }],
        },
        primitive: { topology: "triangle-list", cullMode: "back" },
        depthStencil: {
          format: "depth24plus",
          depthWriteEnabled: true,
          depthCompare: "less",
        },
      });

      this.bindGroup = this.device.createBindGroup({
        layout: this.scenePipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
      });

      return true;
    } catch (e) {
      console.error("[Showcase] Pipeline build failed:", e);
      return false;
    }
  }

  getShaderStages(): ShaderStageDesc[] {
    return [
      { label: "Showcase / Vertex", type: "vertex", code: this.vsCode },
      { label: "Showcase / Fragment", type: "fragment", code: this.fsCode },
    ];
  }

  onShaderReload(stageLabel: string, code: string): boolean {
    if (stageLabel === "Showcase / Vertex") {
      this.vsCode = code;
    } else if (stageLabel === "Showcase / Fragment") {
      this.fsCode = code;
    }
    return this.buildPipeline();
  }

  private ensureMipTargets() {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (this.mipTargets.length > 0 && this.mipTargets[0].width === Math.floor(w / 2)) return;

    for (const t of this.mipTargets) t.destroy();
    this.mipTargets = [];

    let mw = Math.floor(w / 2);
    let mh = Math.floor(h / 2);
    for (let i = 0; i < 5; i++) {
      this.mipTargets.push(
        this.device.createTexture({
          label: `bloom-mip-${i}`,
          size: [Math.max(1, mw), Math.max(1, mh)],
          format: "rgba16float",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
      );
      mw = Math.floor(mw / 2);
      mh = Math.floor(mh / 2);
    }
  }

  private uboData = new Float32Array(64);

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);

    const model = mat4.mul(
      mat4.rotationY(time * 0.5),
      mat4.scaling([1.5, 1.5, 1.5])
    );
    const invTransModel = mat4.transpose(mat4.inverse(model));

    const ubo = this.uboData;
    ubo.set(viewProj as unknown as ArrayLike<number>, 0);
    ubo.set(model as unknown as ArrayLike<number>, 16);
    ubo.set(invTransModel as unknown as ArrayLike<number>, 32);
    ubo[48] = this.camera.position[0];
    ubo[49] = this.camera.position[1];
    ubo[50] = this.camera.position[2];
    ubo[51] = 1.0;
    ubo[52] = -0.5;
    ubo[53] = -1.0;
    ubo[54] = -0.3;
    ubo[55] = 0.0;
    ubo[56] = 3.0;
    ubo[57] = 3.0;
    ubo[58] = 3.0;
    ubo[59] = 1.0;
    ubo[60] = time;
    ubo[61] = 0.9;
    ubo[62] = 0.25;
    ubo[63] = this.emissive;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);
    this.bloomPass.bloomIntensity = this.bloomIntensity;
  }

  render(encoder: GPUCommandEncoder, screenView: GPUTextureView) {
    this.passManager.resize(this.ctx.canvas.width, this.ctx.canvas.height);
    this.ensureMipTargets();

    const sceneRT = this.passManager.getOrCreateTarget("scene-color", "rgba16float");
    const depthTarget = this.passManager.getOrCreateDepth("scene-depth");
    const bloomRT = this.passManager.getOrCreateTarget("bloom-combined", "rgba16float");

    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: sceneRT.view,
            clearValue: { r: 0.01, g: 0.01, b: 0.02, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        depthStencilAttachment: {
          view: depthTarget.view,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      });
      pass.setPipeline(this.scenePipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.setVertexBuffer(0, this.vertexBuffer);
      pass.setIndexBuffer(this.indexBuffer, "uint16");
      pass.drawIndexed(this.indexCount);
      pass.end();
    }

    this.bloomPass.execute(encoder, sceneRT.texture, this.mipTargets, bloomRT.view);

    this.postProcessPass.execute(
      encoder,
      bloomRT.texture,
      depthTarget.texture,
      screenView,
      [this.ctx.canvas.width, this.ctx.canvas.height],
      performance.now() / 1000
    );
  }

  stats() {
    return {
      drawCalls: 1 + 5 * 2 + 1 + 1,
      triangles: this.indexCount / 3,
      custom: {
        "Bloom Mips": 5,
        "Post FX": "ACES+CA+Fog+Vignette",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "bloomIntensity", 0, 3, 0.01).name("Bloom Intensity");
    gui.add(this, "emissive", 0, 5, 0.1).name("Emissive");
    const pp = this.postProcessPass.params;
    const fxFolder = gui.addFolder("Post Process");
    fxFolder.add(pp, "exposure", 0.1, 3, 0.01).name("Exposure");
    fxFolder.add(pp, "chromaticStrength", 0, 0.02, 0.001).name("Chromatic Aberr.");
    fxFolder.add(pp, "fogDensity", 0, 0.1, 0.001).name("Fog Density");
    fxFolder.add(pp, "vignetteStrength", 0, 1, 0.01).name("Vignette");
    fxFolder.add(pp, "saturation", 0, 2, 0.01).name("Saturation");
  }

  destroy() {
    for (const t of this.mipTargets) t.destroy();
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.uniformBuffer.destroy();
    this.passManager.destroy();
    this.bloomPass.destroy();
    this.postProcessPass.destroy();
  }
}
