import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";
import { Skeleton, type BoneDesc } from "../scene/skeleton";
import { Skinning } from "../scene/skinning";
import { AnimationPlayer } from "../scene/animation-player";
import { loadPMX, type PMXModel } from "../utils/pmx-loader";
import { mat4, quat, vec3 } from "wgpu-matrix";

const VS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
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
  let worldPos = (u.model * vec4<f32>(in.position, 1.0)).xyz;
  out.position = u.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldNormal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
  out.worldPos = worldPos;
  out.uv = in.uv;
  return out;
}
`;

const FS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var diffuseTex: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let L = normalize(u.lightDir.xyz);
  let NdotL = max(dot(N, L), 0.0);
  let texColor = textureSample(diffuseTex, texSampler, in.uv);
  let baseColor = texColor.rgb;
  let ambient = baseColor * 0.3;
  let diffuse = baseColor * NdotL;
  let color = ambient + diffuse * u.lightColor.rgb;
  return vec4<f32>(color, texColor.a);
}
`;

export class PMXDemo implements Demo {
  label = "PMX Viewer";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;

  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private uniformData = new Float32Array(80);
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private indexCount = 0;
  private use32bit = false;
  private diffuseTexture!: GPUTexture;
  private bindGroup!: GPUBindGroup;
  private depthTexture!: GPUTexture;

  private skeleton: Skeleton | null = null;
  private skinning: Skinning | null = null;
  private animPlayer: AnimationPlayer | null = null;

  private loaded = false;

  async init(ctx: GPUContext, camera: Camera, engine?: EngineContext): Promise<void> {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;

    try {
      const pmx = await loadPMX("/model.pmx");
      console.log(`[PMXDemo] Loaded: ${pmx.name}, vertices: ${pmx.vertices.length}, indices: ${pmx.indices.length}, materials: ${pmx.materials.length}, bones: ${pmx.bones.length}`);
      this.setupFromPMX(pmx);
    } catch (e) {
      console.error("[PMXDemo] Load failed:", e);
      this.createFallback();
    }
    this.loaded = true;
  }

  private setupFromPMX(pmx: PMXModel): void {
    const vertexCount = pmx.vertices.length;
    const stride = 8;
    const vertexData = new Float32Array(vertexCount * stride);

    for (let i = 0; i < vertexCount; i++) {
      const v = pmx.vertices[i];
      const off = i * stride;
      vertexData[off] = v.position[0];
      vertexData[off + 1] = v.position[1];
      vertexData[off + 2] = v.position[2];
      vertexData[off + 3] = v.normal[0];
      vertexData[off + 4] = v.normal[1];
      vertexData[off + 5] = v.normal[2];
      vertexData[off + 6] = v.uv[0];
      vertexData[off + 7] = v.uv[1];
    }

    this.vertexBuffer = this.device.createBuffer({
      label: "pmx-vb",
      size: vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertexData);
    this.vertexBuffer.unmap();

    this.use32bit = vertexCount > 65535;
    this.indexCount = pmx.indices.length;
    const indexSize = this.use32bit ? 4 : 2;
    this.indexBuffer = this.device.createBuffer({
      label: "pmx-ib",
      size: this.indexCount * indexSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    if (this.use32bit) {
      new Int32Array(this.indexBuffer.getMappedRange()).set(pmx.indices);
    } else {
      new Uint16Array(this.indexBuffer.getMappedRange()).set(pmx.indices);
    }
    this.indexBuffer.unmap();

    this.buildPipeline();
    this.buildResources();

    if (pmx.bones.length > 0) {
      const boneDescs: BoneDesc[] = pmx.bones.map((b) => ({
        name: b.name,
        parentIndex: b.parentIndex,
        position: vec3.create(b.position[0], b.position[1], b.position[2]),
        rotation: quat.identity(quat.create()),
        scale: vec3.create(1, 1, 1),
      }));
      this.skeleton = new Skeleton(boneDescs);

      const joints = new Uint16Array(vertexCount * 4);
      const weights = new Float32Array(vertexCount * 4);
      for (let i = 0; i < vertexCount; i++) {
        const v = pmx.vertices[i];
        for (let j = 0; j < 4; j++) {
          joints[i * 4 + j] = v.boneIndices.length > j ? v.boneIndices[j] : 0;
          weights[i * 4 + j] = v.boneWeights.length > j ? v.boneWeights[j] : 0;
        }
      }
      this.skinning = new Skinning(vertexCount, 4, joints, weights, pmx.bones.length);
      this.skinning.createGPUResources(this.device, "pmx-skin");
      this.animPlayer = new AnimationPlayer(this.skeleton, pmx.morphs.length);
    }
  }

  private createFallback(): void {
    const verts = new Float32Array([
      -0.5,-0.5,0, 0,0,1, 0,0,
       0.5,-0.5,0, 0,0,1, 1,0,
       0.5, 0.5,0, 0,0,1, 1,1,
      -0.5, 0.5,0, 0,0,1, 0,1,
    ]);
    const indices = new Uint16Array([0,1,2, 0,2,3]);
    this.indexCount = 6;

    this.vertexBuffer = this.device.createBuffer({
      label: "fallback-vb", size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(verts);
    this.vertexBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: "fallback-ib", size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    this.buildPipeline();
    this.buildResources();
  }

  private buildPipeline(): void {
    const vsModule = this.device.createShaderModule({ code: VS });
    const fsModule = this.device.createShaderModule({ code: FS });

    this.pipeline = this.device.createRenderPipeline({
      label: "pmx-render",
      layout: "auto",
      vertex: {
        module: vsModule, entryPoint: "vs_main",
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
        module: fsModule, entryPoint: "fs_main",
        targets: [{ format: this.ctx.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
  }

  private buildResources(): void {
    this.uniformBuffer = this.device.createBuffer({
      label: "pmx-ubo", size: 320,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const data = new Uint8Array([255, 255, 255, 255]);
    this.diffuseTexture = this.device.createTexture({
      label: "default-diffuse",
      size: [1, 1], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture({ texture: this.diffuseTexture }, data, { bytesPerRow: 4 }, [1, 1]);

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.diffuseTexture.createView() },
        { binding: 2, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
      ],
    });

    this.depthTexture = this.device.createTexture({
      label: "pmx-depth",
      size: [this.ctx.canvas.width, this.ctx.canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private ensureDepth(): void {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      label: "pmx-depth", size: [w, h], format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  update(time: number, deltaTime: number): void {
    if (!this.loaded) return;

    if (this.animPlayer) {
      this.animPlayer.update(deltaTime);
      this.skeleton!.updateWorldMatrices();
      this.skeleton!.computeSkinMatrices(this.skinning!.skinMatrixData);
      this.skinning!.modified = true;
      this.skinning!.flushToDevice(this.device);
    }

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const model = mat4.identity(mat4.create());

    this.uniformData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.uniformData.set(model as unknown as ArrayLike<number>, 16);
    this.uniformData[32] = -0.5; this.uniformData[33] = -1.0; this.uniformData[34] = -0.3; this.uniformData[35] = 0;
    this.uniformData[36] = 3.0; this.uniformData[37] = 3.0; this.uniformData[38] = 3.0; this.uniformData[39] = 0;
    this.uniformData[40] = this.camera.position[0]; this.uniformData[41] = this.camera.position[1]; this.uniformData[42] = this.camera.position[2]; this.uniformData[43] = 0;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        if (!this.loaded) return;
        this.ensureDepth();

        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
          depthStencilAttachment: {
            view: this.depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");
        pass.drawIndexed(this.indexCount);
        pass.end();
      },
    }];
  }

  destroy(): void {
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.diffuseTexture?.destroy();
    this.depthTexture?.destroy();
    this.skinning?.destroy();
  }
}
