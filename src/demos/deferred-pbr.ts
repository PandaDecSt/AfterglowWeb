import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import { PassManager } from "../passes/render-target";
import { GBuffer, GBUFFER_GEOMETRY_WGSL } from "../passes/gbuffer";
import { DeferredLightingPass } from "../passes/deferred-lighting";
import { CascadedShadowMap, CSM_CASCADE_COUNT } from "../passes/csm";
import { GTAOPass } from "../passes/ssao";
import { BloomPass } from "../passes/bloom";
import { PostProcessPass } from "../passes/post-process";
import { TAAPass } from "../passes/taa";
import { LightScene, createDirectionalLight, createPointLight, createSpotLight } from "../scene/light";
import { EnvironmentMap } from "../passes/environment";
import { BrdfLut } from "../passes/brdf-lut";
import { createCubeGeometry, createSphereGeometry } from "../utils/geometry";
import { mat4, vec3, vec4, type Mat4 } from "wgpu-matrix";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";

const gbufferVS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  prevModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) prevClipXY: vec2<f32>,
  @location(4) prevClipW: f32,
  @location(5) curClipXY: vec2<f32>,
  @location(6) curClipW: f32,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  let clip = u.viewProj * worldPos;
  out.position = clip;
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  let prevWorld = u.prevModel * vec4<f32>(pos, 1.0);
  let prevClip = u.prevViewProj * prevWorld;
  out.prevClipXY = prevClip.xy;
  out.prevClipW = prevClip.w;
  out.curClipXY = clip.xy;
  out.curClipW = clip.w;
  return out;
}
`;

const gbufferFS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  prevModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) prevClipXY: vec2<f32>,
  @location(4) prevClipW: f32,
  @location(5) curClipXY: vec2<f32>,
  @location(6) curClipW: f32,
};

struct GBufferOutput {
  @location(0) albedo: vec4<f32>,
  @location(1) normal: vec4<f32>,
  @location(2) material: vec4<f32>,
  @location(3) motion: vec2<f32>,
  @location(4) depthCopy: vec4<f32>,
};

@fragment
fn fs_main(in: VSOut) -> GBufferOutput {
  var out: GBufferOutput;

  let baseColor = vec3<f32>(
    0.5 + 0.5 * sin(u.params.x * 0.3 + in.uv.x * 6.28),
    0.5 + 0.5 * cos(u.params.x * 0.5 + in.uv.y * 6.28),
    0.7
  );

  out.albedo = vec4<f32>(baseColor, 1.0);
  out.normal = vec4<f32>(normalize(in.worldNormal), 0.0);
  out.material = vec4<f32>(u.params.y, u.params.z, 0.0, 0.0);

  // perspective-correct NDC: clip.xy and clip.w are interpolated separately
  // note: NDC.y grows upward but texture uv.y grows downward, so flip y
  let prevNDC = in.prevClipXY / in.prevClipW;
  let curNDC = in.curClipXY / in.curClipW;
  let motion = vec2<f32>(
    (curNDC.x - prevNDC.x) * 0.5,
    (prevNDC.y - curNDC.y) * 0.5,
  );
  out.motion = motion;

  out.depthCopy = vec4<f32>(in.position.z, 0.0, 0.0, 0.0);

  return out;
}
`;

const shadowVS = `
struct Uniforms {
  lightVP: mat4x4<f32>,
  model: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
) -> @builtin(position) vec4<f32> {
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  return u.lightVP * worldPos;
}
`;

const gizmoVS = `
struct GizmoOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec4<f32>,
  @location(1) color: vec4<f32>,
) -> GizmoOut {
  var out: GizmoOut;
  out.position = pos;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: GizmoOut) -> @location(0) vec4<f32> {
  return in.color;
}
`;

export class DeferredDemo implements Demo {
  label = "Deferred PBR";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private engine!: EngineContext;
  private camera!: Camera;
  private passManager!: PassManager;

  private gbuffer!: GBuffer;
  private deferredLighting!: DeferredLightingPass;
  private csm!: CascadedShadowMap;
  private gtao!: GTAOPass;
  private bloomPass!: BloomPass;
  private postProcessPass!: PostProcessPass;
  private taaPass!: TAAPass;
  private lightScene!: LightScene;
  private envMap!: EnvironmentMap;
  private brdfLut!: BrdfLut;

  private gbufferPipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;

  private cubeVB!: GPUBuffer;
  private cubeIB!: GPUBuffer;
  private cubeIndexCount = 0;
  private sphereVB!: GPUBuffer;
  private sphereIB!: GPUBuffer;
  private sphereIndexCount = 0;

  private sceneUBO!: GPUBuffer;
  private ballUBO!: GPUBuffer;
  private shadowUBO!: GPUBuffer;
  private dummyDepthTexture!: GPUTexture;

  private prevViewProj: Float32Array = new Float32Array(16);
  private prevModel: Float32Array = new Float32Array(16);
  private prevBallModel: Float32Array = new Float32Array(16);

  metallic = 0.1;
  roughness = 0.5;
  bloomIntensity = 0.3;
  useCSM = true;
  useSSAO = true;
  useTAA = true;

  private vsCode = gbufferVS;
  private fsCode = gbufferFS;

  private cubePos: [number, number, number] = [0, 0, 0];
  private ballPos: [number, number, number] = [2.6, 0, 0];
  private ballModel = new Float32Array(16);
  private selectedId = 0;
  private editMode = false;
  private dragAxis = 0;
  private dragging = false;
  private dragLastX = 0;
  private dragLastY = 0;

  private gizmoVB!: GPUBuffer;
  private gizmoPipeline!: GPURenderPipeline;
  private gizmoData = new Float32Array(6 * 8);
  private gizmoNDC = new Float32Array(6 * 2);

  init(ctx: GPUContext, camera: Camera, engine?: EngineContext) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.engine = engine!;

    this.passManager = new PassManager(ctx.device, ctx.format);
    this.gbuffer = new GBuffer(ctx.device);
    this.lightScene = new LightScene();
    this.lightScene.ambientColor = [0.15, 0.15, 0.18];
    this.lightScene.ambientIntensity = 1.0;

    const dirLight = createDirectionalLight([-0.5, -1.0, -0.3], [1, 1, 1], 3.0);
    this.lightScene.addLight(dirLight);
    this.lightScene.addLight(createPointLight([3, 2, 3], [1, 0.8, 0.5], 8.0, 15.0));
    this.lightScene.addLight(createPointLight([-3, 2, -2], [0.3, 0.5, 1.0], 6.0, 12.0));
    this.lightScene.addLight(createSpotLight([0, 5, 0], [0, -1, 0], [1, 0.9, 0.7], 12.0, 20.0));

    this.deferredLighting = new DeferredLightingPass(ctx.device, this.lightScene, "rgba16float");
    this.csm = new CascadedShadowMap(ctx.device, 2048);
    this.gtao = new GTAOPass(ctx.device);
    this.envMap = new EnvironmentMap();
    this.envMap.bake(ctx.device);
    this.brdfLut = new BrdfLut();
    this.brdfLut.bake(ctx.device);
    this.bloomPass = new BloomPass(ctx.device, "rgba16float");
    this.bloomPass.threshold = 0.3;
    this.bloomPass.knee = 0.7;
    this.bloomPass.radius = 2.5;
    this.postProcessPass = new PostProcessPass(ctx.device, ctx.format);
    this.taaPass = new TAAPass(ctx.device, "rgba16float");

    this.createGeometry();
    this.buildPipelines();

    this.sceneUBO = this.device.createBuffer({
      label: "deferred-scene-ubo",
      size: 352,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.ballUBO = this.device.createBuffer({
      label: "deferred-ball-ubo",
      size: 352,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.shadowUBO = this.device.createBuffer({
      label: "deferred-shadow-ubo",
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    mat4.identity(this.prevViewProj);
    mat4.identity(this.prevModel);
    mat4.identity(this.prevBallModel);

    this.dummyDepthTexture = this.device.createTexture({
      label: "deferred-dummy-depth",
      size: [1, 1],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.gizmoVB = this.device.createBuffer({
      label: "gizmo-vb",
      size: 6 * 8 * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const cv = ctx.canvas;
    cv.addEventListener("pointerdown", this.onPointerDown, true);
    cv.addEventListener("pointermove", this.onPointerMove, true);
    cv.addEventListener("pointerup", this.onPointerUp, true);
    cv.addEventListener("pointercancel", this.onPointerUp, true);
  }

  private createGeometry(): void {
    const cube = createCubeGeometry();
    this.cubeIndexCount = cube.indices.length;
    this.cubeVB = this.device.createBuffer({
      label: "deferred-cube-vb",
      size: cube.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.cubeVB.getMappedRange()).set(cube.vertices);
    this.cubeVB.unmap();
    this.cubeIB = this.device.createBuffer({
      label: "deferred-cube-ib",
      size: cube.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.cubeIB.getMappedRange()).set(cube.indices);
    this.cubeIB.unmap();

    const sphere = createSphereGeometry(1.0, 32, 32);
    this.sphereIndexCount = sphere.indices.length;
    this.sphereVB = this.device.createBuffer({
      label: "deferred-sphere-vb",
      size: sphere.vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.sphereVB.getMappedRange()).set(sphere.vertices);
    this.sphereVB.unmap();
    this.sphereIB = this.device.createBuffer({
      label: "deferred-sphere-ib",
      size: sphere.indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.sphereIB.getMappedRange()).set(sphere.indices);
    this.sphereIB.unmap();
  }

  private buildPipelines(): void {
    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const vsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-vs", this.vsCode);
    const fsModule = this.engine.modules.resolveAndCompile(this.device, "deferred-gbuffer-fs", this.fsCode);

    this.gbufferPipeline = this.device.createRenderPipeline({
      label: "deferred-gbuffer",
      layout: "auto",
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: {
        module: fsModule,
        entryPoint: "fs_main",
        targets: [
          { format: GBuffer.ALBEDO_FORMAT },
          { format: GBuffer.NORMAL_FORMAT },
          { format: GBuffer.MATERIAL_FORMAT },
          { format: GBuffer.MOTION_FORMAT },
          { format: GBuffer.DEPTH_COPY_FORMAT },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: {
        format: GBuffer.DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const shadowVsModule = this.device.createShaderModule({ code: shadowVS });
    this.shadowPipeline = this.device.createRenderPipeline({
      label: "deferred-shadow",
      layout: "auto",
      vertex: { module: shadowVsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    const gizmoModule = this.device.createShaderModule({ code: gizmoVS });
    this.gizmoPipeline = this.device.createRenderPipeline({
      label: "gizmo",
      layout: "auto",
      vertex: {
        module: gizmoModule,
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 8 * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x4" },
            { shaderLocation: 1, offset: 16, format: "float32x4" },
          ],
        }],
      },
      fragment: { module: gizmoModule, entryPoint: "fs_main", targets: [{ format: this.ctx.format }] },
      primitive: { topology: "line-list" },
    });
  }

  getShaderStages(): ShaderStageDesc[] {
    return [
      { label: "Deferred / GBuffer VS", type: "vertex", code: this.vsCode },
      { label: "Deferred / GBuffer FS", type: "fragment", code: this.fsCode },
    ];
  }

  onShaderReload(stageLabel: string, code: string): boolean {
    if (stageLabel === "Deferred / GBuffer VS") this.vsCode = code;
    else if (stageLabel === "Deferred / GBuffer FS") this.fsCode = code;
    try { this.buildPipelines(); return true; } catch { return false; }
  }

  private uboData = new Float32Array(88);
  private ballUboData = new Float32Array(88);
  private shadowUboData = new Float32Array(32);
  private jitteredVP = new Float32Array(16);
  private cubeModel = new Float32Array(16);
  private shadowCubeBG: GPUBindGroup | null = null;
  private shadowBallBG: GPUBindGroup | null = null;
  private gbufferCubeBG: GPUBindGroup | null = null;
  private gbufferBallBG: GPUBindGroup | null = null;
  private frameTime = 0;
  private taaFrameIndex = 0;
  private taaJitter = true;
  private pauseAnimation = false;
  private frameViewProj = new Float32Array(16);
  private frameInvViewProj = new Float32Array(16);

  private halton(index: number, base: number): number {
    let result = 0;
    let f = 1 / base;
    let i = index;
    while (i > 0) {
      result += f * (i % base);
      i = Math.floor(i / base);
      f /= base;
    }
    return result;
  }

  private applyJitter(viewProj: Float32Array, index: number, w: number, h: number): Float32Array {
    // 0.5px jitter in NDC space
    const jx = (this.halton(index, 2) - 0.5) * 2.0 / w;
    const jy = (this.halton(index, 3) - 0.5) * 2.0 / h;
    this.jitteredVP.set(viewProj as unknown as ArrayLike<number>);
    // this assumes column-major wgpu-matrix: proj[2][0]=8, proj[2][1]=9
    this.jitteredVP[8] += jx;
    this.jitteredVP[9] += jy;
    return this.jitteredVP;
  }

  update(time: number) {
    if (this.pauseAnimation) time = 0;
    this.frameTime = time;
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const baseVP = this.camera.getViewProjectionMatrix(aspect) as unknown as Float32Array;
    const viewProj = this.useTAA && this.taaJitter ? this.applyJitter(baseVP, this.taaFrameIndex, w, h) : baseVP;

    const ubo = this.uboData;
    ubo.set(viewProj as unknown as ArrayLike<number>, 0);
    ubo.set(this.prevViewProj as unknown as ArrayLike<number>, 16);
    const model = mat4.mul(
      mat4.translation(this.cubePos),
      mat4.mul(mat4.rotationY(time * 0.5), mat4.scaling([1.5, 1.5, 1.5])),
    );
    const invTransModel = mat4.transpose(mat4.inverse(model));
    this.cubeModel.set(model as unknown as ArrayLike<number>);
    ubo.set(model as unknown as ArrayLike<number>, 32);
    ubo.set(invTransModel as unknown as ArrayLike<number>, 48);
    ubo.set(this.prevModel as unknown as ArrayLike<number>, 64);
    ubo[80] = this.camera.position[0];
    ubo[81] = this.camera.position[1];
    ubo[82] = this.camera.position[2];
    ubo[83] = 1.0;
    ubo[84] = time;
    ubo[85] = this.metallic;
    ubo[86] = this.roughness;
    ubo[87] = 0;

    this.device.queue.writeBuffer(this.sceneUBO, 0, ubo as unknown as GPUAllowSharedBufferSource);
    this.prevViewProj.set(viewProj as unknown as ArrayLike<number>);
    this.prevModel.set(model as unknown as ArrayLike<number>);

    // Ball model: offset to the side, slowly rotating
    const ballModel = mat4.mul(
      mat4.translation(this.ballPos),
      mat4.mul(mat4.rotationY(time * 0.4), mat4.scaling([1.0, 1.0, 1.0])),
    );
    const invTransBall = mat4.transpose(mat4.inverse(ballModel));
    this.ballModel.set(ballModel as unknown as ArrayLike<number>);
    const ballUbo = this.ballUboData;
    ballUbo.set(viewProj as unknown as ArrayLike<number>, 0);
    ballUbo.set(this.prevViewProj as unknown as ArrayLike<number>, 16);
    ballUbo.set(ballModel as unknown as ArrayLike<number>, 32);
    ballUbo.set(invTransBall as unknown as ArrayLike<number>, 48);
    ballUbo.set(this.prevBallModel as unknown as ArrayLike<number>, 64);
    ballUbo[80] = this.camera.position[0];
    ballUbo[81] = this.camera.position[1];
    ballUbo[82] = this.camera.position[2];
    ballUbo[83] = 1.0;
    ballUbo[84] = time;
    ballUbo[85] = this.metallic;
    ballUbo[86] = this.roughness;
    ballUbo[87] = 0;

    this.device.queue.writeBuffer(this.ballUBO, 0, ballUbo as unknown as GPUAllowSharedBufferSource);
    this.prevBallModel.set(ballModel as unknown as ArrayLike<number>);

    this.bloomPass.bloomIntensity = this.bloomIntensity;

    const invViewProj = mat4.inverse(viewProj);
    this.frameViewProj.set(viewProj as unknown as ArrayLike<number>);
    this.frameInvViewProj.set(invViewProj as unknown as ArrayLike<number>);
    this.postProcessPass.cameraPos = [this.camera.position[0], this.camera.position[1], this.camera.position[2]];
    this.postProcessPass.invVP.set(invViewProj as unknown as ArrayLike<number>);

    if (this.useCSM) {
      this.csm.updateCascadeVPs(
        viewProj as Mat4,
        invViewProj as Mat4,
        0.1, 50.0,
      );
    }

    this.taaFrameIndex++;
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, screenView: GPUTextureView) => {
        const w = this.ctx.canvas.width;
        const h = this.ctx.canvas.height;
        this.gbuffer.resize(w, h);
        this.passManager.resize(w, h);

        // Step 1: CSM shadow passes
        if (this.useCSM) {
          for (let i = 0; i < CSM_CASCADE_COUNT; i++) {
            const shadowPass = this.csm.beginCascadePass(encoder, i);
            shadowPass.setPipeline(this.shadowPipeline);

            const shadowUbo = this.shadowUboData;
            shadowUbo.set(this.csm.cascadeVPs[i] as unknown as ArrayLike<number>, 0);
            shadowUbo.set(this.cubeModel as unknown as ArrayLike<number>, 16);
            this.device.queue.writeBuffer(this.shadowUBO, 0, shadowUbo as unknown as GPUAllowSharedBufferSource);

            if (!this.shadowCubeBG) {
              this.shadowCubeBG = this.device.createBindGroup({
                layout: this.shadowPipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
              });
            }
            shadowPass.setBindGroup(0, this.shadowCubeBG);
            shadowPass.setVertexBuffer(0, this.cubeVB);
            shadowPass.setIndexBuffer(this.cubeIB, "uint16");
            shadowPass.drawIndexed(this.cubeIndexCount);

            shadowUbo.set(this.csm.cascadeVPs[i] as unknown as ArrayLike<number>, 0);
            shadowUbo.set(this.ballUboData.subarray(32, 48) as unknown as ArrayLike<number>, 16);
            this.device.queue.writeBuffer(this.shadowUBO, 0, shadowUbo as unknown as GPUAllowSharedBufferSource);

            if (!this.shadowBallBG) {
              this.shadowBallBG = this.device.createBindGroup({
                layout: this.shadowPipeline.getBindGroupLayout(0),
                entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
              });
            }
            shadowPass.setBindGroup(0, this.shadowBallBG);
            shadowPass.setVertexBuffer(0, this.sphereVB);
            shadowPass.setIndexBuffer(this.sphereIB, "uint16");
            shadowPass.drawIndexed(this.sphereIndexCount);
            shadowPass.end();
          }
        }

        // Step 2: GBuffer pass
        {
          const gbufferPass = this.gbuffer.beginGBufferPass(encoder);
          gbufferPass.setPipeline(this.gbufferPipeline);
          if (!this.gbufferCubeBG) {
            this.gbufferCubeBG = this.device.createBindGroup({
              layout: this.gbufferPipeline.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: { buffer: this.sceneUBO } }],
            });
          }
          gbufferPass.setBindGroup(0, this.gbufferCubeBG);
          gbufferPass.setVertexBuffer(0, this.cubeVB);
          gbufferPass.setIndexBuffer(this.cubeIB, "uint16");
          gbufferPass.drawIndexed(this.cubeIndexCount);

          if (!this.gbufferBallBG) {
            this.gbufferBallBG = this.device.createBindGroup({
              layout: this.gbufferPipeline.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: { buffer: this.ballUBO } }],
            });
          }
          gbufferPass.setBindGroup(0, this.gbufferBallBG);
          gbufferPass.setVertexBuffer(0, this.sphereVB);
          gbufferPass.setIndexBuffer(this.sphereIB, "uint16");
          gbufferPass.end();
        }

        // Step 3: SSAO pass
        if (this.useSSAO) {
          this.gtao.execute(encoder, this.gbuffer, this.frameViewProj as unknown as Float32Array);
        }

        // Step 4: Lighting pass
        const lightingRT = this.passManager.getOrCreateTarget("deferred-lighting", "rgba16float");
        {
          this.deferredLighting.update(
            this.frameViewProj as Mat4,
            [this.camera.position[0], this.camera.position[1], this.camera.position[2]],
            this.frameInvViewProj as Mat4,
            w, h, 0.1, 50.0,
          );
          this.deferredLighting.execute(
            encoder,
            this.gbuffer,
            lightingRT.view,
            this.useCSM ? this.csm.views[0] : undefined,
            this.useCSM ? this.csm.sampler : undefined,
            this.useCSM ? this.csm.ubo : undefined,
            this.useSSAO ? this.gtao.view : undefined,
            {
              irradiance: this.envMap.irradianceView,
              prefilter: this.envMap.prefilterView,
              brdfLut: this.brdfLut.view,
            },
          );
        }

        // Step 5: TAA (temporal accumulation; skip when disabled or when history is stale)
        let taaResultTex = lightingRT.texture;
        if (this.useTAA) {
          const taaResult = this.taaPass.execute(encoder, lightingRT.texture, this.gbuffer.motionView, w, h);
          taaResultTex = taaResult.texture;
        } else {
          this.taaPass.reset();
        }

        // Step 6: Bloom (extract bright pixels, downsample/upsample pyramid)
        const bloomResult = this.bloomPass.execute(encoder, taaResultTex);
        const bloomCombineRT = this.passManager.getOrCreateTarget("bloom-combine", "rgba16float");
        this.bloomPass.combine(
          encoder,
          taaResultTex,
          bloomResult.view,
          bloomCombineRT.view,
          this.bloomIntensity,
        );

        // Step 7: Post-process (tonemap, saturation, vignette, chromatic) → screen
        this.postProcessPass.execute(
          encoder,
          bloomCombineRT.texture,
          this.gbuffer.depthSampledView,
          screenView,
          [w, h],
          this.frameTime,
        );

        // Step 8: Gizmo (selection axes) drawn over the final image
        if (this.selectedId > 0) {
          this.buildGizmoData();
          this.device.queue.writeBuffer(this.gizmoVB, 0, this.gizmoData as unknown as GPUAllowSharedBufferSource);
          const gizmoPass = encoder.beginRenderPass({
            label: "gizmo",
            colorAttachments: [{
              view: screenView,
              clearValue: { r: 0, g: 0, b: 0, a: 1 },
              loadOp: "load",
              storeOp: "store",
            }],
          });
          gizmoPass.setPipeline(this.gizmoPipeline);
          gizmoPass.setVertexBuffer(0, this.gizmoVB);
          gizmoPass.draw(6);
          gizmoPass.end();
        }
      },
    }];
  }

  stats() {
    return {
      drawCalls: 2 + CSM_CASCADE_COUNT,
      triangles: this.cubeIndexCount / 3,
      custom: {
        "Pipeline": "Deferred (G-Buffer MRT4)",
        "Lights": `${this.lightScene.count} (${this.lightScene.directionalLights.length}D + ${this.lightScene.pointLights.length}P + ${this.lightScene.spotLights.length}S)`,
        "CSM": this.useCSM ? `${CSM_CASCADE_COUNT} cascades` : "Off",
        "SSAO": this.useSSAO ? "GTAO" : "Off",
      },
    };
  }

  private gizmoAxes: [number, number, number][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  private gizmoColors: [number, number, number, number][] = [
    [1, 0.25, 0.25, 1],
    [0.3, 1, 0.3, 1],
    [0.35, 0.45, 1, 1],
  ];
  private GIZMO_LEN = 0.6;

  private buildGizmoData(): void {
    const pos = this.selectedId === 1 ? this.cubePos : this.ballPos;
    const d = this.gizmoData;
    const ndc = this.gizmoNDC;
    for (let a = 0; a < 3; a++) {
      const o = [pos[0], pos[1], pos[2], 1] as const;
      const tip = [
        pos[0] + this.gizmoAxes[a][0] * this.GIZMO_LEN,
        pos[1] + this.gizmoAxes[a][1] * this.GIZMO_LEN,
        pos[2] + this.gizmoAxes[a][2] * this.GIZMO_LEN,
        1,
      ] as const;
      const c0 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(o[0], o[1], o[2], o[3]));
      const c1 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(tip[0], tip[1], tip[2], tip[3]));
      const base = a * 16;
      d[base] = c0[0] / c0[3]; d[base + 1] = c0[1] / c0[3]; d[base + 2] = c0[2] / c0[3]; d[base + 3] = 1;
      d[base + 4] = this.gizmoColors[a][0];
      d[base + 5] = this.gizmoColors[a][1];
      d[base + 6] = this.gizmoColors[a][2];
      d[base + 7] = this.gizmoColors[a][3];
      d[base + 8] = c1[0] / c1[3]; d[base + 9] = c1[1] / c1[3]; d[base + 10] = c1[2] / c1[3]; d[base + 11] = 1;
      d[base + 12] = this.gizmoColors[a][0];
      d[base + 13] = this.gizmoColors[a][1];
      d[base + 14] = this.gizmoColors[a][2];
      d[base + 15] = this.gizmoColors[a][3];
      ndc[a * 4] = c0[0] / c0[3]; ndc[a * 4 + 1] = c0[1] / c0[3];
      ndc[a * 4 + 2] = c1[0] / c1[3]; ndc[a * 4 + 3] = c1[1] / c1[3];
    }
  }

  private eventToNDC(e: PointerEvent): [number, number] {
    const rect = this.ctx.canvas.getBoundingClientRect();
    return [
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -(((e.clientY - rect.top) / rect.height) * 2 - 1),
    ];
  }

  private distToSegment(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
    const abx = bx - ax, aby = by - ay;
    const len2 = abx * abx + aby * aby;
    let t = len2 > 0 ? ((px - ax) * abx + (py - ay) * aby) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const dx = px - (ax + abx * t);
    const dy = py - (ay + aby * t);
    return Math.hypot(dx, dy);
  }

  private hitGizmoAxis(ndc: [number, number]): number {
    if (this.selectedId <= 0) return 0;
    const thr = 10 / (this.ctx.canvas.height * 0.5);
    let best = 0;
    let bestD = Infinity;
    for (let a = 0; a < 3; a++) {
      const p = a * 4;
      const d = this.distToSegment(ndc[0], ndc[1], this.gizmoNDC[p], this.gizmoNDC[p + 1], this.gizmoNDC[p + 2], this.gizmoNDC[p + 3]);
      if (d < bestD) { bestD = d; best = a + 1; }
    }
    return bestD <= thr ? best : 0;
  }

  private applyGizmoDrag(dxPx: number, dyPx: number): void {
    if (this.dragAxis < 1 || this.dragAxis > 3) return;
    const axis = this.gizmoAxes[this.dragAxis - 1];
    const pos = this.selectedId === 1 ? this.cubePos : this.ballPos;
    const p0 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(pos[0], pos[1], pos[2], 1));
    const p1 = mat4.mul(this.frameViewProj as unknown as Mat4, vec4.create(pos[0] + axis[0], pos[1] + axis[1], pos[2] + axis[2], 1));
    let sx = p1[0] / p1[3] - p0[0] / p0[3];
    let sy = p1[1] / p1[3] - p0[1] / p0[3];
    const len = Math.hypot(sx, sy);
    if (len < 1e-6) return;
    sx /= len; sy /= len;
    // screen pixels moved along the axis (NDC y is flipped vs mouse y)
    const sPx = dxPx * sx - dyPx * sy;
    const scale = (2 * this.camera.distance * Math.tan(this.camera.fov * Math.PI / 360)) / this.ctx.canvas.height;
    const t = sPx * scale;
    pos[0] += axis[0] * t;
    pos[1] += axis[1] * t;
    pos[2] += axis[2] * t;
  }

  private onPointerDown = (e: PointerEvent) => {
    if (!this.editMode) return;
    const ndc = this.eventToNDC(e);
    const axis = this.hitGizmoAxis(ndc);
    if (axis > 0) {
      e.stopPropagation();
      e.preventDefault();
      this.ctx.canvas.setPointerCapture(e.pointerId);
      this.dragAxis = axis;
      this.dragging = true;
      this.dragLastX = e.clientX;
      this.dragLastY = e.clientY;
      return;
    }
    this.selectedId = this.pickAtNDC(ndc);
  };

  private onPointerMove = (e: PointerEvent) => {
    if (!this.dragging) return;
    e.stopPropagation();
    e.preventDefault();
    const dx = e.clientX - this.dragLastX;
    const dy = e.clientY - this.dragLastY;
    this.dragLastX = e.clientX;
    this.dragLastY = e.clientY;
    this.applyGizmoDrag(dx, dy);
  };

  private onPointerUp = (e: PointerEvent) => {
    if (!this.dragging) return;
    e.stopPropagation();
    this.dragging = false;
    this.dragAxis = 0;
    this.ctx.canvas.releasePointerCapture(e.pointerId);
  };

  private pickAtNDC(ndc: [number, number]): number {
    // Unproject a ray through the clicked pixel using the last frame's invVP.
    const cam = this.camera.position;
    const near = mat4.mul(this.frameInvViewProj as unknown as Mat4, vec4.create(ndc[0], ndc[1], 0.0, 1.0));
    const rd = vec3.normalize(vec3.create(
      near[0] / near[3] - cam[0],
      near[1] / near[3] - cam[1],
      near[2] / near[3] - cam[2],
    ));

    let bestT = Infinity;
    let bestId = 0;

    // Ball: sphere at ballPos with radius 1
    {
      const ocx = cam[0] - this.ballPos[0];
      const ocy = cam[1] - this.ballPos[1];
      const ocz = cam[2] - this.ballPos[2];
      const b = ocx * rd[0] + ocy * rd[1] + ocz * rd[2];
      const c = ocx * ocx + ocy * ocy + ocz * ocz - 1;
      const disc = b * b - c;
      if (disc >= 0) {
        const t = -b - Math.sqrt(disc);
        if (t > 0 && t < bestT) { bestT = t; bestId = 2; }
      }
    }

    // Cube: OBB via inverse model; local AABB is [-1.5, 1.5] (scale 1.5).
    // Ray param t is preserved under the affine transform.
    {
      const inv = mat4.inverse(this.cubeModel as unknown as Mat4) as unknown as Float32Array;
      const om = mat4.mul(inv, vec4.create(cam[0], cam[1], cam[2], 1));
      const dm = mat4.mul(inv, vec4.create(cam[0] + rd[0], cam[1] + rd[1], cam[2] + rd[2], 1));
      const ox = om[0] / om[3], oy = om[1] / om[3], oz = om[2] / om[3];
      const dx = dm[0] / dm[3] - ox, dy = dm[1] / dm[3] - oy, dz = dm[2] / dm[3] - oz;
      let tmin = -Infinity;
      let tmax = Infinity;
      const slabs = [[dx, ox, -1.5, 1.5], [dy, oy, -1.5, 1.5], [dz, oz, -1.5, 1.5]] as const;
      for (let i = 0; i < 3; i++) {
        const d = slabs[i][0], o = slabs[i][1], lo = slabs[i][2], hi = slabs[i][3];
        if (Math.abs(d) < 1e-8) {
          if (o < lo || o > hi) return bestId;
        } else {
          let t1 = (lo - o) / d;
          let t2 = (hi - o) / d;
          if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
          tmin = Math.max(tmin, t1);
          tmax = Math.min(tmax, t2);
          if (tmin > tmax) return bestId;
        }
      }
      const t = Math.max(tmin, 0);
      if (t < bestT) { bestT = t; bestId = 1; }
    }

    return bestId;
  }

  registerGUI(gui: any) {
    gui.add(this, "metallic", 0, 1, 0.01).name("Metallic");
    gui.add(this, "roughness", 0.01, 1, 0.01).name("Roughness");
    gui.add(this, "bloomIntensity", 0, 1, 0.01).name("Bloom Intensity");
    gui.add(this, "useCSM").name("CSM Shadows");
    gui.add(this, "useSSAO").name("GTAO");
    gui.add(this, "useTAA").name("TAA");
    gui.add(this, "taaJitter").name("TAA Jitter");
    gui.add(this, "pauseAnimation").name("Pause");
    const editFolder = gui.addFolder("Edit");
    editFolder.add(this, "editMode").name("Edit Mode (pick & move)");
    editFolder.add(this, "selectedId", 0, 2, 1).name("Selection").listen();
    gui.add(this.taaPass, "alpha", 0.02, 0.5, 0.01).name("TAA Alpha");
    const taaFolder = gui.addFolder("TAA Debug");
    taaFolder.add(this.taaPass, "debugMode", {
      "OFF": 0,
      "Motion": 1,
      "No-Reproj": 2,
      "No-History": 3,
    }).name("Mode");
    gui.add(this.deferredLighting, "envIntensity", 0, 3, 0.05).name("IBL Intensity");
    gui.add(this.bloomPass, "threshold", 0, 2, 0.01).name("Bloom Threshold");
    const pp = this.postProcessPass.params;
    const fxFolder = gui.addFolder("Post Process");
    fxFolder.add(pp, "exposure", 0.1, 3, 0.01).name("Exposure");
    fxFolder.add(pp, "chromaticStrength", 0, 0.05, 0.001).name("Chromatic Aberr.");
    fxFolder.add(pp, "vignetteStrength", 0, 1, 0.01).name("Vignette");
    fxFolder.add(pp, "saturation", 0, 2, 0.01).name("Saturation");
  }

  destroy() {
    this.ctx.canvas.removeEventListener("pointerdown", this.onPointerDown, true);
    this.ctx.canvas.removeEventListener("pointermove", this.onPointerMove, true);
    this.ctx.canvas.removeEventListener("pointerup", this.onPointerUp, true);
    this.ctx.canvas.removeEventListener("pointercancel", this.onPointerUp, true);
    this.cubeVB.destroy();
    this.cubeIB.destroy();
    this.sphereVB.destroy();
    this.sphereIB.destroy();
    this.sceneUBO.destroy();
    this.ballUBO.destroy();
    this.shadowUBO.destroy();
    this.gizmoVB.destroy();
    this.dummyDepthTexture.destroy();
    this.gbuffer.destroy();
    this.passManager.destroy();
    this.deferredLighting.destroy();
    this.csm.destroy();
    this.gtao.destroy();
    this.envMap.destroy();
    this.brdfLut.destroy();
    this.bloomPass.destroy();
    this.taaPass.destroy();
    this.postProcessPass.destroy();
  }
}