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
import { LightScene, createDirectionalLight, createPointLight, createSpotLight } from "../scene/light";
import { createCubeGeometry, createSphereGeometry } from "../utils/geometry";
import { mat4, vec3, type Mat4 } from "wgpu-matrix";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";

const gbufferVS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) prevScreenPos: vec2<f32>,
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
  out.worldPos = worldPos.xyz;
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.uv = uv;
  let prevClip = u.prevViewProj * worldPos;
  out.prevScreenPos = prevClip.xy / prevClip.w;
  return out;
}
`;

const gbufferFS = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  prevViewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) prevScreenPos: vec2<f32>,
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

  let curScreen = in.position.xy / in.position.w;
  let motion = (curScreen - in.prevScreenPos) * 0.5;
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
  private lightScene!: LightScene;

  private gbufferPipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;

  private cubeVB!: GPUBuffer;
  private cubeIB!: GPUBuffer;
  private cubeIndexCount = 0;
  private sphereVB!: GPUBuffer;
  private sphereIB!: GPUBuffer;
  private sphereIndexCount = 0;

  private sceneUBO!: GPUBuffer;
  private shadowUBO!: GPUBuffer;
  private dummyDepthTexture!: GPUTexture;

  private prevViewProj: Float32Array = new Float32Array(16);

  metallic = 0.1;
  roughness = 0.5;
  bloomIntensity = 0.3;
  useCSM = true;
  useSSAO = true;

  private vsCode = gbufferVS;
  private fsCode = gbufferFS;

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
    this.bloomPass = new BloomPass(ctx.device, "rgba16float");
    this.bloomPass.threshold = 0.3;
    this.bloomPass.knee = 0.7;
    this.bloomPass.radius = 2.5;
    this.postProcessPass = new PostProcessPass(ctx.device, ctx.format);

    this.createGeometry();
    this.buildPipelines();

    this.sceneUBO = this.device.createBuffer({
      label: "deferred-scene-ubo",
      size: 288,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.shadowUBO = this.device.createBuffer({
      label: "deferred-shadow-ubo",
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    mat4.identity(this.prevViewProj);

    this.dummyDepthTexture = this.device.createTexture({
      label: "deferred-dummy-depth",
      size: [1, 1],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
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

  private uboData = new Float32Array(72);
  private frameTime = 0;

  update(time: number) {
    this.frameTime = time;
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);

    const ubo = this.uboData;
    ubo.set(viewProj as unknown as ArrayLike<number>, 0);
    ubo.set(this.prevViewProj as unknown as ArrayLike<number>, 16);
    const model = mat4.mul(mat4.rotationY(time * 0.5), mat4.scaling([1.5, 1.5, 1.5]));
    const invTransModel = mat4.transpose(mat4.inverse(model));
    ubo.set(model as unknown as ArrayLike<number>, 32);
    ubo.set(invTransModel as unknown as ArrayLike<number>, 48);
    ubo[64] = this.camera.position[0];
    ubo[65] = this.camera.position[1];
    ubo[66] = this.camera.position[2];
    ubo[67] = 1.0;
    ubo[68] = time;
    ubo[69] = this.metallic;
    ubo[70] = this.roughness;
    ubo[71] = 0;

    this.device.queue.writeBuffer(this.sceneUBO, 0, ubo as unknown as GPUAllowSharedBufferSource);
    this.prevViewProj.set(viewProj as unknown as ArrayLike<number>);

    this.bloomPass.bloomIntensity = this.bloomIntensity;

    const invViewProj = mat4.inverse(viewProj);
    this.postProcessPass.cameraPos = [this.camera.position[0], this.camera.position[1], this.camera.position[2]];
    this.postProcessPass.invVP.set(invViewProj as unknown as ArrayLike<number>);

    if (this.useCSM) {
      this.csm.updateCascadeVPs(
        viewProj as Mat4,
        invViewProj as Mat4,
        0.1, 50.0,
      );
    }
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

            const shadowUbo = new Float32Array(32);
            shadowUbo.set(this.csm.cascadeVPs[i] as unknown as ArrayLike<number>, 0);
            const model = mat4.mul(mat4.rotationY(0), mat4.scaling([1.5, 1.5, 1.5]));
            shadowUbo.set(model as unknown as ArrayLike<number>, 16);
            this.device.queue.writeBuffer(this.shadowUBO, 0, shadowUbo as unknown as GPUAllowSharedBufferSource);

            const shadowBG = this.device.createBindGroup({
              layout: this.shadowPipeline.getBindGroupLayout(0),
              entries: [{ binding: 0, resource: { buffer: this.shadowUBO } }],
            });
            shadowPass.setBindGroup(0, shadowBG);
            shadowPass.setVertexBuffer(0, this.cubeVB);
            shadowPass.setIndexBuffer(this.cubeIB, "uint16");
            shadowPass.drawIndexed(this.cubeIndexCount);
            shadowPass.end();
          }
        }

        // Step 2: GBuffer pass
        {
          const gbufferPass = this.gbuffer.beginGBufferPass(encoder);
          gbufferPass.setPipeline(this.gbufferPipeline);
          const gbufferBG = this.device.createBindGroup({
            layout: this.gbufferPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.sceneUBO } }],
          });
          gbufferPass.setBindGroup(0, gbufferBG);
          gbufferPass.setVertexBuffer(0, this.cubeVB);
          gbufferPass.setIndexBuffer(this.cubeIB, "uint16");
          gbufferPass.drawIndexed(this.cubeIndexCount);
          gbufferPass.end();
        }

        // Step 3: SSAO pass
        if (this.useSSAO) {
          const aspect = w / h;
          const proj = this.camera.getViewProjectionMatrix(aspect);
          this.gtao.execute(encoder, this.gbuffer, proj as unknown as Float32Array);
        }

        // Step 4: Lighting pass
        const lightingRT = this.passManager.getOrCreateTarget("deferred-lighting", "rgba16float");
        {
          const aspect = w / h;
          const viewProj = this.camera.getViewProjectionMatrix(aspect);
          const invViewProj = mat4.inverse(viewProj);
          this.deferredLighting.update(
            viewProj as Mat4,
            [this.camera.position[0], this.camera.position[1], this.camera.position[2]],
            invViewProj as Mat4,
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
          );
        }

        // Step 5: Bloom (extract bright pixels, downsample/upsample pyramid)
        const bloomResult = this.bloomPass.execute(encoder, lightingRT.texture);
        const bloomCombineRT = this.passManager.getOrCreateTarget("bloom-combine", "rgba16float");
        this.bloomPass.combine(
          encoder,
          lightingRT.texture,
          bloomResult.view,
          bloomCombineRT.view,
          this.bloomIntensity,
        );

        // Step 6: Post-process (tonemap, saturation, vignette, chromatic) → screen
        this.postProcessPass.execute(
          encoder,
          bloomCombineRT.texture,
          this.gbuffer.depthSampledView,
          screenView,
          [w, h],
          this.frameTime,
        );
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

  registerGUI(gui: any) {
    gui.add(this, "metallic", 0, 1, 0.01).name("Metallic");
    gui.add(this, "roughness", 0.01, 1, 0.01).name("Roughness");
    gui.add(this, "bloomIntensity", 0, 1, 0.01).name("Bloom Intensity");
    gui.add(this, "useCSM").name("CSM Shadows");
    gui.add(this, "useSSAO").name("GTAO");
    gui.add(this.bloomPass, "threshold", 0, 2, 0.01).name("Bloom Threshold");
    const pp = this.postProcessPass.params;
    const fxFolder = gui.addFolder("Post Process");
    fxFolder.add(pp, "exposure", 0.1, 3, 0.01).name("Exposure");
    fxFolder.add(pp, "chromaticStrength", 0, 0.02, 0.001).name("Chromatic Aberr.");
    fxFolder.add(pp, "vignetteStrength", 0, 1, 0.01).name("Vignette");
    fxFolder.add(pp, "saturation", 0, 2, 0.01).name("Saturation");
  }

  destroy() {
    this.cubeVB.destroy();
    this.cubeIB.destroy();
    this.sphereVB.destroy();
    this.sphereIB.destroy();
    this.sceneUBO.destroy();
    this.shadowUBO.destroy();
    this.dummyDepthTexture.destroy();
    this.gbuffer.destroy();
    this.passManager.destroy();
    this.deferredLighting.destroy();
    this.csm.destroy();
    this.gtao.destroy();
    this.bloomPass.destroy();
    this.postProcessPass.destroy();
  }
}