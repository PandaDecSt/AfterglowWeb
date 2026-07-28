import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { GPUTerrain } from "../utils/gpu-terrain";
import { GPUWindField } from "../utils/gpu-wind-field";

const GRASS_COUNT = 65536;
const BLADE_SEGMENTS = 4;
const BLADE_VERTS = (BLADE_SEGMENTS + 1) * 2; // 10 verts per blade
const BLADE_INDICES = BLADE_SEGMENTS * 6; // 24 indices per blade

// Grass blade geometry + wind animation via compute
const grassComputeShader = `
struct GrassUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  time: f32,
  windStrength: f32,
  bladeHeight: f32,
  grassCount: f32,
  _pad: f32,
};

struct GrassInstance {
  position: vec3<f32>,
  height: f32,
  bendDir: vec2<f32>,
  width: f32,
  colorVariation: f32,
};

@group(0) @binding(0) var<uniform> u: GrassUniforms;
@group(0) @binding(1) var<storage, read_write> instances: array<GrassInstance>;
@group(0) @binding(2) var windTex: texture_2d<f32>;
@group(0) @binding(3) var windSampler: sampler;
@group(0) @binding(4) var terrainTex: texture_2d<f32>;
@group(0) @binding(5) var terrainSampler: sampler;

fn hash1d(x: f32, seed: f32) -> f32 {
  return fract(sin(x * 127.1 + seed) * 43758.5453);
}

fn hash2d(p: vec2<f32>, seed: f32) -> vec2<f32> {
  return fract(sin(vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)))) * 43758.5453 + seed);
}

fn loadTerrainHeight(worldXY: vec2<f32>) -> f32 {
  let uv = worldXY / 30.0 + 0.5;
  return textureSampleLevel(terrainTex, terrainSampler, uv, 0.0).r;
}

fn sampleWind(worldXY: vec2<f32>) -> vec2<f32> {
  let uv = worldXY / 24.0 + 0.5;
  return textureSampleLevel(windTex, windSampler, uv, 0.0).rg;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= u32(u.grassCount)) {
    return;
  }

  let fi = f32(index);
  let seed = 42.123;

  // Random position on terrain
  let rndPos = hash2d(vec2<f32>(fi, fi * 0.7), seed);
  let worldX = (rndPos.x - 0.5) * 24.0;
  let worldZ = (rndPos.y - 0.5) * 24.0;
  let terrainH = loadTerrainHeight(vec2<f32>(worldX, worldZ));

  // Wind bending - subtle, driven by wind field
  let wind = sampleWind(vec2<f32>(worldX, worldZ));
  let windPhase = sin(u.time * 1.5 + worldX * 0.3 + worldZ * 0.2);
  let bendStrength = length(wind) * u.windStrength * (0.3 + abs(windPhase) * 0.7);
  let bendDir = normalize(wind + vec2<f32>(0.001)) * bendStrength;

  var inst: GrassInstance;
  inst.position = vec3<f32>(worldX, terrainH, worldZ);
  inst.height = u.bladeHeight * (0.5 + hash1d(fi * 1.3, seed) * 0.5);
  inst.bendDir = bendDir;
  inst.width = 0.015 + hash1d(fi * 2.7, seed) * 0.015;
  inst.colorVariation = hash1d(fi * 3.1, seed);

  instances[index] = inst;
}
`;

// Grass rendering: instanced blade quads with wind bending and lighting
const grassRenderShader = `
struct GrassUniforms {
  viewProj: mat4x4<f32>,
  cameraPos: vec3<f32>,
  time: f32,
  windStrength: f32,
  bladeHeight: f32,
  grassCount: f32,
  _pad: f32,
};

struct GrassInstance {
  position: vec3<f32>,
  height: f32,
  bendDir: vec2<f32>,
  width: f32,
  colorVariation: f32,
};

@group(0) @binding(0) var<uniform> u: GrassUniforms;
@group(0) @binding(1) var<storage, read> instances: array<GrassInstance>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) heightFactor: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let instanceIndex = vertexIndex / 10u;
  let vertIndex = vertexIndex % 10u;

  let inst = instances[instanceIndex];

  // Blade segment (0..BLADE_SEGMENTS), each segment has 2 verts (left/right)
  let segment = vertIndex / 2u;
  let side = vertIndex % 2u;
  let t = f32(segment) / 4.0;

  // Wind bend increases with height (quadratic, but subtle)
  let bendFactor = t * t;
  let bendOffset = inst.bendDir * bendFactor * inst.height * 0.3;

  // Blade width tapers smoothly to zero at tip
  let halfWidth = inst.width * (1.0 - t) * (1.0 - t * 0.3);
  let sideOffset = f32(side) * 2.0 - 1.0;

  // Blade faces camera (billboard-like perpendicular to view direction)
  let toCamera = normalize(u.cameraPos - inst.position);
  let bladeDir = normalize(toCamera.xz);
  // Slight random rotation for natural look
  let rotAngle = inst.colorVariation * 0.5;
  let cosR = cos(rotAngle);
  let sinR = sin(rotAngle);
  let rotatedDir = vec2<f32>(bladeDir.x * cosR - bladeDir.y * sinR, bladeDir.x * sinR + bladeDir.y * cosR);

  let worldPos = vec3<f32>(
    inst.position.x + rotatedDir.x * sideOffset * halfWidth + bendOffset.x,
    inst.position.y + t * inst.height,
    inst.position.z + rotatedDir.y * sideOffset * halfWidth + bendOffset.y
  );

  // Compute blade normal for lighting
  let normal = vec3<f32>(-inst.bendDir.x * 0.3, 1.0, -inst.bendDir.y * 0.3);

  var out: VSOut;
  out.position = u.viewProj * vec4<f32>(worldPos, 1.0);

  // Lighting: hemisphere ambient + directional diffuse
  let lightDir = normalize(vec3<f32>(0.4, 0.8, 0.3));
  let skyColor = vec3<f32>(0.35, 0.5, 0.25);
  let groundColor = vec3<f32>(0.15, 0.12, 0.05);
  let sunColor = vec3<f32>(0.8, 0.7, 0.3);
  let ambient = mix(groundColor, skyColor, normal.y * 0.5 + 0.5);
  let diffuse = max(dot(normal, lightDir), 0.0) * sunColor * 0.6;
  let lighting = ambient + diffuse;

  // Color: gradient from dark green (base) to bright green (tip)
  let baseGreen = vec3<f32>(0.12, 0.28, 0.06);
  let tipGreen = vec3<f32>(0.25, 0.55, 0.12);
  let variation = (inst.colorVariation - 0.5) * 0.15;
  let bladeColor = mix(baseGreen, tipGreen, t) + vec3<f32>(variation, variation * 0.5, variation * 0.3);
  out.color = bladeColor * lighting;
  out.heightFactor = t;

  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // Fade out at blade tip for softer look
  let alpha = 1.0 - smoothstep(0.8, 1.0, in.heightFactor);
  if (alpha < 0.05) {
    discard;
  }
  return vec4<f32>(in.color, alpha);
}
`;

export class GrassDemo implements Demo {
  label = "Grass";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private instanceBuffer!: GPUBuffer;
  private uniformBuffer!: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;

  private terrain!: GPUTerrain;
  private windField!: GPUWindField;

  private uniformData = new Float32Array(24);
  private depthTexture!: GPUTexture;
  private terrainInitialized = false;

  windStrength = 1.0;
  bladeHeight = 0.4;

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    this.terrain = new GPUTerrain(this.device, 256, 5.0);
    this.windField = new GPUWindField(this.device, 64);

    // Depth texture
    this.depthTexture = this.device.createTexture({
      label: "grass-depth",
      size: [ctx.canvas.width, ctx.canvas.height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Instance buffer: 8 floats per instance (WGSL aligned to 32 bytes)
    this.instanceBuffer = this.device.createBuffer({
      label: "grass-instances",
      size: GRASS_COUNT * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "grass-ubo",
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute pipeline
    const computeModule = this.device.createShaderModule({ code: grassComputeShader });
    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      ],
    });
    this.computePipeline = this.device.createComputePipeline({
      label: "grass-compute",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Render pipeline with depth testing
    const renderModule = this.device.createShaderModule({ code: grassRenderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "grass-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    });

    // Bind groups
    this.computeBindGroup = this.device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.instanceBuffer } },
        { binding: 2, resource: this.windField.texture.createView() },
        { binding: 3, resource: this.windField.createSampler() },
        { binding: 4, resource: this.terrain.view },
        { binding: 5, resource: this.terrain.heightSampler },
      ],
    });

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.instanceBuffer } },
      ],
    });
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    this.uniformData.set(viewProj as unknown as ArrayLike<number>, 0);
    // Camera position (extract from view matrix inverse)
    const cam = this.camera;
    this.uniformData[16] = cam.position?.[0] ?? 0;
    this.uniformData[17] = cam.position?.[1] ?? 2;
    this.uniformData[18] = cam.position?.[2] ?? 5;
    this.uniformData[19] = time;
    this.uniformData[20] = this.windStrength;
    this.uniformData[21] = this.bladeHeight;
    this.uniformData[22] = GRASS_COUNT;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    if (!this.terrainInitialized) {
      this.terrain.dispatchOnce(encoder);
      this.terrainInitialized = true;
    }

    this.windField.dispatch(encoder, performance.now() / 1000);

    // Update grass instances
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.computeBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(GRASS_COUNT / 256));
    computePass.end();

    // Render grass
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.4, g: 0.6, b: 0.8, a: 1 },
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
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroup);
    renderPass.draw(GRASS_COUNT * BLADE_VERTS);
    renderPass.end();
  }

  stats() {
    return {
      drawCalls: 1,
      instances: GRASS_COUNT,
      triangles: GRASS_COUNT * BLADE_SEGMENTS * 2,
      computeDispatches: 2,
      custom: {
        "Blades": GRASS_COUNT.toLocaleString(),
        "Segments/Blade": BLADE_SEGMENTS,
        "Wind": "GPU texture driven",
        "Terrain": "GPU heightmap",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "windStrength", 0, 3, 0.1).name("Wind Strength");
    gui.add(this, "bladeHeight", 0.1, 1.5, 0.05).name("Blade Height");
  }

  destroy() {
    this.instanceBuffer.destroy();
    this.uniformBuffer.destroy();
    this.depthTexture.destroy();
    this.terrain.destroy();
    this.windField.destroy();
  }
}
