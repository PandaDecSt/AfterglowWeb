import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { GPUWindField } from "../utils/gpu-wind-field";
import { GPUTerrain } from "../utils/gpu-terrain";
import type { RenderPass } from "../core/renderer";

const PARTICLE_COUNT = 8192;

// Faithful port of ParticleExample_CS.hlsl
// Particle struct: position(3) + elapseTime(1) + velocity(3) + maxLifeTime(1) + color(4) = 48 bytes (WGSL aligned)
const computeShader = `
struct Particle {
  position: vec3<f32>,
  elapseTime: f32,
  velocity: vec3<f32>,
  maxLifeTime: f32,
  color: vec4<f32>,
};

struct SimUniforms {
  deltaTime: f32,
  time: f32,
  particleCount: f32,
  pad: f32,
  cameraPosition: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sim: SimUniforms;
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(3) var windTex: texture_2d<f32>;
@group(0) @binding(4) var windSampler: sampler;
@group(0) @binding(5) var terrainTex: texture_2d<f32>;
@group(0) @binding(6) var terrainSampler: sampler;

fn hash3d(xyz: vec3<f32>, seed: f32) -> vec3<f32> {
  return fract(sin(vec3<f32>(
    dot(xyz, vec3<f32>(83.7247, 71.7823, 24.274)),
    dot(xyz, vec3<f32>(64.4634, 49.4349, 82.263)),
    dot(xyz, vec3<f32>(94.262, 20.9245, 34.8256))
  )) * 52567.0925 + seed);
}

fn hash1d(x: f32, seed: f32) -> f32 {
  return fract(sin(x * 14523.46187) * seed);
}

fn toSnorm(v: vec3<f32>) -> vec3<f32> {
  return v * 2.0 - 1.0;
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
  if (index >= u32(sim.particleCount)) {
    return;
  }

  let pIn = particlesIn[index];

  // Constants from original
  let fadeThreshold = 0.25;
  let regeneratedRange = 8.0;
  let maxHorizontalSpeed = 0.5;
  let frictionFactor = 0.005;

  let boundaryMin = vec3<f32>(-10.0, -10.0, 0.0) + sim.cameraPosition.xyz;
  let boundaryMax = vec3<f32>(10.0, 10.0, 10.0) + sim.cameraPosition.xyz;

  var deltaElapseTime = sim.deltaTime;
  var positionOut = pIn.position;

  // If particle leaving boundary, increase elapsing time (fade out faster)
  if (any(positionOut < boundaryMin) || any(positionOut > boundaryMax)) {
    deltaElapseTime *= 4.0;
  }

  var pOut: Particle;
  pOut.elapseTime = pIn.elapseTime + deltaElapseTime;

  if (pIn.elapseTime > pIn.maxLifeTime) {
    // Respawn: random position around camera, above terrain
    pOut.elapseTime = 0.0;
    var rnd = hash3d(vec3<f32>(f32(index), f32(index), f32(index)), 1312.21551);
    rnd = toSnorm(rnd);
    let offset = vec3<f32>(sim.cameraPosition.x, sim.cameraPosition.y, 0.0);
    positionOut = rnd * regeneratedRange + offset;
    // Spawn above terrain
    let terrainH = loadTerrainHeight(positionOut.xy);
    positionOut.z += terrainH;
    pOut.velocity = vec3<f32>(0.0);
  } else {
    // Integrate position
    positionOut = pIn.position + pIn.velocity * sim.deltaTime;
  }

  pOut.position = positionOut;

  // Accumulate wind velocity (from original: friction + wind field + speed clamp)
  var velXY = pIn.velocity.xy * (1.0 - frictionFactor);
  velXY = velXY + sampleWind(positionOut.xy) * sim.deltaTime;
  let speed = length(velXY);
  if (speed > maxHorizontalSpeed) {
    velXY = normalize(velXY) * maxHorizontalSpeed;
  }
  pOut.velocity = vec3<f32>(velXY, pIn.velocity.z);

  // Particle opacity: fade in/out
  let fadeValue = min(
    min(pIn.maxLifeTime - pIn.elapseTime, pIn.elapseTime) / pIn.maxLifeTime,
    fadeThreshold
  );
  pOut.color = vec4<f32>(pIn.color.rgb, fadeValue * (1.0 / fadeThreshold));
  pOut.maxLifeTime = pIn.maxLifeTime;

  particlesOut[index] = pOut;
}
`;

// Initialize particles (from ParticleExampleInitializer_CS.hlsl)
const initShader = `
struct Particle {
  position: vec3<f32>,
  elapseTime: f32,
  velocity: vec3<f32>,
  maxLifeTime: f32,
  color: vec4<f32>,
};

struct InitUniforms {
  particleCount: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> u: InitUniforms;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

fn hash3d(xyz: vec3<f32>, seed: f32) -> vec3<f32> {
  return fract(sin(vec3<f32>(
    dot(xyz, vec3<f32>(83.7247, 71.7823, 24.274)),
    dot(xyz, vec3<f32>(64.4634, 49.4349, 82.263)),
    dot(xyz, vec3<f32>(94.262, 20.9245, 34.8256))
  )) * 52567.0925 + seed);
}

fn hash1d(x: f32, seed: f32) -> f32 {
  return fract(sin(x * 14523.46187) * seed);
}

fn toSnorm3(v: vec3<f32>) -> vec3<f32> {
  return v * 2.0 - 1.0;
}

@compute @workgroup_size(256)
fn cs_init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= u32(u.particleCount)) {
    return;
  }

  let fi = f32(index);
  let seed0 = 1312.21551;
  let seed1 = 5441.35424;

  var p: Particle;
  p.position = toSnorm3(hash3d(vec3<f32>(fi, fi, fi), seed0)) * 8.0;
  p.position.z = abs(p.position.z) * 0.5 + 0.5;
  p.maxLifeTime = (hash1d(1.0 / max(fi, 1.0), seed1) + 1.0) * 10.0;
  p.velocity = toSnorm3(hash3d(p.position * 100.0, seed0)) * 0.25;
  p.elapseTime = hash1d(fi * 0.1, seed1) * p.maxLifeTime;
  p.color = vec4<f32>(hash3d(vec3<f32>(fi, fi, fi), seed0 * 0.5), 1.0);

  particles[index] = p;
}
`;

// Point sprite billboard rendering (from ParticleExample_VS/FS.hlsl)
const renderShader = `
struct RenderUniforms {
  viewProj: mat4x4<f32>,
  screenResolution: vec2<f32>,
  cameraFov: f32,
  pad: f32,
};

struct Particle {
  position: vec3<f32>,
  elapseTime: f32,
  velocity: vec3<f32>,
  maxLifeTime: f32,
  color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) @interpolate(flat) centerPosition: vec2<f32>,
  @location(2) @interpolate(flat) pointSize: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let particleIndex = vertexIndex / 6u;
  let cornerIndex = vertexIndex % 6u;

  let p = particles[particleIndex];
  let clipPos = u.viewProj * vec4<f32>(p.position, 1.0);

  // Screen-space center (from original: centerPosition = clip.xy/w * 0.5+0.5 * screenRes)
  let centerPosition = (clipPos.xy / clipPos.w * 0.5 + 0.5) * u.screenResolution;

  // Point size: radius in pixels (from original: pointSize / w * screenHeight / fov)
  let basePointSize = 0.08;
  let pointSize = basePointSize / clipPos.w * u.screenResolution.y / u.cameraFov;

  // Billboard quad corners
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
  );

  let offset = corners[cornerIndex] * pointSize;
  let finalScreen = centerPosition + offset;
  let finalNdc = (finalScreen / u.screenResolution) * 2.0 - 1.0;

  var out: VSOut;
  out.position = vec4<f32>(finalNdc * clipPos.w, clipPos.z, clipPos.w);
  out.color = p.color;
  out.centerPosition = centerPosition;
  out.pointSize = pointSize;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  // Circle alpha (from original: circleAlpha = 1 - dist/pointSize*2, smoothstep)
  let dist = distance(in.centerPosition, in.position.xy);
  var circleAlpha = 1.0 - dist / in.pointSize * 2.0;
  circleAlpha = smoothstep(0.0, 1.0, circleAlpha);

  var color: vec3<f32> = in.color.rgb;
  let alpha = in.color.a * circleAlpha;

  if (alpha < 0.01) {
    discard;
  }

  return vec4<f32>(color * alpha, alpha);
}
`;

export class ParticleDemo implements Demo {
  label = "Particles";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private initPipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;

  private particleBuffers: GPUBuffer[] = [];
  private simUniformBuffer!: GPUBuffer;
  private initUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;

  private computeBindGroups: GPUBindGroup[] = [];
  private initBindGroup!: GPUBindGroup;
  private renderBindGroups: GPUBindGroup[] = [];

  private windField!: GPUWindField;
  private terrain!: GPUTerrain;

  private current = 0;
  private initialized = false;
  private simData = new Float32Array(8);
  private renderData = new Float32Array(20);

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    this.windField = new GPUWindField(this.device, 64);
    this.terrain = new GPUTerrain(this.device, 256, 5.0);

    // Particle buffer: 48 bytes per particle (WGSL aligned)
    const particleStride = 48;
    for (let i = 0; i < 2; i++) {
      this.particleBuffers.push(this.device.createBuffer({
        label: `particles-${i}`,
        size: PARTICLE_COUNT * particleStride,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
    }

    this.simUniformBuffer = this.device.createBuffer({
      label: "particle-sim-ubo",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.initUniformBuffer = this.device.createBuffer({
      label: "particle-init-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.renderUniformBuffer = this.device.createBuffer({
      label: "particle-render-ubo",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute pipeline (simulation)
    const computeModule = this.device.createShaderModule({ code: computeShader });
    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: {} },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      ],
    });
    const computeLayout = this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] });
    this.computePipeline = this.device.createComputePipeline({
      label: "particle-compute",
      layout: computeLayout,
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Init pipeline
    const initModule = this.device.createShaderModule({ code: initShader });
    const initBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.initPipeline = this.device.createComputePipeline({
      label: "particle-init",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [initBGL] }),
      compute: { module: initModule, entryPoint: "cs_init" },
    });

    // Render pipeline
    const renderModule = this.device.createShaderModule({ code: renderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "particle-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });

    // Bind groups
    const windSampler = this.windField.createSampler();
    const terrainSampler = this.terrain.heightSampler;

    for (let i = 0; i < 2; i++) {
      this.computeBindGroups.push(this.device.createBindGroup({
        layout: computeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuffer } },
          { binding: 1, resource: { buffer: this.particleBuffers[i] } },
          { binding: 2, resource: { buffer: this.particleBuffers[1 - i] } },
          { binding: 3, resource: this.windField.texture.createView() },
          { binding: 4, resource: windSampler },
          { binding: 5, resource: this.terrain.view },
          { binding: 6, resource: terrainSampler },
        ],
      }));
    }

    this.initBindGroup = this.device.createBindGroup({
      layout: initBGL,
      entries: [
        { binding: 0, resource: { buffer: this.initUniformBuffer } },
        { binding: 1, resource: { buffer: this.particleBuffers[0] } },
      ],
    });

    for (let i = 0; i < 2; i++) {
      this.renderBindGroups.push(this.device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderUniformBuffer } },
          { binding: 1, resource: { buffer: this.particleBuffers[i] } },
        ],
      }));
    }
  }

  update(time: number, deltaTime: number) {
    const dt = Math.min(deltaTime, 0.05);

    // Sim uniforms: deltaTime, time, particleCount, pad, cameraPosition(4)
    this.simData[0] = dt;
    this.simData[1] = time;
    this.simData[2] = PARTICLE_COUNT;
    this.simData[3] = 0;
    this.simData[4] = this.camera.position[0];
    this.simData[5] = this.camera.position[1];
    this.simData[6] = this.camera.position[2];
    this.simData[7] = 1;
    this.device.queue.writeBuffer(this.simUniformBuffer, 0, this.simData as unknown as GPUAllowSharedBufferSource);

    // Render uniforms
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    this.renderData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.renderData[16] = w;
    this.renderData[17] = h;
    this.renderData[18] = (this.camera.fov * Math.PI) / 180;
    this.renderData[19] = 0;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        // Initialize particles once
        if (!this.initialized) {
          // Generate terrain first
          this.terrain.dispatchOnce(encoder);

          // Init particles
          this.device.queue.writeBuffer(this.initUniformBuffer, 0, new Float32Array([PARTICLE_COUNT, 0, 0, 0]) as unknown as GPUAllowSharedBufferSource);
          const initPass = encoder.beginComputePass();
          initPass.setPipeline(this.initPipeline);
          initPass.setBindGroup(0, this.initBindGroup);
          initPass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
          initPass.end();
          this.initialized = true;
        }

        // Update wind field
        this.windField.dispatch(encoder, performance.now() / 1000);

        // Simulate particles
        const simPass = encoder.beginComputePass();
        simPass.setPipeline(this.computePipeline);
        simPass.setBindGroup(0, this.computeBindGroups[this.current]);
        simPass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
        simPass.end();

        this.current = 1 - this.current;

        // Render
        const renderPass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.current]);
        renderPass.draw(PARTICLE_COUNT * 6);
        renderPass.end();
      },
    }];
  }

  stats() {
    return {
      drawCalls: 1,
      instances: PARTICLE_COUNT,
      triangles: PARTICLE_COUNT * 2,
      computeDispatches: 3,
      custom: {
        "Wind Field": "GPU 64x64 rg32f",
        "Terrain": "GPU 256x256 r32f",
        "Billboard": "Point Sprite",
      },
    };
  }

  registerGUI(gui: any) {
    // No per-material controls for particles
  }

  destroy() {
    for (const b of this.particleBuffers) b.destroy();
    this.simUniformBuffer.destroy();
    this.initUniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.windField.destroy();
    this.terrain.destroy();
  }
}
