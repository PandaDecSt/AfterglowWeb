import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { GPUTerrain } from "../utils/gpu-terrain";
import type { RenderPass } from "../core/renderer";

const BOID_COUNT = 512;

// Faithful port of BoidInstancing_CS.hlsl
// Boid attributes: acceleration(3) + influenceRadius(1) + maxAcceleration(3) + turnSpeed(1) + velocity(3) + curiosity(1) + maxVelocity(1) + perception(1)
// Instance: position stored in instanceModel[0..2].w, rotation in [0..2][0..2]
const boidComputeShader = `
struct BoidAttribs {
  acceleration: vec3<f32>,
  influenceRadius: f32,
  maxAcceleration: vec3<f32>,
  turnSpeed: f32,
  velocity: vec3<f32>,
  curiosity: f32,
  maxVelocity: f32,
  perception: f32,
};

struct InstanceData {
  // 4x4 matrix stored as 4 vec4s (column-major)
  col0: vec4<f32>,
  col1: vec4<f32>,
  col2: vec4<f32>,
  col3: vec4<f32>,
};

struct SimUniforms {
  deltaTime: f32,
  time: f32,
  boidCount: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> sim: SimUniforms;
@group(0) @binding(1) var<storage, read> boidsIn: array<BoidAttribs>;
@group(0) @binding(2) var<storage, read_write> boidsOut: array<BoidAttribs>;
@group(0) @binding(3) var<storage, read> instancesIn: array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> instancesOut: array<InstanceData>;
@group(0) @binding(5) var terrainTex: texture_2d<f32>;
@group(0) @binding(6) var terrainSampler: sampler;

fn loadTerrainHeight(worldXY: vec2<f32>) -> f32 {
  let uv = worldXY / 30.0 + 0.5;
  return textureSampleLevel(terrainTex, terrainSampler, uv, 0.0).r;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let count = u32(sim.boidCount);
  if (index >= count) {
    return;
  }

  // Home position: moving target (from original)
  let homePos = vec3<f32>(sin(sim.time * 0.2) * 100.0, cos(sim.time * 0.2) * 100.0, 0.5);

  let boid = boidsIn[index];
  let instanceInfo = instancesIn[index];

  // Extract position from instance matrix (col3 = translation)
  let position = instanceInfo.col3.xyz;
  var acceleration = boid.acceleration;
  let rcpPerception = 1.0 / boid.perception;

  // Alignment weight (from original: solves low-frame loose problem)
  let alignedAccelerationWeight = pow(rcpPerception, 0.04 * sqrt(1.0 / max(sim.deltaTime, 0.001)));

  // Accumulation for flocking
  var centerPos = vec3<f32>(0.0);
  var numBoids = 0u;
  var accumAcceleration = vec3<f32>(0.0);
  var closestDist = 1e16;
  var closestPos = position;
  var closestIndex = index;

  // O(n) neighbor search
  for (var i = 0u; i < count; i++) {
    if (i == index) {
      continue;
    }
    let otherInstance = instancesIn[i];
    let otherPos = otherInstance.col3.xyz;
    let d = distance(position, otherPos);

    // Perception-based accumulation (from original: lerp for gathering magnitude)
    if (d <= boid.perception) {
      centerPos += mix(position, otherPos, rcpPerception);
      numBoids++;
      accumAcceleration += mix(acceleration, boidsIn[i].acceleration, alignedAccelerationWeight);
    }

    if (d < closestDist) {
      closestDist = d;
      closestPos = otherPos;
      closestIndex = i;
    }
  }

  // Flocking: move toward center of perceived boids, or home if alone
  let rcpNumBoids = 1.0 / max(f32(numBoids), 1.0);
  var toCenter: vec3<f32>;
  if (numBoids > 0u) {
    toCenter = centerPos * rcpNumBoids - position;
  } else {
    toCenter = homePos - position;
  }
  let toCenterDir = normalize(toCenter + vec3<f32>(1e-6));
  let maxAccelerationToCenter = boid.maxAcceleration * toCenterDir;

  // Closest boid avoidance
  let closestBoid = boidsIn[closestIndex];
  let toClosestBoid = closestPos - position;
  let toClosestBoidDir = normalize(toClosestBoid + vec3<f32>(1e-6));
  let maxAccelerationToClosest = boid.maxAcceleration * toClosestBoidDir;

  // Collision influence (from original: Square(max(influence, 0) / dist))
  let closestInfluence = (closestBoid.influenceRadius + boid.influenceRadius) - closestDist;
  let closestInfluenceWeight = pow(max(closestInfluence, 0.0) / max(closestDist, 0.01), 2.0);

  // P2: Alignment (matching neighbors' acceleration, weighted by curiosity)
  var alignedAccel: vec3<f32>;
  if (numBoids > 0u) {
    alignedAccel = accumAcceleration * rcpNumBoids;
  } else {
    alignedAccel = acceleration;
  }
  acceleration = mix(alignedAccel, acceleration, boid.curiosity);

  // P0: Collision avoidance vs P1: Flock centering
  let steeringAcceleration = normalize(toClosestBoidDir + toCenterDir + vec3<f32>(1e-6)) * length(boid.maxAcceleration);
  if (closestInfluenceWeight == 0.0) {
    acceleration += maxAccelerationToCenter * sim.deltaTime;
  } else {
    acceleration += steeringAcceleration * boid.turnSpeed;
  }
  acceleration -= maxAccelerationToClosest * closestInfluenceWeight;

  // Terrain avoidance (from original: pull up from terrain)
  let terrainH = loadTerrainHeight(position.xy);
  let terrainAvoidance = 5.0;
  acceleration.z += max(terrainH + terrainAvoidance - position.z, 0.0) * 0.05 * boid.maxAcceleration.z;

  // Clamp acceleration
  let chaseAcceleration = boid.maxAcceleration + (closestDist / boid.perception) * boid.maxAcceleration * 2.0;
  acceleration = clamp(acceleration, -chaseAcceleration, chaseAcceleration);

  // Update velocity
  let velocity = clamp(boid.velocity + acceleration * sim.deltaTime, -vec3<f32>(boid.maxVelocity), vec3<f32>(boid.maxVelocity));

  // Write boid output
  boidsOut[index].velocity = velocity;
  boidsOut[index].acceleration = acceleration;
  boidsOut[index].influenceRadius = boid.influenceRadius;
  boidsOut[index].maxAcceleration = boid.maxAcceleration;
  boidsOut[index].turnSpeed = boid.turnSpeed;
  boidsOut[index].curiosity = boid.curiosity;
  boidsOut[index].maxVelocity = boid.maxVelocity;
  boidsOut[index].perception = boid.perception;

  // Apply position
  let newPosition = position + velocity * sim.deltaTime;

  // Process orientation (from original: yaw/pitch from velocity direction)
  let velocityDir = normalize(velocity + vec3<f32>(1e-6));
  let yaw = atan2(velocityDir.y, velocityDir.x);
  let pitch = acos(clamp(velocityDir.z, -1.0, 1.0)) - 1.5707963;

  let sinYaw = sin(yaw); let cosYaw = cos(yaw);
  let sinPitch = sin(pitch); let cosPitch = cos(pitch);

  // Build rotation matrix (from original)
  var outInstance: InstanceData;
  outInstance.col0 = vec4<f32>(cosYaw * cosPitch, -sinYaw, cosYaw * sinPitch, 0.0);
  outInstance.col1 = vec4<f32>(sinYaw * cosPitch, cosYaw, sinYaw * sinPitch, 0.0);
  outInstance.col2 = vec4<f32>(-sinPitch, 0.0, cosPitch, 0.0);
  outInstance.col3 = vec4<f32>(newPosition, 1.0);

  instancesOut[index] = outInstance;
}
`;

// Initialize boid attributes (from BoidAttributeInitializer_CS.hlsl)
const boidInitShader = `
struct BoidAttribs {
  acceleration: vec3<f32>,
  influenceRadius: f32,
  maxAcceleration: vec3<f32>,
  turnSpeed: f32,
  velocity: vec3<f32>,
  curiosity: f32,
  maxVelocity: f32,
  perception: f32,
};

struct InstanceData {
  col0: vec4<f32>,
  col1: vec4<f32>,
  col2: vec4<f32>,
  col3: vec4<f32>,
};

struct InitUniforms {
  boidCount: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> u: InitUniforms;
@group(0) @binding(1) var<storage, read_write> boids: array<BoidAttribs>;
@group(0) @binding(2) var<storage, read_write> instances: array<InstanceData>;

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
  if (index >= u32(u.boidCount)) {
    return;
  }

  let fi = f32(index);
  let seed0 = 1312.21551;
  let seed1 = 5441.35424;

  // Initialize boid attributes (from BoidAttributeInitializer_CS.hlsl)
  var b: BoidAttribs;
  b.acceleration = toSnorm3(hash3d(vec3<f32>(fi, fi, fi), seed0 - 104.351)) * 0.5;
  b.influenceRadius = (hash1d(fi, seed0) + 2.0) * 1.0;
  b.maxAcceleration = hash3d(1.0 / max(vec3<f32>(fi, fi, fi), vec3<f32>(1.0)), seed0 + 34.124) + vec3<f32>(8.0);
  b.turnSpeed = (fract(sin(fi * 127.1) * 43758.5453) + 2.0) * 0.0001;
  b.velocity = vec3<f32>(0.0);
  b.curiosity = clamp((hash1d(fi, seed1) + 0.2) * 0.25, 0.0, 1.0);
  b.maxVelocity = length(hash3d(b.maxAcceleration, seed1) + vec3<f32>(8.0));
  b.perception = (hash1d(b.curiosity, seed0) + 0.2) * 20.0;
  boids[index] = b;

  // Initialize instance: spherical distribution (from BoidInstanceInitializer_CS.hlsl)
  let center = vec3<f32>(0.0, 0.0, 20.0);
  let direction = toSnorm3(hash3d(vec3<f32>(fi, fi, fi), seed0));
  let radius = hash1d(fi, seed0) * 1.0;
  let translation = direction * radius + center;

  var inst: InstanceData;
  inst.col0 = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  inst.col1 = vec4<f32>(0.0, 1.0, 0.0, 0.0);
  inst.col2 = vec4<f32>(0.0, 0.0, 1.0, 0.0);
  inst.col3 = vec4<f32>(translation, 1.0);
  instances[index] = inst;
}
`;

// Instanced rendering (from BoidInstancing_VS.hlsl)
const boidRenderShader = `
struct RenderUniforms {
  viewProj: mat4x4<f32>,
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

struct InstanceData {
  col0: vec4<f32>,
  col1: vec4<f32>,
  col2: vec4<f32>,
  col3: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) instanceColor: vec3<f32>,
};

// Simple cone/arrow mesh for boid visualization (6 triangles = 18 verts)
fn getBoidVertex(vid: u32) -> vec3<f32> {
  // Arrow shape pointing +Z
  let s = 0.15;
  var verts = array<vec3<f32>, 18>(
    // Body (elongated tetrahedron)
    vec3<f32>(0.0, 0.0, s * 2.0),   vec3<f32>(-s, 0.0, -s),   vec3<f32>(s, 0.0, -s),
    vec3<f32>(0.0, 0.0, s * 2.0),   vec3<f32>(s, 0.0, -s),    vec3<f32>(0.0, s * 0.6, -s * 0.5),
    vec3<f32>(0.0, 0.0, s * 2.0),   vec3<f32>(0.0, s * 0.6, -s * 0.5), vec3<f32>(-s, 0.0, -s),
    vec3<f32>(-s, 0.0, -s),         vec3<f32>(s, 0.0, -s),    vec3<f32>(0.0, s * 0.6, -s * 0.5),
    // Wings
    vec3<f32>(0.0, 0.0, s * 0.5),   vec3<f32>(-s * 2.0, 0.0, -s * 0.5), vec3<f32>(0.0, 0.0, -s),
    vec3<f32>(0.0, 0.0, s * 0.5),   vec3<f32>(0.0, 0.0, -s),  vec3<f32>(s * 2.0, 0.0, -s * 0.5),
  );
  return verts[vid];
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let instanceIndex = vertexIndex / 18u;
  let localVertIndex = vertexIndex % 18u;

  let inst = instances[instanceIndex];
  let instanceModel = mat4x4<f32>(inst.col0, inst.col1, inst.col2, inst.col3);

  let localPos = getBoidVertex(localVertIndex);
  let worldPos = instanceModel * vec4<f32>(localPos, 1.0);

  var out: VSOut;
  out.position = u.viewProj * worldPos;

  // Simple normal from instance rotation
  let rotMat = mat3x3<f32>(inst.col0.xyz, inst.col1.xyz, inst.col2.xyz);
  out.worldNormal = normalize(rotMat * vec3<f32>(0.0, 0.0, 1.0));

  // Per-instance color from index
  let hue = fract(f32(instanceIndex) * 0.618033988);
  out.instanceColor = 0.5 + 0.5 * cos(6.28318 * (hue + vec3<f32>(0.0, 0.33, 0.67)));
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
  let NdotL = max(dot(normalize(in.worldNormal), lightDir), 0.0);
  let color = in.instanceColor * (0.2 + NdotL * 0.8);
  return vec4<f32>(pow(color, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;

export class BoidDemo implements Demo {
  label = "Boid";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private initPipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;

  private boidBuffers: GPUBuffer[] = [];
  private instanceBuffers: GPUBuffer[] = [];
  private simUniformBuffer!: GPUBuffer;
  private initUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;

  private computeBindGroups: GPUBindGroup[] = [];
  private initBindGroup!: GPUBindGroup;
  private renderBindGroups: GPUBindGroup[] = [];

  private terrain!: GPUTerrain;

  private current = 0;
  private initialized = false;
  private simData = new Float32Array(4);
  private renderData = new Float32Array(20);

  // Boid attribs: 8 vec4-aligned fields = 128 bytes per boid
  // Instance: 4x4 matrix = 64 bytes per instance
  private static BOID_STRIDE = 128;
  private static INSTANCE_STRIDE = 64;

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    this.terrain = new GPUTerrain(this.device, 256, 5.0);

    for (let i = 0; i < 2; i++) {
      this.boidBuffers.push(this.device.createBuffer({
        label: `boid-attribs-${i}`,
        size: BOID_COUNT * BoidDemo.BOID_STRIDE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
      this.instanceBuffers.push(this.device.createBuffer({
        label: `boid-instances-${i}`,
        size: BOID_COUNT * BoidDemo.INSTANCE_STRIDE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }));
    }

    this.simUniformBuffer = this.device.createBuffer({
      label: "boid-sim-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.initUniformBuffer = this.device.createBuffer({
      label: "boid-init-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.renderUniformBuffer = this.device.createBuffer({
      label: "boid-render-ubo",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute pipeline
    const computeModule = this.device.createShaderModule({ code: boidComputeShader });
    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: {} },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: {} },
      ],
    });
    this.computePipeline = this.device.createComputePipeline({
      label: "boid-compute",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Init pipeline
    const initModule = this.device.createShaderModule({ code: boidInitShader });
    const initBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.initPipeline = this.device.createComputePipeline({
      label: "boid-init",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [initBGL] }),
      compute: { module: initModule, entryPoint: "cs_init" },
    });

    // Render pipeline
    const renderModule = this.device.createShaderModule({ code: boidRenderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "boid-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    // Bind groups
    const terrainSampler = this.terrain.heightSampler;
    for (let i = 0; i < 2; i++) {
      this.computeBindGroups.push(this.device.createBindGroup({
        layout: computeBGL,
        entries: [
          { binding: 0, resource: { buffer: this.simUniformBuffer } },
          { binding: 1, resource: { buffer: this.boidBuffers[i] } },
          { binding: 2, resource: { buffer: this.boidBuffers[1 - i] } },
          { binding: 3, resource: { buffer: this.instanceBuffers[i] } },
          { binding: 4, resource: { buffer: this.instanceBuffers[1 - i] } },
          { binding: 5, resource: this.terrain.view },
          { binding: 6, resource: terrainSampler },
        ],
      }));
    }

    this.initBindGroup = this.device.createBindGroup({
      layout: initBGL,
      entries: [
        { binding: 0, resource: { buffer: this.initUniformBuffer } },
        { binding: 1, resource: { buffer: this.boidBuffers[0] } },
        { binding: 2, resource: { buffer: this.instanceBuffers[0] } },
      ],
    });

    for (let i = 0; i < 2; i++) {
      this.renderBindGroups.push(this.device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderUniformBuffer } },
          { binding: 1, resource: { buffer: this.instanceBuffers[i] } },
        ],
      }));
    }
  }

  update(time: number, deltaTime: number) {
    const dt = Math.min(deltaTime, 0.05);
    this.simData[0] = dt;
    this.simData[1] = time;
    this.simData[2] = BOID_COUNT;
    this.simData[3] = 0;
    this.device.queue.writeBuffer(this.simUniformBuffer, 0, this.simData as unknown as GPUAllowSharedBufferSource);

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    this.renderData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.renderData[16] = time;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        if (!this.initialized) {
          this.terrain.dispatchOnce(encoder);

          this.device.queue.writeBuffer(this.initUniformBuffer, 0, new Float32Array([BOID_COUNT, 0, 0, 0]) as unknown as GPUAllowSharedBufferSource);
          const initPass = encoder.beginComputePass();
          initPass.setPipeline(this.initPipeline);
          initPass.setBindGroup(0, this.initBindGroup);
          initPass.dispatchWorkgroups(Math.ceil(BOID_COUNT / 256));
          initPass.end();
          this.initialized = true;
        }

        // Simulate
        const simPass = encoder.beginComputePass();
        simPass.setPipeline(this.computePipeline);
        simPass.setBindGroup(0, this.computeBindGroups[this.current]);
        simPass.dispatchWorkgroups(Math.ceil(BOID_COUNT / 256));
        simPass.end();

        this.current = 1 - this.current;

        // Render
        const renderPass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0.03, g: 0.03, b: 0.08, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroups[this.current]);
        renderPass.draw(BOID_COUNT * 18);
        renderPass.end();
      },
    }];
  }

  stats() {
    return {
      drawCalls: 1,
      instances: BOID_COUNT,
      triangles: BOID_COUNT * 6,
      computeDispatches: 2,
      custom: {
        "Algorithm": "Reynolds Flocking",
        "Terrain": "GPU avoidance",
        "Orientation": "Yaw/Pitch matrix",
      },
    };
  }

  registerGUI(gui: any) {}

  destroy() {
    for (const b of this.boidBuffers) b.destroy();
    for (const b of this.instanceBuffers) b.destroy();
    this.simUniformBuffer.destroy();
    this.initUniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.terrain.destroy();
  }
}
