import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { GPUTerrain } from "../utils/gpu-terrain";

const BOID_COUNT = 512;

const computeShader = `
struct Boid {
  position: vec3<f32>,
  maxVelocity: f32,
  velocity: vec3<f32>,
  maxAcceleration: f32,
  acceleration: vec3<f32>,
  perception: f32,
};

struct SimUniforms {
  deltaTime: f32,
  time: f32,
  boidCount: f32,
  terrainSize: f32,
};

@group(0) @binding(0) var<uniform> sim: SimUniforms;
@group(0) @binding(1) var<storage, read> boidsIn: array<Boid>;
@group(0) @binding(2) var<storage, read_write> boidsOut: array<Boid>;
@group(0) @binding(3) var terrainTex: texture_2d<f32>;
@group(0) @binding(4) var terrainSampler: sampler;

fn sampleTerrainHeight(worldXY: vec2<f32>) -> f32 {
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

  let boid = boidsIn[index];
  let position = boid.position;
  var acceleration = boid.acceleration;

  let homePos = vec3<f32>(sin(sim.time * 0.2) * 8.0, cos(sim.time * 0.2) * 8.0, 4.0);

  var centerPos = vec3<f32>(0.0);
  var centerVel = vec3<f32>(0.0);
  var numNeighbors = 0u;
  var closestDist = 1e10;
  var closestPos = position;

  let rcpPerception = 1.0 / boid.perception;

  for (var i = 0u; i < count; i++) {
    if (i == index) {
      continue;
    }
    let other = boidsIn[i];
    let d = distance(position, other.position);

    if (d <= boid.perception) {
      centerPos += mix(position, other.position, rcpPerception);
      centerVel += other.velocity;
      numNeighbors++;
    }
    if (d < closestDist) {
      closestDist = d;
      closestPos = other.position;
    }
  }

  var toCenter: vec3<f32>;
  if (numNeighbors > 0u) {
    toCenter = centerPos / f32(numNeighbors) - position;
  } else {
    toCenter = homePos - position;
  }
  let toCenterDir = normalize(toCenter + vec3<f32>(1e-6));

  let toClosest = closestPos - position;
  let toClosestDir = normalize(toClosest + vec3<f32>(1e-6));

  let influenceRadius = 0.8;
  let closestInfluence = max((influenceRadius * 2.0) - closestDist, 0.0);
  let closestWeight = closestInfluence * closestInfluence / max(closestDist, 0.01);

  if (numNeighbors > 0u) {
    let alignedVel = centerVel / f32(numNeighbors);
    acceleration = mix(alignedVel * 0.1, acceleration, 0.5);
  }

  let steering = normalize(toClosestDir + toCenterDir + vec3<f32>(1e-6)) * boid.maxAcceleration;
  if (closestWeight > 0.0) {
    acceleration += steering * 0.5;
    acceleration -= toClosestDir * boid.maxAcceleration * closestWeight;
  } else {
    acceleration += toCenterDir * boid.maxAcceleration * sim.deltaTime;
  }

  // Terrain avoidance: sample height at boid XY, push up if too close
  let terrainHeight = sampleTerrainHeight(position.xy);
  let clearance = position.z - terrainHeight;
  let avoidStrength = max(2.0 - clearance, 0.0) * boid.maxAcceleration * 0.8;
  acceleration.z += avoidStrength;

  let chaseAccel = boid.maxAcceleration * 3.0;
  acceleration = clamp(acceleration, vec3<f32>(-chaseAccel), vec3<f32>(chaseAccel));

  var velocity = clamp(
    boid.velocity + acceleration * sim.deltaTime,
    vec3<f32>(-boid.maxVelocity),
    vec3<f32>(boid.maxVelocity)
  );

  var newPos = position + velocity * sim.deltaTime;

  // Hard floor: never go below terrain
  let newTerrainH = sampleTerrainHeight(newPos.xy);
  if (newPos.z < newTerrainH + 0.3) {
    newPos.z = newTerrainH + 0.3;
    velocity.z = max(velocity.z, 0.0);
  }

  let bound = 15.0;
  if (abs(newPos.x) > bound || abs(newPos.y) > bound || newPos.z > 12.0) {
    let toHome = normalize(homePos - newPos + vec3<f32>(1e-6));
    velocity = toHome * boid.maxVelocity * 0.5;
    newPos = position + velocity * sim.deltaTime;
  }

  var out: Boid;
  out.position = newPos;
  out.velocity = velocity;
  out.acceleration = acceleration * 0.95;
  out.maxVelocity = boid.maxVelocity;
  out.maxAcceleration = boid.maxAcceleration;
  out.perception = boid.perception;
  boidsOut[index] = out;
}
`;

const renderShader = `
struct RenderUniforms {
  viewProj: mat4x4<f32>,
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

struct Boid {
  position: vec3<f32>,
  maxVelocity: f32,
  velocity: vec3<f32>,
  maxAcceleration: f32,
  acceleration: vec3<f32>,
  perception: f32,
};

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<storage, read> boids: array<Boid>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let boidIndex = vertexIndex / 9u;
  let vertIndex = vertexIndex % 9u;

  let b = boids[boidIndex];
  let velDir = normalize(b.velocity + vec3<f32>(1e-6));

  var right = normalize(cross(velDir, vec3<f32>(0.0, 0.0, 1.0)) + vec3<f32>(1e-6));
  let up = normalize(cross(right, velDir));

  let size = 0.15;
  var localPos: vec3<f32>;
  if (vertIndex < 3u) {
    let triVerts = array<vec3<f32>, 3>(
      vec3<f32>(0.0, 0.0, size * 2.0),
      vec3<f32>(-size, 0.0, -size),
      vec3<f32>(size, 0.0, -size),
    );
    localPos = triVerts[vertIndex];
  } else if (vertIndex < 6u) {
    let triVerts = array<vec3<f32>, 3>(
      vec3<f32>(0.0, 0.0, size * 2.0),
      vec3<f32>(size, 0.0, -size),
      vec3<f32>(0.0, size * 0.6, -size * 0.5),
    );
    localPos = triVerts[vertIndex - 3u];
  } else {
    let triVerts = array<vec3<f32>, 3>(
      vec3<f32>(0.0, 0.0, size * 2.0),
      vec3<f32>(0.0, size * 0.6, -size * 0.5),
      vec3<f32>(-size, 0.0, -size),
    );
    localPos = triVerts[vertIndex - 6u];
  }

  let worldOffset = right * localPos.x + up * localPos.y + velDir * localPos.z;
  let worldPos = b.position + worldOffset;

  var out: VSOut;
  out.position = ru.viewProj * vec4<f32>(worldPos, 1.0);

  let hue = fract(f32(boidIndex) * 0.00618 + ru.time * 0.02);
  out.color = 0.5 + 0.5 * cos(6.28318 * (hue + vec3<f32>(0.0, 0.33, 0.67)));
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
`;

export class BoidDemo implements Demo {
  label = "Boid";

  private device!: GPUDevice;
  private format!: GPUTextureFormat;
  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private boidBuffers: GPUBuffer[] = [];
  private simUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private computeBindGroups: GPUBindGroup[] = [];
  private camera!: Camera;
  private ctx!: GPUContext;
  private current = 0;
  private depthTexture: GPUTexture | null = null;
  private terrain!: GPUTerrain;
  private terrainDispatched = false;

  init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.format = ctx.format;
    this.camera = camera;
    this.ctx = ctx;

    this.terrain = new GPUTerrain(this.device, 256, 5.0);

    const boidFloats = 12;
    const initData = new Float32Array(BOID_COUNT * boidFloats);
    for (let i = 0; i < BOID_COUNT; i++) {
      const base = i * boidFloats;
      initData[base + 0] = (Math.random() * 2 - 1) * 8;
      initData[base + 1] = (Math.random() * 2 - 1) * 8;
      initData[base + 2] = Math.random() * 6;
      initData[base + 3] = 3.0 + Math.random() * 2.0;
      initData[base + 4] = (Math.random() - 0.5) * 2;
      initData[base + 5] = (Math.random() - 0.5) * 2;
      initData[base + 6] = (Math.random() - 0.5) * 1;
      initData[base + 7] = 4.0;
      initData[base + 8] = 0;
      initData[base + 9] = 0;
      initData[base + 10] = 0;
      initData[base + 11] = 4.0 + Math.random() * 2.0;
    }

    for (let i = 0; i < 2; i++) {
      this.boidBuffers.push(
        this.device.createBuffer({
          label: `boids-${i}`,
          size: BOID_COUNT * boidFloats * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        })
      );
      new Float32Array(this.boidBuffers[i].getMappedRange()).set(initData);
      this.boidBuffers[i].unmap();
    }

    this.simUniformBuffer = this.device.createBuffer({
      label: "boid-sim-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUniformBuffer = this.device.createBuffer({
      label: "boid-render-ubo",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: "boid-compute",
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: computeShader }),
        entryPoint: "cs_main",
      },
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: "boid-render",
      layout: "auto",
      vertex: {
        module: this.device.createShaderModule({ code: renderShader }),
        entryPoint: "vs_main",
      },
      fragment: {
        module: this.device.createShaderModule({ code: renderShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    for (let i = 0; i < 2; i++) {
      this.computeBindGroups.push(
        this.device.createBindGroup({
          layout: this.computePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.simUniformBuffer } },
            { binding: 1, resource: { buffer: this.boidBuffers[i] } },
            { binding: 2, resource: { buffer: this.boidBuffers[1 - i] } },
            { binding: 3, resource: this.terrain.view },
            { binding: 4, resource: this.terrain.heightSampler },
          ],
        })
      );
    }
  }

  private ensureDepth(ctx: GPUContext) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    if (this.depthTexture && this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  private simData = new Float32Array(4);
  private renderData = new Float32Array(20);
  private renderBindGroups: GPUBindGroup[] = [];
  private cachedDepthView: GPUTextureView | null = null;

  update(time: number, deltaTime: number) {
    const dt = Math.min(deltaTime, 0.05);
    this.simData[0] = dt;
    this.simData[1] = time;
    this.simData[2] = BOID_COUNT;
    this.simData[3] = 0;
    this.device.queue.writeBuffer(this.simUniformBuffer, 0, this.simData as unknown as GPUAllowSharedBufferSource);

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);
    this.renderData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.renderData[16] = time;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderData as unknown as GPUAllowSharedBufferSource);
  }

  private ensureRenderBindGroups() {
    if (this.renderBindGroups.length === 2) return;
    const layout = this.renderPipeline.getBindGroupLayout(0);
    for (let i = 0; i < 2; i++) {
      this.renderBindGroups[i] = this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.renderUniformBuffer } },
          { binding: 1, resource: { buffer: this.boidBuffers[i] } },
        ],
      });
    }
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    this.ensureDepth(this.ctx);
    this.ensureRenderBindGroups();

    // Generate terrain heightmap once on GPU
    if (!this.terrainDispatched) {
      this.terrain.dispatchOnce(encoder);
      this.terrainDispatched = true;
    }

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.computeBindGroups[this.current]);
    computePass.dispatchWorkgroups(Math.ceil(BOID_COUNT / 256));
    computePass.end();

    this.current = 1 - this.current;

    if (!this.cachedDepthView) this.cachedDepthView = this.depthTexture!.createView();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.03, g: 0.03, b: 0.08, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.cachedDepthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroups[this.current]);
    renderPass.draw(BOID_COUNT * 9);
    renderPass.end();
  }

  stats() {
    return {
      drawCalls: 1,
      instances: BOID_COUNT,
      triangles: BOID_COUNT * 3,
      computeDispatches: 2,
      custom: { "Algorithm": "Sep+Align+Cohesion", "Terrain": "GPU 256x256 r32f" },
    };
  }

  destroy() {
    for (const b of this.boidBuffers) b.destroy();
    this.simUniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.depthTexture?.destroy();
    this.terrain.destroy();
  }
}
