import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { GPUWindField } from "../utils/gpu-wind-field";

const PARTICLE_COUNT = 4096;

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
  windFieldSize: f32,
};

@group(0) @binding(0) var<uniform> sim: SimUniforms;
@group(0) @binding(1) var<storage, read> particlesIn: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut: array<Particle>;
@group(0) @binding(3) var windTex: texture_2d<f32>;
@group(0) @binding(4) var windSampler: sampler;

fn hash3d(xyz: vec3<f32>, seed: f32) -> vec3<f32> {
  return fract(sin(vec3<f32>(
    dot(xyz, vec3<f32>(83.7247, 71.7823, 24.274)),
    dot(xyz, vec3<f32>(64.4634, 49.4349, 82.263)),
    dot(xyz, vec3<f32>(94.262, 20.9245, 34.8256))
  )) * 52567.0925 + seed);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= u32(sim.particleCount)) {
    return;
  }

  let pIn = particlesIn[index];
  var pOut: Particle;

  let boundaryMin = vec3<f32>(-12.0, -12.0, -1.0);
  let boundaryMax = vec3<f32>(12.0, 12.0, 12.0);

  var deltaElapse = sim.deltaTime;
  var pos = pIn.position + pIn.velocity * sim.deltaTime;

  if (any(pos < boundaryMin) || any(pos > boundaryMax)) {
    deltaElapse *= 4.0;
  }

  pOut.elapseTime = pIn.elapseTime + deltaElapse;

  if (pIn.elapseTime > pIn.maxLifeTime) {
    pOut.elapseTime = 0.0;
    let rnd = hash3d(vec3<f32>(f32(index), f32(index) * 0.7, f32(index) * 1.3), 1312.21551 + sim.time);
    pos = (rnd * 2.0 - 1.0) * 10.0;
    pos.z = abs(pos.z) * 0.5;
    pOut.velocity = vec3<f32>(0.0);
  } else {
    var vel = pIn.velocity;
    vel.z -= 0.3 * sim.deltaTime;

    // Sample GPU wind field texture
    let windUV = vec2<f32>(pos.x / 24.0 + 0.5, pos.y / 24.0 + 0.5);
    let windSample = textureSampleLevel(windTex, windSampler, windUV, 0.0);
    let wind = vec3<f32>(windSample.r, windSample.g, 0.0) * 0.4;
    vel += wind * sim.deltaTime;

    vel *= 0.999;
    let maxSpeed = 2.0;
    let speed = length(vel);
    if (speed > maxSpeed) {
      vel = vel / speed * maxSpeed;
    }
    pOut.velocity = vel;
  }

  pOut.position = pos;

  let fadeThreshold = 0.25;
  let lifeRatio = min(
    min(pIn.maxLifeTime - pIn.elapseTime, pIn.elapseTime) / pIn.maxLifeTime,
    fadeThreshold
  );
  let alpha = lifeRatio * (1.0 / fadeThreshold);

  let hue = fract(f32(index) / sim.particleCount + sim.time * 0.05);
  let rgb = 0.5 + 0.5 * cos(6.28318 * (hue + vec3<f32>(0.0, 0.33, 0.67)));
  pOut.color = vec4<f32>(rgb, alpha);
  pOut.maxLifeTime = pIn.maxLifeTime;

  particlesOut[index] = pOut;
}
`;

const renderShader = `
struct RenderUniforms {
  viewProj: mat4x4<f32>,
  resolution: vec2<f32>,
  time: f32,
  pad: f32,
};

struct Particle {
  position: vec3<f32>,
  elapseTime: f32,
  velocity: vec3<f32>,
  maxLifeTime: f32,
  color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<storage, read> particles: array<Particle>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) @interpolate(flat) center: vec2<f32>,
  @location(2) @interpolate(flat) pointRadius: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let particleIndex = vertexIndex / 6u;
  let cornerIndex = vertexIndex % 6u;

  let p = particles[particleIndex];
  let clipPos = ru.viewProj * vec4<f32>(p.position, 1.0);

  let ndc = clipPos.xy / clipPos.w;
  let screenPos = (ndc * 0.5 + 0.5) * ru.resolution;

  var pointSize = 0.15 / max(clipPos.w, 0.1) * ru.resolution.y;
  pointSize = clamp(pointSize, 2.0, 64.0);

  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );

  let offset = corners[cornerIndex] * pointSize;
  let finalScreen = screenPos + offset;
  let finalNdc = (finalScreen / ru.resolution) * 2.0 - 1.0;

  var out: VSOut;
  out.position = vec4<f32>(finalNdc * clipPos.w, clipPos.z, clipPos.w);
  out.color = p.color;
  out.center = screenPos;
  out.pointRadius = pointSize;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let screenPos = in.position.xy;
  let dist = distance(in.center, screenPos);
  let circleAlpha = smoothstep(1.0, 0.3, dist / in.pointRadius);
  let alpha = in.color.a * circleAlpha;
  if (alpha < 0.01) {
    discard;
  }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;

export class ParticleDemo implements Demo {
  label = "Particles";

  private device!: GPUDevice;
  private format!: GPUTextureFormat;
  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private particleBuffers: GPUBuffer[] = [];
  private simUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private computeBindGroups: GPUBindGroup[] = [];
  private camera!: Camera;
  private ctx!: GPUContext;
  private current = 0;
  private windField!: GPUWindField;
  private windSampler!: GPUSampler;

  init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.format = ctx.format;
    this.camera = camera;
    this.ctx = ctx;

    this.windField = new GPUWindField(this.device, 64);
    this.windSampler = this.windField.createSampler();

    const particleSize = 12 * 4;
    const initData = new Float32Array(PARTICLE_COUNT * 12);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const base = i * 12;
      initData[base + 0] = (Math.random() * 2 - 1) * 10;
      initData[base + 1] = (Math.random() * 2 - 1) * 10;
      initData[base + 2] = Math.random() * 8;
      initData[base + 3] = Math.random() * 3;
      initData[base + 4] = (Math.random() - 0.5) * 0.5;
      initData[base + 5] = (Math.random() - 0.5) * 0.5;
      initData[base + 6] = Math.random() * 0.3;
      initData[base + 7] = 3 + Math.random() * 4;
      initData[base + 8] = 1;
      initData[base + 9] = 1;
      initData[base + 10] = 1;
      initData[base + 11] = 1;
    }

    for (let i = 0; i < 2; i++) {
      this.particleBuffers.push(
        this.device.createBuffer({
          label: `particles-${i}`,
          size: PARTICLE_COUNT * particleSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        })
      );
      new Float32Array(this.particleBuffers[i].getMappedRange()).set(initData);
      this.particleBuffers[i].unmap();
    }

    this.simUniformBuffer = this.device.createBuffer({
      label: "sim-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUniformBuffer = this.device.createBuffer({
      label: "render-ubo",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: "particle-compute",
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: computeShader }),
        entryPoint: "cs_main",
      },
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: "particle-render",
      layout: "auto",
      vertex: {
        module: this.device.createShaderModule({ code: renderShader }),
        entryPoint: "vs_main",
      },
      fragment: {
        module: this.device.createShaderModule({ code: renderShader }),
        entryPoint: "fs_main",
        targets: [
          {
            format: this.format,
            blend: {
              color: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });

    for (let i = 0; i < 2; i++) {
      this.computeBindGroups.push(
        this.device.createBindGroup({
          layout: this.computePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.simUniformBuffer } },
            { binding: 1, resource: { buffer: this.particleBuffers[i] } },
            { binding: 2, resource: { buffer: this.particleBuffers[1 - i] } },
            { binding: 3, resource: this.windField.texture.createView() },
            { binding: 4, resource: this.windSampler },
          ],
        })
      );
    }

  }

  private simData = new Float32Array(4);
  private renderData = new Float32Array(20);
  private renderBindGroups: GPUBindGroup[] = [];

  update(time: number, deltaTime: number) {
    const dt = Math.min(deltaTime, 0.05);
    this.simData[0] = dt;
    this.simData[1] = time;
    this.simData[2] = PARTICLE_COUNT;
    this.simData[3] = 0;
    this.device.queue.writeBuffer(this.simUniformBuffer, 0, this.simData as unknown as GPUAllowSharedBufferSource);

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);
    this.renderData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.renderData[16] = w;
    this.renderData[17] = h;
    this.renderData[18] = time;
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
          { binding: 1, resource: { buffer: this.particleBuffers[i] } },
        ],
      });
    }
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    this.ensureRenderBindGroups();

    // Update wind field on GPU before particle sim
    this.windField.dispatch(encoder, performance.now() / 1000);

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.computeBindGroups[this.current]);
    computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 256));
    computePass.end();

    this.current = 1 - this.current;

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroups[this.current]);
    renderPass.draw(PARTICLE_COUNT * 6);
    renderPass.end();
  }

  stats() {
    return {
      drawCalls: 1,
      instances: PARTICLE_COUNT,
      triangles: PARTICLE_COUNT * 2,
      computeDispatches: 2,
      custom: { "Wind Field": "GPU 64x64 rg32f", "Billboard": "6 verts/particle" },
    };
  }

  destroy() {
    for (const b of this.particleBuffers) b.destroy();
    this.simUniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.windField.destroy();
  }
}
