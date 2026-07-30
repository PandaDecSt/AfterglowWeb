import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import type { RenderPass } from "../core/renderer";

const GRID_SIZE = 64;

// GlitchedMosaics: compute shader generates glitched mosaic pattern
const glitchComputeShader = `
struct GlitchUniforms {
  time: f32,
  gridSize: f32,
  glitchIntensity: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> u: GlitchUniforms;
@group(0) @binding(1) var outputTex: texture_storage_2d<rgba8unorm, write>;

fn hash1d(x: f32, seed: f32) -> f32 {
  return fract(sin(x * 127.1 + seed) * 43758.5453);
}

fn hash2d(p: vec2<f32>, seed: f32) -> vec2<f32> {
  return fract(sin(vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)))) * 43758.5453 + seed);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gs = u32(u.gridSize);
  if (gid.x >= gs || gid.y >= gs) {
    return;
  }

  let uv = vec2<f32>(f32(gid.x), f32(gid.y)) / vec2<f32>(f32(gs), f32(gs));
  let t = u.time;
  let intensity = u.glitchIntensity;

  // Base mosaic grid
  let mosaicSize = 8.0 + floor(hash1d(floor(t * 2.0), 42.0) * 8.0);
  let cell = floor(uv * mosaicSize);
  let cellRnd = hash2d(cell, floor(t * 3.0));

  // Glitch displacement
  let glitchLine = step(1.0 - intensity * 0.3, hash1d(floor(uv.y * mosaicSize), floor(t * 8.0)));
  let displacement = glitchLine * (cellRnd.x - 0.5) * intensity * 0.2;

  // Color channels with RGB split
  let shiftedUV = uv + vec2<f32>(displacement, 0.0);
  let r = hash1d(dot(floor(shiftedUV * mosaicSize), vec2<f32>(1.0, 0.0)), floor(t * 4.0));
  let g = hash1d(dot(floor(uv * mosaicSize), vec2<f32>(0.0, 1.0)), floor(t * 5.0) + 1.0);
  let b = hash1d(dot(floor(shiftedUV * mosaicSize + 0.5), vec2<f32>(1.0, 1.0)), floor(t * 6.0) + 2.0);

  // Scanline effect
  let scanline = step(0.5, fract(uv.y * f32(gs) * 0.5)) * 0.1;

  // Block corruption
  let corruptBlock = step(1.0 - intensity * 0.1, hash1d(dot(cell, vec2<f32>(7.0, 13.0)), floor(t * 10.0)));
  let corruptColor = vec3<f32>(hash1d(cellRnd.x * 100.0, t), hash1d(cellRnd.y * 100.0, t + 1.0), hash1d(cellRnd.x + cellRnd.y, t + 2.0));

  var color = vec3<f32>(r, g, b) * 0.7 + 0.15;
  color = mix(color, corruptColor, corruptBlock * 0.8);
  color -= scanline;

  // Digital noise
  let noise = hash1d(dot(uv, vec2<f32>(f32(gs), f32(gs))) + t * 100.0, 99.0) * intensity * 0.15;
  color += noise;

  textureStore(outputTex, gid.xy, vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

// Fullscreen display
const glitchRenderShader = `
struct RenderUniforms {
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var displayTex: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
  );
  var out: VSOut;
  out.position = vec4<f32>(pos[vi], 0.0, 1.0);
  out.uv = pos[vi] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  var uv = in.uv;
  // Subtle time-based UV jitter
  uv.x += sin(uv.y * 50.0 + u.time * 3.0) * 0.001;
  let color = textureSample(displayTex, texSampler, uv);
  return color;
}
`;

export class GlitchedMosaicsDemo implements Demo {
  label = "GlitchedMosaics";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private outputTexture!: GPUTexture;
  private uniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;
  private uniformData = new Float32Array(4);

  glitchIntensity = 0.5;

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    this.outputTexture = this.device.createTexture({
      label: "glitch-output",
      size: [GRID_SIZE, GRID_SIZE],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "glitch-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUniformBuffer = this.device.createBuffer({
      label: "glitch-render-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute pipeline
    const computeModule = this.device.createShaderModule({ code: glitchComputeShader });
    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
      ],
    });
    this.computePipeline = this.device.createComputePipeline({
      label: "glitch-compute",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Render pipeline
    const renderModule = this.device.createShaderModule({ code: glitchRenderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "glitch-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    const sampler = this.device.createSampler({ magFilter: "nearest", minFilter: "nearest" });

    this.computeBindGroup = this.device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.outputTexture.createView() },
      ],
    });

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: this.outputTexture.createView() },
        { binding: 2, resource: sampler },
      ],
    });
  }

  update(time: number) {
    this.uniformData[0] = time;
    this.uniformData[1] = GRID_SIZE;
    this.uniformData[2] = this.glitchIntensity;
    this.uniformData[3] = 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, new Float32Array([time, 0, 0, 0]) as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(GRID_SIZE / 8), Math.ceil(GRID_SIZE / 8));
        computePass.end();

        const renderPass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(3);
        renderPass.end();
      },
    }];
  }

  stats() {
    return {
      drawCalls: 1,
      computeDispatches: 1,
      custom: {
        "Grid": `${GRID_SIZE}x${GRID_SIZE}`,
        "Effects": "RGB Split + Scanline + Corruption",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "glitchIntensity", 0, 1, 0.01).name("Glitch Intensity");
  }

  destroy() {
    this.outputTexture.destroy();
    this.uniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
  }
}
