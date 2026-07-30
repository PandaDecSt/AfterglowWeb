import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import type { RenderPass } from "../core/renderer";

const GRID_SIZE = 128;
const VERTEX_COUNT = GRID_SIZE * GRID_SIZE;
const INDEX_COUNT = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6;
const VERTEX_STRIDE = 48; // WGSL aligned: vec3+vec3+vec2

// Water simulation: height field + wave propagation via compute
const waterComputeShader = `
struct WaterUniforms {
  time: f32,
  gridSize: f32,
  waveSpeed: f32,
  damping: f32,
};

struct WaterVertex {
  position: vec3<f32>,
  normal: vec3<f32>,
  uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: WaterUniforms;
@group(0) @binding(1) var<storage, read_write> vertices: array<WaterVertex>;
@group(0) @binding(2) var<storage, read> heightCurr: array<f32>;
@group(0) @binding(3) var<storage, read_write> heightNext: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gs = u32(u.gridSize);
  if (gid.x >= gs || gid.y >= gs) {
    return;
  }

  let idx = gid.y * gs + gid.x;
  let nx = f32(gid.x) / f32(gs - 1u);
  let nz = f32(gid.y) / f32(gs - 1u);
  let worldX = (nx - 0.5) * 20.0;
  let worldZ = (nz - 0.5) * 20.0;

  // Wave simulation: sum of sine waves + interference
  let t = u.time * u.waveSpeed;
  var h = 0.0;
  h += sin(worldX * 1.5 + t * 1.2) * 0.15;
  h += sin(worldZ * 2.0 + t * 0.8) * 0.1;
  h += sin((worldX + worldZ) * 0.8 + t * 1.5) * 0.08;
  h += sin(worldX * 3.0 - worldZ * 1.5 + t * 2.0) * 0.05;
  h += sin(length(vec2<f32>(worldX, worldZ)) * 2.0 - t * 1.8) * 0.12;

  // Compute normal via finite differences
  let eps = 0.1;
  let hR = sin((worldX + eps) * 1.5 + t * 1.2) * 0.15 + sin(worldZ * 2.0 + t * 0.8) * 0.1 + sin((worldX + eps + worldZ) * 0.8 + t * 1.5) * 0.08;
  let hL = sin((worldX - eps) * 1.5 + t * 1.2) * 0.15 + sin(worldZ * 2.0 + t * 0.8) * 0.1 + sin((worldX - eps + worldZ) * 0.8 + t * 1.5) * 0.08;
  let hU = sin(worldX * 1.5 + t * 1.2) * 0.15 + sin((worldZ + eps) * 2.0 + t * 0.8) * 0.1 + sin((worldX + worldZ + eps) * 0.8 + t * 1.5) * 0.08;
  let hD = sin(worldX * 1.5 + t * 1.2) * 0.15 + sin((worldZ - eps) * 2.0 + t * 0.8) * 0.1 + sin((worldX + worldZ - eps) * 0.8 + t * 1.5) * 0.08;

  let normal = normalize(vec3<f32>(hL - hR, 2.0 * eps, hD - hU));

  var v: WaterVertex;
  v.position = vec3<f32>(worldX, h, worldZ);
  v.normal = normal;
  v.uv = vec2<f32>(nx, nz);
  vertices[idx] = v;

  heightNext[idx] = h;
}
`;

// Water rendering with Fresnel reflection + depth-based color
const waterRenderShader = `
struct WaterUniforms {
  viewProj: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

struct WaterVertex {
  position: vec3<f32>,
  normal: vec3<f32>,
  uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: WaterUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<WaterVertex>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldPos: vec3<f32>,
  @location(1) worldNormal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let idx = indices[vertexIndex];
  let v = vertices[idx];

  var out: VSOut;
  out.position = u.viewProj * vec4<f32>(v.position, 1.0);
  out.worldPos = v.position;
  out.worldNormal = v.normal;
  out.uv = v.uv;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);
  let L = normalize(vec3<f32>(0.5, 1.0, 0.3));
  let H = normalize(V + L);

  // Fresnel (Schlick approximation)
  let F0 = 0.02;
  let fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);

  // Specular (Blinn-Phong)
  let NdotH = max(dot(N, H), 0.0);
  let spec = pow(NdotH, 128.0) * 1.5;

  // Water color: deep blue-green with depth variation
  let deepColor = vec3<f32>(0.0, 0.1, 0.2);
  let shallowColor = vec3<f32>(0.0, 0.3, 0.4);
  let heightFactor = clamp(in.worldPos.y * 2.0 + 0.5, 0.0, 1.0);
  let waterColor = mix(deepColor, shallowColor, heightFactor);

  // Sky reflection color
  let skyColor = vec3<f32>(0.4, 0.6, 0.9);

  // Combine: diffuse + fresnel reflection + specular
  let diffuse = waterColor * max(dot(N, L), 0.0) * 0.6;
  let reflection = skyColor * fresnel;
  let finalColor = diffuse + reflection + spec;

  // Subtle caustic pattern
  let caustic = sin(in.uv.x * 40.0 + u.time * 2.0) * sin(in.uv.y * 40.0 + u.time * 1.5) * 0.03;

  return vec4<f32>(pow(finalColor + caustic, vec3<f32>(1.0 / 2.2)), 0.9);
}
`;

export class WaterDemo implements Demo {
  label = "Water";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private heightBuffer!: GPUBuffer;
  private heightNextBuffer!: GPUBuffer;
  private computeUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;

  private computeUniformData = new Float32Array(4);
  private renderUniformData = new Float32Array(24);

  waveSpeed = 1.0;

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    this.vertexBuffer = this.device.createBuffer({
      label: "water-vb",
      size: VERTEX_COUNT * VERTEX_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.indexBuffer = this.device.createBuffer({
      label: "water-ib",
      size: INDEX_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
    });

    this.heightBuffer = this.device.createBuffer({
      label: "water-height-curr",
      size: VERTEX_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.heightNextBuffer = this.device.createBuffer({
      label: "water-height-next",
      size: VERTEX_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.computeUniformBuffer = this.device.createBuffer({
      label: "water-compute-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUniformBuffer = this.device.createBuffer({
      label: "water-render-ubo",
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Generate indices on CPU (static topology)
    const indices = new Uint32Array(INDEX_COUNT);
    let idx = 0;
    for (let y = 0; y < GRID_SIZE - 1; y++) {
      for (let x = 0; x < GRID_SIZE - 1; x++) {
        const a = y * GRID_SIZE + x;
        const b = a + 1;
        const c = a + GRID_SIZE;
        const d = c + 1;
        indices[idx++] = a; indices[idx++] = c; indices[idx++] = b;
        indices[idx++] = b; indices[idx++] = c; indices[idx++] = d;
      }
    }
    this.device.queue.writeBuffer(this.indexBuffer, 0, indices);

    // Compute pipeline
    const computeModule = this.device.createShaderModule({ code: waterComputeShader });
    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.computePipeline = this.device.createComputePipeline({
      label: "water-compute",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Render pipeline
    const renderModule = this.device.createShaderModule({ code: waterRenderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "water-render",
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
    });

    this.computeBindGroup = this.device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.computeUniformBuffer } },
        { binding: 1, resource: { buffer: this.vertexBuffer } },
        { binding: 2, resource: { buffer: this.heightBuffer } },
        { binding: 3, resource: { buffer: this.heightNextBuffer } },
      ],
    });

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: { buffer: this.vertexBuffer } },
        { binding: 2, resource: { buffer: this.indexBuffer } },
      ],
    });
  }

  update(time: number) {
    // Compute uniforms: time, gridSize, waveSpeed, damping
    this.computeUniformData[0] = time;
    this.computeUniformData[1] = GRID_SIZE;
    this.computeUniformData[2] = this.waveSpeed;
    this.computeUniformData[3] = 0.98;
    this.device.queue.writeBuffer(this.computeUniformBuffer, 0, this.computeUniformData as unknown as GPUAllowSharedBufferSource);

    // Render uniforms: viewProj(16) + cameraPosition(4) + time(1) + pads(3) = 24 floats = 96 bytes
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    this.renderUniformData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.renderUniformData[16] = this.camera.position[0];
    this.renderUniformData[17] = this.camera.position[1];
    this.renderUniformData[18] = this.camera.position[2];
    this.renderUniformData[19] = 1.0;
    this.renderUniformData[20] = time;
    this.renderUniformData[21] = 0;
    this.renderUniformData[22] = 0;
    this.renderUniformData[23] = 0;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderUniformData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        // Compute wave simulation
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(GRID_SIZE / 8), Math.ceil(GRID_SIZE / 8));
        computePass.end();

        // Render water
        const renderPass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0.05, g: 0.1, b: 0.2, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.draw(INDEX_COUNT);
        renderPass.end();
      },
    }];
  }

  stats() {
    return {
      drawCalls: 1,
      triangles: INDEX_COUNT / 3,
      computeDispatches: 1,
      custom: {
        "Grid": `${GRID_SIZE}x${GRID_SIZE}`,
        "Vertices": VERTEX_COUNT.toLocaleString(),
        "Waves": "5 sine interference",
        "Reflection": "Fresnel + Specular",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "waveSpeed", 0.1, 3, 0.1).name("Wave Speed");
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.heightBuffer.destroy();
    this.heightNextBuffer.destroy();
    this.computeUniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
  }
}
