import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import { mat4 } from "wgpu-matrix";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";

const GRID_SIZE = 64;
const VERTEX_COUNT = GRID_SIZE * GRID_SIZE;
const INDEX_COUNT = (GRID_SIZE - 1) * (GRID_SIZE - 1) * 6;
// WGSL struct alignment: vec3 has 16-byte alignment, so Vertex struct stride = 48 bytes
const VERTEX_STRIDE = 48;

const meshGenShader = `
struct MeshUniforms {
  time: f32,
  gridSize: f32,
  amplitude: f32,
  frequency: f32,
};

struct Vertex {
  position: vec3<f32>,
  normal: vec3<f32>,
  uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: MeshUniforms;
@group(0) @binding(1) var<storage, read_write> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> drawArgs: array<u32>;

fn heightFn(x: f32, z: f32, t: f32) -> f32 {
  let freq = u.frequency;
  let amp = u.amplitude;
  return sin(x * freq + t) * cos(z * freq * 0.8 + t * 0.7) * amp
       + sin((x + z) * freq * 0.5 + t * 1.3) * amp * 0.5;
}

@compute @workgroup_size(8, 8, 1)
fn cs_genVertices(@builtin(global_invocation_id) gid: vec3<u32>) {
  let gs = u32(u.gridSize);
  if (gid.x >= gs || gid.y >= gs) {
    return;
  }

  let idx = gid.y * gs + gid.x;
  let nx = f32(gid.x) / f32(gs - 1u);
  let nz = f32(gid.y) / f32(gs - 1u);

  let worldX = (nx - 0.5) * 10.0;
  let worldZ = (nz - 0.5) * 10.0;
  let h = heightFn(worldX, worldZ, u.time);

  // Compute normal via finite differences
  let eps = 0.05;
  let hR = heightFn(worldX + eps, worldZ, u.time);
  let hL = heightFn(worldX - eps, worldZ, u.time);
  let hU = heightFn(worldX, worldZ + eps, u.time);
  let hD = heightFn(worldX, worldZ - eps, u.time);
  let normal = normalize(vec3<f32>(hL - hR, 2.0 * eps, hD - hU));

  var v: Vertex;
  v.position = vec3<f32>(worldX, h, worldZ);
  v.normal = normal;
  v.uv = vec2<f32>(nx, nz);
  vertices[idx] = v;
}

@compute @workgroup_size(1)
fn cs_genIndices() {
  let gs = u32(u.gridSize);
  var idx = 0u;
  for (var y = 0u; y < gs - 1u; y++) {
    for (var x = 0u; x < gs - 1u; x++) {
      let a = y * gs + x;
      let b = a + 1u;
      let c = a + gs;
      let d = c + 1u;
      indices[idx] = a; idx++;
      indices[idx] = c; idx++;
      indices[idx] = b; idx++;
      indices[idx] = b; idx++;
      indices[idx] = c; idx++;
      indices[idx] = d; idx++;
    }
  }
  // Write indirect draw args: [indexCount, instanceCount, firstIndex, baseVertex, firstInstance]
  drawArgs[0] = idx;
  drawArgs[1] = 1u;
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
  drawArgs[4] = 0u;
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

struct Vertex {
  position: vec3<f32>,
  normal: vec3<f32>,
  uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) height: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  let v = vertices[vi];
  var out: VSOut;
  out.position = u.viewProj * vec4<f32>(v.position, 1.0);
  out.normal = v.normal;
  out.uv = v.uv;
  out.height = v.position.y;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
  let N = normalize(in.normal);
  let diffuse = max(dot(N, lightDir), 0.0);

  // Height-based color gradient
  let t = clamp(in.height * 0.5 + 0.5, 0.0, 1.0);
  let lowColor = vec3<f32>(0.1, 0.3, 0.6);
  let midColor = vec3<f32>(0.2, 0.7, 0.4);
  let highColor = vec3<f32>(0.9, 0.85, 0.7);
  var baseColor: vec3<f32>;
  if (t < 0.5) {
    baseColor = mix(lowColor, midColor, t * 2.0);
  } else {
    baseColor = mix(midColor, highColor, (t - 0.5) * 2.0);
  }

  // Wireframe-like grid overlay
  let grid = fract(in.uv * 32.0);
  let line = min(min(grid.x, 1.0 - grid.x), min(grid.y, 1.0 - grid.y));
  let wire = smoothstep(0.0, 0.05, line);
  baseColor = mix(baseColor * 1.5, baseColor, wire);

  let ambient = 0.15;
  let color = baseColor * (ambient + diffuse * 0.85);
  return vec4<f32>(pow(color, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;

export class MeshGenDemo implements Demo {
  label = "MeshGen (CS→Indirect)";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private engine: EngineContext | null = null;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private genVerticesPipeline!: GPUComputePipeline;
  private genIndicesPipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;

  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private drawArgsBuffer!: GPUBuffer;
  private meshUBO!: GPUBuffer;
  private renderUBO!: GPUBuffer;

  private genBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;

  private meshData = new Float32Array(4);
  private renderData = new Float32Array(20);
  private indicesGenerated = false;

  private computeCode = meshGenShader;
  private renderCode = renderShader;

  amplitude = 1.0;
  frequency = 1.5;

  init(ctx: GPUContext, camera: Camera, engine?: EngineContext) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;
    this.engine = engine ?? null;

    // Vertex buffer: position(3) + normal(3) + uv(2) = 8 floats per vertex
    this.vertexBuffer = this.device.createBuffer({
      label: "meshgen-vb",
      size: VERTEX_COUNT * VERTEX_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.indexBuffer = this.device.createBuffer({
      label: "meshgen-ib",
      size: INDEX_COUNT * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
    });

    // Indirect draw args: 5 x u32 = 20 bytes
    this.drawArgsBuffer = this.device.createBuffer({
      label: "meshgen-draw-args",
      size: 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    this.meshUBO = this.device.createBuffer({
      label: "meshgen-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUBO = this.device.createBuffer({
      label: "meshgen-render-ubo",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const genModule = this.device.createShaderModule({ code: meshGenShader });

    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const computePipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [computeBGL],
    });

    this.genVerticesPipeline = this.device.createComputePipeline({
      label: "meshgen-vertices",
      layout: computePipelineLayout,
      compute: { module: genModule, entryPoint: "cs_genVertices" },
    });
    this.genIndicesPipeline = this.device.createComputePipeline({
      label: "meshgen-indices",
      layout: computePipelineLayout,
      compute: { module: genModule, entryPoint: "cs_genIndices" },
    });

    this.genBindGroup = this.device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.meshUBO } },
        { binding: 1, resource: { buffer: this.vertexBuffer } },
        { binding: 2, resource: { buffer: this.indexBuffer } },
        { binding: 3, resource: { buffer: this.drawArgsBuffer } },
      ],
    });

    const renderModule = this.device.createShaderModule({ code: renderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "meshgen-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
    });

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUBO } },
        { binding: 1, resource: { buffer: this.vertexBuffer } },
      ],
    });
  }

  update(time: number) {
    this.meshData[0] = time;
    this.meshData[1] = GRID_SIZE;
    this.meshData[2] = this.amplitude;
    this.meshData[3] = this.frequency;
    this.device.queue.writeBuffer(this.meshUBO, 0, this.meshData as unknown as GPUAllowSharedBufferSource);

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    this.renderData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.renderData[16] = time;
    this.device.queue.writeBuffer(this.renderUBO, 0, this.renderData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        // Generate indices + draw args once (topology is static)
        if (!this.indicesGenerated) {
          const idxPass = encoder.beginComputePass();
          idxPass.setPipeline(this.genIndicesPipeline);
          idxPass.setBindGroup(0, this.genBindGroup);
          idxPass.dispatchWorkgroups(1);
          idxPass.end();
          this.indicesGenerated = true;
        }

        // Generate vertices every frame (animated height field)
        const vertPass = encoder.beginComputePass();
        vertPass.setPipeline(this.genVerticesPipeline);
        vertPass.setBindGroup(0, this.genBindGroup);
        vertPass.dispatchWorkgroups(Math.ceil(GRID_SIZE / 8), Math.ceil(GRID_SIZE / 8));
        vertPass.end();

        // Render via indirect draw (GPU decides index count)
        const renderPass = encoder.beginRenderPass({
          colorAttachments: [{
            view,
            clearValue: { r: 0.02, g: 0.02, b: 0.05, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        renderPass.setPipeline(this.renderPipeline);
        renderPass.setBindGroup(0, this.renderBindGroup);
        renderPass.setIndexBuffer(this.indexBuffer, "uint32");
        renderPass.drawIndexedIndirect(this.drawArgsBuffer, 0);
        renderPass.end();
      },
    }];
  }

  stats() {
    return {
      drawCalls: 1,
      triangles: INDEX_COUNT / 3,
      computeDispatches: 2,
      custom: {
        "Grid": `${GRID_SIZE}x${GRID_SIZE}`,
        "Vertices": VERTEX_COUNT.toLocaleString(),
        "Draw Mode": "Indirect (GPU args)",
        "Mesh Gen": "Compute Shader",
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "amplitude", 0.1, 3.0, 0.1).name("Amplitude");
    gui.add(this, "frequency", 0.5, 5.0, 0.1).name("Frequency");
  }

  getShaderStages(): ShaderStageDesc[] {
    return [
      { label: "MeshGen / Compute", type: "compute", code: this.computeCode },
      { label: "MeshGen / Render", type: "fragment", code: this.renderCode },
    ];
  }

  onShaderReload(stageLabel: string, code: string): boolean {
    if (stageLabel === "MeshGen / Compute") this.computeCode = code;
    else if (stageLabel === "MeshGen / Render") this.renderCode = code;
    return this.rebuildPipelines();
  }

  private rebuildPipelines(): boolean {
    try {
      const compile = (label: string, code: string) =>
        this.engine
          ? this.engine.modules.resolveAndCompile(this.device, label, code)
          : this.device.createShaderModule({ label, code });

      const genModule = compile("meshgen-compute", this.computeCode);
      this.genVerticesPipeline = this.device.createComputePipeline({
        label: "meshgen-vertices",
        layout: "auto",
        compute: { module: genModule, entryPoint: "cs_genVertices" },
      });
      this.genIndicesPipeline = this.device.createComputePipeline({
        label: "meshgen-indices",
        layout: "auto",
        compute: { module: genModule, entryPoint: "cs_genIndices" },
      });

      this.genBindGroup = this.device.createBindGroup({
        layout: this.genVerticesPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.meshUBO } },
          { binding: 1, resource: { buffer: this.vertexBuffer } },
          { binding: 2, resource: { buffer: this.indexBuffer } },
          { binding: 3, resource: { buffer: this.drawArgsBuffer } },
        ],
      });

      const renderModule = compile("meshgen-render", this.renderCode);
      this.renderPipeline = this.device.createRenderPipeline({
        label: "meshgen-render",
        layout: "auto",
        vertex: { module: renderModule, entryPoint: "vs_main" },
        fragment: {
          module: renderModule,
          entryPoint: "fs_main",
          targets: [{ format: this.format }],
        },
        primitive: { topology: "triangle-list", cullMode: "none" },
      });

      this.renderBindGroup = this.device.createBindGroup({
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderUBO } },
          { binding: 1, resource: { buffer: this.vertexBuffer } },
        ],
      });

      this.indicesGenerated = false;
      return true;
    } catch (e) {
      console.error("[MeshGen] Pipeline rebuild failed:", e);
      return false;
    }
  }

  destroy() {
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.drawArgsBuffer.destroy();
    this.meshUBO.destroy();
    this.renderUBO.destroy();
  }
}
