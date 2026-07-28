import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";
import { mat4 } from "wgpu-matrix";

const INSTANCE_COUNT = 10000;

const cullShader = `
struct InstanceData {
  model: mat4x4<f32>,
  color: vec4<f32>,
};

struct DrawIndirectArgs {
  indexCount: u32,
  instanceCount: u32,
  firstIndex: u32,
  baseVertex: i32,
  firstInstance: u32,
};

struct CullUniforms {
  viewProj: mat4x4<f32>,
  frustumPlanes: array<vec4<f32>, 6>,
  time: f32,
  instanceCount: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> cu: CullUniforms;
@group(0) @binding(1) var<storage, read> instancesIn: array<InstanceData>;
@group(0) @binding(2) var<storage, read_write> instancesOut: array<InstanceData>;
@group(0) @binding(3) var<storage, read_write> drawArgs: DrawIndirectArgs;
@group(0) @binding(4) var<storage, read_write> visibleCount: atomic<u32>;

fn sphereInFrustum(center: vec3<f32>, radius: f32) -> bool {
  for (var i = 0; i < 6; i++) {
    let plane = cu.frustumPlanes[i];
    let dist = dot(plane.xyz, center) + plane.w;
    if (dist < -radius) {
      return false;
    }
  }
  return true;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= u32(cu.instanceCount)) {
    return;
  }

  let inst = instancesIn[index];
  let center = vec3<f32>(inst.model[3][0], inst.model[3][1], inst.model[3][2]);
  let scale = length(vec3<f32>(inst.model[0][0], inst.model[0][1], inst.model[0][2]));

  if (sphereInFrustum(center, scale * 1.5)) {
    let slot = atomicAdd(&visibleCount, 1u);
    instancesOut[slot] = inst;
  }
}

@compute @workgroup_size(1)
fn cs_finalize() {
  let count = atomicLoad(&visibleCount);
  drawArgs.instanceCount = count;
}
`;

const renderShader = `
struct InstanceData {
  model: mat4x4<f32>,
  color: vec4<f32>,
};

struct RenderUniforms {
  viewProj: mat4x4<f32>,
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> ru: RenderUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @builtin(instance_index) instanceIndex: u32,
) -> VSOut {
  let inst = instances[instanceIndex];
  let worldPos = inst.model * vec4<f32>(pos, 1.0);

  var out: VSOut;
  out.position = ru.viewProj * worldPos;
  out.color = inst.color.rgb;
  out.normal = normalize((inst.model * vec4<f32>(normal, 0.0)).xyz);
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let lightDir = normalize(vec3<f32>(0.5, 1.0, 0.3));
  let diffuse = max(dot(normalize(in.normal), lightDir), 0.0);
  let color = in.color * (0.15 + diffuse * 0.85);
  return vec4<f32>(color, 1.0);
}
`;

export class IndirectDrawDemo implements Demo {
  label = "IndirectDraw";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private cullPipeline!: GPUComputePipeline;
  private finalizePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private instanceBuffers: GPUBuffer[] = [];
  private drawArgsBuffer!: GPUBuffer;
  private visibleCountBuffer!: GPUBuffer;
  private cullUniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private indexCount = 0;
  private depthTexture: GPUTexture | null = null;
  private current = 0;

  init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;

    const { vertices, indices } = this.createSphere();
    this.indexCount = indices.length;

    this.vertexBuffer = this.device.createBuffer({
      label: "indirect-vb",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: "indirect-ib",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();

    const instanceFloats = 20;
    const instanceData = new Float32Array(INSTANCE_COUNT * instanceFloats);
    for (let i = 0; i < INSTANCE_COUNT; i++) {
      const base = i * instanceFloats;
      const x = (Math.random() * 2 - 1) * 40;
      const y = (Math.random() * 2 - 1) * 40;
      const z = Math.random() * 20 - 5;
      const s = 0.2 + Math.random() * 0.6;

      instanceData[base + 0] = s; instanceData[base + 1] = 0; instanceData[base + 2] = 0; instanceData[base + 3] = 0;
      instanceData[base + 4] = 0; instanceData[base + 5] = s; instanceData[base + 6] = 0; instanceData[base + 7] = 0;
      instanceData[base + 8] = 0; instanceData[base + 9] = 0; instanceData[base + 10] = s; instanceData[base + 11] = 0;
      instanceData[base + 12] = x; instanceData[base + 13] = y; instanceData[base + 14] = z; instanceData[base + 15] = 1;

      const hue = Math.random();
      const rgb = this.hsvToRgb(hue, 0.7, 0.9);
      instanceData[base + 16] = rgb[0];
      instanceData[base + 17] = rgb[1];
      instanceData[base + 18] = rgb[2];
      instanceData[base + 19] = 1;
    }

    for (let i = 0; i < 2; i++) {
      this.instanceBuffers.push(
        this.device.createBuffer({
          label: `indirect-instances-${i}`,
          size: INSTANCE_COUNT * instanceFloats * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
        })
      );
      new Float32Array(this.instanceBuffers[i].getMappedRange()).set(instanceData);
      this.instanceBuffers[i].unmap();
    }

    // DrawIndirectArgs: indexCount, instanceCount, firstIndex, baseVertex, firstInstance
    const drawArgsData = new Uint32Array([this.indexCount, 0, 0, 0, 0]);
    this.drawArgsBuffer = this.device.createBuffer({
      label: "indirect-draw-args",
      size: 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(this.drawArgsBuffer.getMappedRange()).set(drawArgsData);
    this.drawArgsBuffer.unmap();

    this.visibleCountBuffer = this.device.createBuffer({
      label: "indirect-visible-count",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.cullUniformBuffer = this.device.createBuffer({
      label: "indirect-cull-ubo",
      size: 176,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUniformBuffer = this.device.createBuffer({
      label: "indirect-render-ubo",
      size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const cullModule = this.device.createShaderModule({ code: cullShader });
    const computeBindGroupLayout = this.device.createBindGroupLayout({
      label: "indirect-cull-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const computePipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [computeBindGroupLayout],
    });
    this.cullPipeline = this.device.createComputePipeline({
      label: "indirect-cull",
      layout: computePipelineLayout,
      compute: { module: cullModule, entryPoint: "cs_main" },
    });
    this.finalizePipeline = this.device.createComputePipeline({
      label: "indirect-finalize",
      layout: computePipelineLayout,
      compute: { module: cullModule, entryPoint: "cs_finalize" },
    });

    const vertexLayout: GPUVertexBufferLayout = {
      arrayStride: 8 * 4,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ],
    };

    const renderModule = this.device.createShaderModule({ code: renderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "indirect-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{ format: ctx.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
  }

  private cullUboData = new Float32Array(44);
  private renderUboData = new Float32Array(20);
  private zeroU32 = new Uint32Array(1);
  private cullBindGroups: GPUBindGroup[] = [];
  private renderBindGroups: GPUBindGroup[] = [];
  private cachedDepthView: GPUTextureView | null = null;

  private ensureDepth() {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (this.depthTexture && this.depthTexture.width === w && this.depthTexture.height === h) return;
    this.depthTexture?.destroy();
    this.depthTexture = this.device.createTexture({
      size: [w, h],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.cachedDepthView = this.depthTexture.createView();
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);

    const planes = this.extractFrustumPlanes(viewProj as unknown as number[]);
    const cullUbo = this.cullUboData;
    cullUbo.set(viewProj as unknown as ArrayLike<number>, 0);
    for (let i = 0; i < 6; i++) {
      cullUbo[16 + i * 4 + 0] = planes[i][0];
      cullUbo[16 + i * 4 + 1] = planes[i][1];
      cullUbo[16 + i * 4 + 2] = planes[i][2];
      cullUbo[16 + i * 4 + 3] = planes[i][3];
    }
    cullUbo[40] = time;
    cullUbo[41] = INSTANCE_COUNT;
    this.device.queue.writeBuffer(this.cullUniformBuffer, 0, cullUbo as unknown as GPUAllowSharedBufferSource);

    const renderUbo = this.renderUboData;
    renderUbo.set(viewProj as unknown as ArrayLike<number>, 0);
    renderUbo[16] = time;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, renderUbo as unknown as GPUAllowSharedBufferSource);

    this.zeroU32[0] = 0;
    this.device.queue.writeBuffer(this.visibleCountBuffer, 0, this.zeroU32 as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.drawArgsBuffer, 4, this.zeroU32 as unknown as GPUAllowSharedBufferSource);
  }

  private ensureBindGroups() {
    if (this.cullBindGroups.length === 2) return;
    const cullLayout = this.cullPipeline.getBindGroupLayout(0);
    const renderLayout = this.renderPipeline.getBindGroupLayout(0);
    for (let i = 0; i < 2; i++) {
      const src = i;
      const dst = 1 - i;
      this.cullBindGroups[i] = this.device.createBindGroup({
        layout: cullLayout,
        entries: [
          { binding: 0, resource: { buffer: this.cullUniformBuffer } },
          { binding: 1, resource: { buffer: this.instanceBuffers[src] } },
          { binding: 2, resource: { buffer: this.instanceBuffers[dst] } },
          { binding: 3, resource: { buffer: this.drawArgsBuffer } },
          { binding: 4, resource: { buffer: this.visibleCountBuffer } },
        ],
      });
      this.renderBindGroups[i] = this.device.createBindGroup({
        layout: renderLayout,
        entries: [
          { binding: 0, resource: { buffer: this.renderUniformBuffer } },
          { binding: 1, resource: { buffer: this.instanceBuffers[dst] } },
        ],
      });
    }
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    this.ensureDepth();
    this.ensureBindGroups();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.cullPipeline);
    computePass.setBindGroup(0, this.cullBindGroups[this.current]);
    computePass.dispatchWorkgroups(Math.ceil(INSTANCE_COUNT / 256));
    computePass.setPipeline(this.finalizePipeline);
    computePass.setBindGroup(0, this.cullBindGroups[this.current]);
    computePass.dispatchWorkgroups(1);
    computePass.end();

    this.current = 1 - this.current;

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.cachedDepthView!,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroups[this.current]);
    renderPass.setVertexBuffer(0, this.vertexBuffer);
    renderPass.setIndexBuffer(this.indexBuffer, "uint16");
    renderPass.drawIndexedIndirect(this.drawArgsBuffer, 0);
    renderPass.end();
  }

  private extractFrustumPlanes(vp: number[]): number[][] {
    const m = (r: number, c: number) => vp[c * 4 + r];
    const planes: number[][] = [];
    // Left
    planes.push(this.normalizePlane([m(3,0)+m(0,0), m(3,1)+m(0,1), m(3,2)+m(0,2), m(3,3)+m(0,3)]));
    // Right
    planes.push(this.normalizePlane([m(3,0)-m(0,0), m(3,1)-m(0,1), m(3,2)-m(0,2), m(3,3)-m(0,3)]));
    // Bottom
    planes.push(this.normalizePlane([m(3,0)+m(1,0), m(3,1)+m(1,1), m(3,2)+m(1,2), m(3,3)+m(1,3)]));
    // Top
    planes.push(this.normalizePlane([m(3,0)-m(1,0), m(3,1)-m(1,1), m(3,2)-m(1,2), m(3,3)-m(1,3)]));
    // Near
    planes.push(this.normalizePlane([m(3,0)+m(2,0), m(3,1)+m(2,1), m(3,2)+m(2,2), m(3,3)+m(2,3)]));
    // Far
    planes.push(this.normalizePlane([m(3,0)-m(2,0), m(3,1)-m(2,1), m(3,2)-m(2,2), m(3,3)-m(2,3)]));
    return planes;
  }

  private normalizePlane(p: number[]): number[] {
    const len = Math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    if (len < 1e-8) return p;
    return [p[0]/len, p[1]/len, p[2]/len, p[3]/len];
  }

  private createSphere(): { vertices: Float32Array; indices: Uint16Array } {
    const segments = 12;
    const rings = 8;
    const verts: number[] = [];
    const inds: number[] = [];

    for (let y = 0; y <= rings; y++) {
      const phi = (y / rings) * Math.PI;
      for (let x = 0; x <= segments; x++) {
        const theta = (x / segments) * Math.PI * 2;
        const nx = Math.sin(phi) * Math.cos(theta);
        const ny = Math.cos(phi);
        const nz = Math.sin(phi) * Math.sin(theta);
        verts.push(nx, ny, nz, nx, ny, nz, x / segments, y / rings);
      }
    }

    for (let y = 0; y < rings; y++) {
      for (let x = 0; x < segments; x++) {
        const a = y * (segments + 1) + x;
        const b = a + segments + 1;
        inds.push(a, b, a + 1, a + 1, b, b + 1);
      }
    }

    return { vertices: new Float32Array(verts), indices: new Uint16Array(inds) };
  }

  private hsvToRgb(h: number, s: number, v: number): [number, number, number] {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, p, v];
      default: return [v, p, q];
    }
  }

  stats() {
    return {
      drawCalls: 1,
      instances: INSTANCE_COUNT,
      triangles: (this.indexCount / 3) * INSTANCE_COUNT,
      computeDispatches: 2,
      custom: { "Culling": "GPU Frustum", "Draw Mode": "Indirect" },
    };
  }

  destroy() {
    for (const b of this.instanceBuffers) b.destroy();
    this.drawArgsBuffer.destroy();
    this.visibleCountBuffer.destroy();
    this.cullUniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
    this.vertexBuffer.destroy();
    this.indexBuffer.destroy();
    this.depthTexture?.destroy();
  }
}
