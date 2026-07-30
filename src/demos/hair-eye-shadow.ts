import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import type { Demo } from "./types";
import { mat4 } from "wgpu-matrix";
import type { RenderPass } from "../core/renderer";

const hairEyeShader = `
struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  time: f32,
  materialID: f32,
  shadowIntensity: f32,
  rimWidth: f32,
  rimIntensity: f32,
  specularPower: f32,
  pad0: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
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
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  return out;
}

fn KajiyaKaySpecular(tangent: vec3<f32>, normal: vec3<f32>, V: vec3<f32>, L: vec3<f32>, power: f32) -> f32 {
  let T = normalize(tangent);
  let TdotV = dot(T, V);
  let TdotL = dot(T, L);
  return pow(sqrt(1.0 - TdotV * TdotV) * sqrt(1.0 - TdotL * TdotL) - TdotV * TdotL, power);
}

fn HairDiffuse(N: vec3<f32>, L: vec3<f32>) -> f32 {
  return max(dot(N, L) * 0.5 + 0.5, 0.0);
}

fn RimLight(N: vec3<f32>, V: vec3<f32>, width: f32) -> f32 {
  let rim = 1.0 - max(dot(N, V), 0.0);
  return smoothstep(1.0 - width, 1.0, rim);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);
  let L = normalize(u.lightDir.xyz);
  let H = normalize(V + L);

  let matID = i32(u.materialID);
  var color: vec3<f32>;

  if (matID == 0) {
    let tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), N));
    let baseColor = vec3<f32>(0.35, 0.25, 0.45);
    let diffuse = HairDiffuse(N, L) * 0.7;
    let spec = KajiyaKaySpecular(tangent, N, V, L, u.specularPower);
    let rim = RimLight(N, V, u.rimWidth) * u.rimIntensity;
    let strands = sin(in.uv.x * 80.0 + in.uv.y * 20.0) * 0.5 + 0.5;
    color = baseColor * diffuse;
    color += vec3<f32>(0.6, 0.5, 0.8) * spec * 0.5;
    color += baseColor * rim;
    color *= 0.9 + strands * 0.1;
  } else {
    let baseColor = vec3<f32>(0.2, 0.7, 0.9);
    let spec = pow(max(dot(N, H), 0.0), 64.0);
    let shadow = 1.0 - smoothstep(0.7, 1.0, in.uv.y) * 0.4;
    color = baseColor * shadow;
    color += vec3<f32>(1.0) * spec * 0.9;
    let reflPos = vec2<f32>(0.38, 0.38);
    let refl = smoothstep(0.06, 0.03, distance(in.uv, reflPos));
    color += refl * 0.8;
  }

  color *= u.lightDir.w;
  return vec4<f32>(color, 1.0);
}
`;

export class HairEyeShadowDemo implements Demo {
  label = "Hair/Eye Shadow";
  private ctx!: GPUContext;
  private camera!: Camera;
  private device!: GPUDevice;
  private format!: GPUTextureFormat;

  private pipeline!: GPURenderPipeline;
  private hairUBO!: GPUBuffer;
  private eyeUBO!: GPUBuffer;
  private hairBindGroup!: GPUBindGroup;
  private eyeBindGroup!: GPUBindGroup;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private depthTexture: GPUTexture | null = null;
  private cachedDepthView: GPUTextureView | null = null;

  private specularPower = 32.0;

  async init(ctx: GPUContext, camera: Camera) {
    this.ctx = ctx;
    this.camera = camera;
    this.device = ctx.device;
    this.format = ctx.format;

    const hairGeo = this.createCylinder(0.3, 1.5, 16, 16);
    const eyeGeo = this.createSphere(0.4, 16, 16);

    const allVerts: number[] = [];
    const allInds: number[] = [];

    for (let i = 0; i < hairGeo.vertices.length; i += 8) {
      allVerts.push(hairGeo.vertices[i] - 1.0, hairGeo.vertices[i + 1], hairGeo.vertices[i + 2]);
      allVerts.push(hairGeo.vertices[i + 3], hairGeo.vertices[i + 4], hairGeo.vertices[i + 5]);
      allVerts.push(hairGeo.vertices[i + 6], hairGeo.vertices[i + 7]);
    }
    for (const idx of hairGeo.indices) { allInds.push(idx); }

    const hairVertCount = allVerts.length / 8;
    for (let i = 0; i < eyeGeo.vertices.length; i += 8) {
      allVerts.push(eyeGeo.vertices[i] + 1.0, eyeGeo.vertices[i + 1], eyeGeo.vertices[i + 2]);
      allVerts.push(eyeGeo.vertices[i + 3], eyeGeo.vertices[i + 4], eyeGeo.vertices[i + 5]);
      allVerts.push(eyeGeo.vertices[i + 6], eyeGeo.vertices[i + 7]);
    }
    for (const idx of eyeGeo.indices) { allInds.push(idx + hairVertCount); }

    this.vertexBuffer = this.device.createBuffer({
      size: allVerts.length * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, new Float32Array(allVerts));

    this.indexBuffer = this.device.createBuffer({
      size: allInds.length * 2,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indexBuffer, 0, new Uint16Array(allInds));

    // Two separate uniform buffers
    this.hairUBO = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.eyeUBO = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const bgLayout = this.device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }],
    });

    this.hairBindGroup = this.device.createBindGroup({
      layout: bgLayout,
      entries: [{ binding: 0, resource: { buffer: this.hairUBO } }],
    });

    this.eyeBindGroup = this.device.createBindGroup({
      layout: bgLayout,
      entries: [{ binding: 0, resource: { buffer: this.eyeUBO } }],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }),
      vertex: {
        module: this.device.createShaderModule({ code: hairEyeShader }),
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ],
        }],
      },
      fragment: {
        module: this.device.createShaderModule({ code: hairEyeShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
  }

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

  private createSphere(radius: number, wSeg: number, hSeg: number) {
    const verts: number[] = [];
    const inds: number[] = [];
    for (let y = 0; y <= hSeg; y++) {
      for (let x = 0; x <= wSeg; x++) {
        const u = x / wSeg;
        const v = y / hSeg;
        const theta = u * Math.PI * 2;
        const phi = v * Math.PI;
        const px = radius * Math.sin(phi) * Math.cos(theta);
        const py = radius * Math.cos(phi);
        const pz = radius * Math.sin(phi) * Math.sin(theta);
        verts.push(px, py, pz, px, py, pz, u, v);
      }
    }
    for (let y = 0; y < hSeg; y++) {
      for (let x = 0; x < wSeg; x++) {
        const a = y * (wSeg + 1) + x;
        const b = a + wSeg + 1;
        inds.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    return { vertices: new Float32Array(verts), indices: new Uint16Array(inds) };
  }

  private createCylinder(radius: number, height: number, seg: number, hSeg: number) {
    const verts: number[] = [];
    const inds: number[] = [];
    for (let y = 0; y <= hSeg; y++) {
      for (let x = 0; x <= seg; x++) {
        const u = x / seg;
        const v = y / hSeg;
        const theta = u * Math.PI * 2;
        const py = -height / 2 + v * height;
        const px = radius * Math.cos(theta);
        const pz = radius * Math.sin(theta);
        verts.push(px, py, pz, Math.cos(theta), 0, Math.sin(theta), u, v);
      }
    }
    for (let y = 0; y < hSeg; y++) {
      for (let x = 0; x < seg; x++) {
        const a = y * (seg + 1) + x;
        const b = a + seg + 1;
        inds.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    return { vertices: new Float32Array(verts), indices: new Uint16Array(inds) };
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    const aspect = w / h;
    const viewProj = this.camera.getViewProjectionMatrix(aspect);

    const model = mat4.mul(mat4.rotationY(time * 0.2), mat4.scaling([1.5, 1.5, 1.5]));
    const invTransModel = mat4.transpose(mat4.inverse(model));

    // Hair UBO
    const hairUbo = new Float32Array(64);
    hairUbo.set(viewProj as unknown as ArrayLike<number>, 0);
    hairUbo.set(model as unknown as ArrayLike<number>, 16);
    hairUbo.set(invTransModel as unknown as ArrayLike<number>, 32);
    hairUbo[48] = this.camera.position[0];
    hairUbo[49] = this.camera.position[1];
    hairUbo[50] = this.camera.position[2];
    hairUbo[51] = 1.0;
    hairUbo[52] = -0.4; hairUbo[53] = -1.0; hairUbo[54] = -0.3; hairUbo[55] = 1.0;
    hairUbo[56] = time;
    hairUbo[57] = 0.0; // materialID: hair
    hairUbo[58] = 0.5; // shadowIntensity
    hairUbo[59] = 0.3; // rimWidth
    hairUbo[60] = 0.5; // rimIntensity
    hairUbo[61] = this.specularPower;
    this.device.queue.writeBuffer(this.hairUBO, 0, hairUbo as unknown as GPUAllowSharedBufferSource);

    // Eye UBO
    const eyeUbo = new Float32Array(64);
    eyeUbo.set(viewProj as unknown as ArrayLike<number>, 0);
    eyeUbo.set(model as unknown as ArrayLike<number>, 16);
    eyeUbo.set(invTransModel as unknown as ArrayLike<number>, 32);
    eyeUbo[48] = this.camera.position[0];
    eyeUbo[49] = this.camera.position[1];
    eyeUbo[50] = this.camera.position[2];
    eyeUbo[51] = 1.0;
    eyeUbo[52] = -0.4; eyeUbo[53] = -1.0; eyeUbo[54] = -0.3; eyeUbo[55] = 1.0;
    eyeUbo[56] = time;
    eyeUbo[57] = 1.0; // materialID: eye
    eyeUbo[58] = 0.5;
    eyeUbo[59] = 0.3;
    eyeUbo[60] = 0.5;
    eyeUbo[61] = 64.0;
    this.device.queue.writeBuffer(this.eyeUBO, 0, eyeUbo as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        this.ensureDepth();

        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0.15, g: 0.15, b: 0.2, a: 1 } }],
          depthStencilAttachment: { view: this.cachedDepthView!, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1.0 },
        });

        pass.setPipeline(this.pipeline);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setIndexBuffer(this.indexBuffer, "uint16");

        // Draw hair
        pass.setBindGroup(0, this.hairBindGroup);
        pass.drawIndexed(16 * 16 * 6, 1, 0);

        // Draw eye
        const hairVertCount = 17 * 17;
        pass.setBindGroup(0, this.eyeBindGroup);
        pass.drawIndexed(16 * 16 * 6, 1, hairVertCount);

        pass.end();
      },
    }];
  }

  destroy() {
    this.depthTexture?.destroy();
  }

  registerGUI(gui: any) {
    const folder = gui.addFolder("Hair/Eye Shadow");
    folder.add(this, "specularPower", 8, 128).name("Specular Power");
  }
}
