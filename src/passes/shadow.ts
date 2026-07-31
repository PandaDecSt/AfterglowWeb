import { mat4, vec3, type Mat4, type Vec3 } from "wgpu-matrix";

export class ShadowMap {
  private device: GPUDevice;
  private shadowTexture!: GPUTexture;
  private shadowView!: GPUTextureView;
  private shadowSampler!: GPUSampler;
  private shadowBindGroupLayout!: GPUBindGroupLayout;

  size: number;
  format: GPUTextureFormat = "depth32float";
  lightVP: Mat4 = mat4.identity(mat4.create());
  lightPosition: Vec3 = vec3.create(2, 5, 3);
  lightTarget: Vec3 = vec3.create(0, 0, 0);
  near = 0.1;
  far = 50.0;
  orthoSize = 20.0;

  private shadowPipeline: GPURenderPipeline | null = null;
  private uniformBuffer!: GPUBuffer;
  private uniformData = new Float32Array(16);

  constructor(device: GPUDevice, size = 2048) {
    this.device = device;
    this.size = size;
    this.createResources();
  }

  private createResources(): void {
    this.shadowTexture = this.device.createTexture({
      label: "shadow-map",
      size: [this.size, this.size],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this.shadowTexture.createView();

    this.shadowSampler = this.device.createSampler({
      label: "shadow-comparison",
      compare: "less",
      magFilter: "linear",
      minFilter: "linear",
    });

    this.shadowBindGroupLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
      ],
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "shadow-vp-ubo",
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  get texture(): GPUTexture {
    return this.shadowTexture;
  }

  get view(): GPUTextureView {
    return this.shadowView;
  }

  get sampler(): GPUSampler {
    return this.shadowSampler;
  }

  get bindGroupLayout(): GPUBindGroupLayout {
    return this.shadowBindGroupLayout;
  }

  createBindGroup(layout: GPUBindGroupLayout, binding0: number, binding1: number): GPUBindGroup {
    return this.device.createBindGroup({
      layout,
      entries: [
        { binding: binding0, resource: this.shadowView },
        { binding: binding1, resource: this.shadowSampler },
      ],
    });
  }

  updateLightVP(): void {
    const dir = vec3.subtract(this.lightPosition, this.lightTarget);
    const len = vec3.length(dir) || 1;
    const nd = vec3.divide(dir, vec3.create(len, len, len));
    const up = Math.abs(nd[1]) > 0.99 ? vec3.create(0, 0, -1) : vec3.create(0, 1, 0);
    const view = mat4.lookAt(this.lightPosition, this.lightTarget, up);
    const half = this.orthoSize / 2;
    const proj = mat4.ortho(-half, half, -half, half, this.near, this.far);

    const remap = mat4.identity(mat4.create());
    remap[10] = 0.5;
    remap[14] = 0.5;

    this.lightVP = mat4.multiply(remap, mat4.multiply(proj, view));

    this.uniformData.set(this.lightVP as unknown as ArrayLike<number>, 0);
    this.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      this.uniformData as unknown as GPUAllowSharedBufferSource
    );
  }

  getVPBuffer(): GPUBuffer {
    return this.uniformBuffer;
  }

  beginShadowPass(encoder: GPUCommandEncoder): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.shadowView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
  }

  resize(size: number): void {
    if (size === this.size) return;
    this.shadowTexture.destroy();
    this.size = size;
    this.shadowTexture = this.device.createTexture({
      label: "shadow-map",
      size: [this.size, this.size],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.shadowView = this.shadowTexture.createView();
  }

  destroy(): void {
    this.shadowTexture.destroy();
    this.uniformBuffer.destroy();
  }
}

export const SHADOW_WGSL = `
fn sampleShadowPCF(
  shadowTexture: texture_depth_2d,
  shadowSampler: sampler_comparison,
  lightVP: mat4x4<f32>,
  worldPos: vec3<f32>,
  normal: vec3<f32>,
  lightDir: vec3<f32>
) -> f32 {
  if (dot(normal, lightDir) <= 0.0) { return 1.0; }
  let biasedPos = worldPos + normal * 0.08;
  let lclip = lightVP * vec4<f32>(biasedPos, 1.0);
  let ndc = lclip.xyz / max(lclip.w, 1e-6);
  let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let cmpZ = ndc.z - 0.001;
  let ts = 1.0 / f32(textureDimensions(shadowTexture).x);
  let s00 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv, cmpZ);
  let s21 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(shadowTexture, shadowSampler, suv + vec2f( ts,  ts), cmpZ);
  var vis = (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);
  let inZ = select(0.0, 1.0, ndc.z > 0.0 && ndc.z < 1.0);
  let frustum = (1.0 - smoothstep(0.88, 0.96, abs(ndc.x)))
              * (1.0 - smoothstep(0.88, 0.96, abs(ndc.y))) * inZ;
  return mix(1.0, vis, frustum);
}
`;