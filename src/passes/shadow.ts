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
      compare: "less-equal",
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
    const up = vec3.create(0, 1, 0);
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
  bias: f32
) -> f32 {
  let lightPos = lightVP * vec4<f32>(worldPos, 1.0);
  let shadowCoord = clamp(lightPos.xy / lightPos.w * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
  let depth = lightPos.z / lightPos.w - bias;

  let texelSize = 1.0 / vec2<f32>(textureDimensions(shadowTexture));

  var shadow = 0.0;
  for (var y = -1; y <= 1; y++) {
    for (var x = -1; x <= 1; x++) {
      let offset = vec2<f32>(f32(x), f32(y)) * texelSize;
      shadow += textureSampleCompare(
        shadowTexture, shadowSampler,
        clamp(shadowCoord + offset, vec2<f32>(0.0), vec2<f32>(1.0)), depth
      );
    }
  }
  return shadow / 9.0;
}
`;