import { RenderTarget, DepthTarget } from "./render-target";

export type ToneMappingMode = "off" | "aces" | "filmic" | "reinhard";

export class HDRRenderTarget {
  private device: GPUDevice;
  private width = 0;
  private height = 0;

  colorTarget!: RenderTarget;
  depthTarget!: DepthTarget;
  format: GPUTextureFormat;
  private depthFormat: GPUTextureFormat;
  sampleCount: number;

  private msaaColorTexture: GPUTexture | null = null;
  private msaaColorView: GPUTextureView | null = null;
  private msaaDepthTexture: GPUTexture | null = null;
  private msaaDepthView: GPUTextureView | null = null;
  private resolveTexture: GPUTexture | null = null;
  private resolveView: GPUTextureView | null = null;

  toneMapping: ToneMappingMode = "aces";
  exposure = 1.0;
  gamma = 2.2;

  private tonePipeline: GPURenderPipeline | null = null;
  private toneBindGroup: GPUBindGroup | null = null;
  private toneUniformBuffer!: GPUBuffer;
  private toneUniformData = new Float32Array(8);

  constructor(device: GPUDevice, format: GPUTextureFormat = "rgba16float", depthFormat: GPUTextureFormat = "depth24plus", sampleCount = 1) {
    this.device = device;
    this.format = format;
    this.depthFormat = depthFormat;
    this.sampleCount = sampleCount;
    this.toneUniformBuffer = device.createBuffer({
      label: "tone-mapping-ubo",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;

    this.colorTarget?.destroy();
    this.depthTarget?.destroy();
    this.msaaColorTexture?.destroy();
    this.msaaDepthTexture?.destroy();
    this.resolveTexture?.destroy();

    if (this.sampleCount > 1) {
      this.msaaColorTexture = this.device.createTexture({
        label: "msaa-hdr-color",
        size: [width, height],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: this.sampleCount,
      });
      this.msaaColorView = this.msaaColorTexture.createView();

      this.msaaDepthTexture = this.device.createTexture({
        label: "msaa-hdr-depth",
        size: [width, height],
        format: this.depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        sampleCount: this.sampleCount,
      });
      this.msaaDepthView = this.msaaDepthTexture.createView();

      this.resolveTexture = this.device.createTexture({
        label: "hdr-resolve",
        size: [width, height],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.resolveView = this.resolveTexture.createView();

      this.colorTarget = new RenderTarget(
        this.device, width, height, this.format, "hdr-color",
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      );
      this.depthTarget = new DepthTarget(this.device, width, height, "hdr-depth", this.depthFormat);
    } else {
      this.colorTarget = new RenderTarget(
        this.device, width, height, this.format, "hdr-color",
        GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
      );
      this.depthTarget = new DepthTarget(this.device, width, height, "hdr-depth", this.depthFormat);
    }

    this.tonePipeline = null;
    this.toneBindGroup = null;
  }

  get w(): number { return this.width; }
  get h(): number { return this.height; }

  get msaa(): boolean { return this.sampleCount > 1; }

  get msaaColorView_(): GPUTextureView | null { return this.msaaColorView; }
  get msaaDepthView_(): GPUTextureView | null { return this.msaaDepthView; }
  get resolveView_(): GPUTextureView | null { return this.resolveView; }

  get resolvedColorView(): GPUTextureView {
    return this.msaa ? this.resolveView! : this.colorTarget.view;
  }

  get resolvedColorTexture(): GPUTexture {
    return this.msaa ? this.resolveTexture! : this.colorTarget.texture;
  }

  get depthViewForPass(): GPUTextureView {
    return this.msaa ? this.msaaDepthView! : this.depthTarget.view;
  }

  beginRenderPass(encoder: GPUCommandEncoder, clear = true): GPURenderPassEncoder {
    if (this.msaa) {
      return encoder.beginRenderPass({
        colorAttachments: [{
          view: this.msaaColorView!,
          resolveTarget: this.resolveView!,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: clear ? "clear" : "load",
          storeOp: "discard",
        }],
        depthStencilAttachment: {
          view: this.msaaDepthView!,
          depthClearValue: 1.0,
          depthLoadOp: clear ? "clear" : "load",
          depthStoreOp: "discard",
        },
      });
    }
    return encoder.beginRenderPass({
      colorAttachments: [{
        view: this.colorTarget.view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: clear ? "clear" : "load",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depthTarget.view,
        depthClearValue: 1.0,
        depthLoadOp: clear ? "clear" : "load",
        depthStoreOp: "store",
      },
    });
  }

  applyToneMapping(
    encoder: GPUCommandEncoder,
    screenView: GPUTextureView,
    screenFormat: GPUTextureFormat
  ): void {
    this.toneUniformData[0] = this.exposure;
    this.toneUniformData[1] = this.gamma;
    this.toneUniformData[2] = this.toneMapping === "off" ? 0 :
                              this.toneMapping === "aces" ? 1 :
                              this.toneMapping === "filmic" ? 2 : 3;
    this.toneUniformData[3] = 0;
    this.toneUniformData[4] = this.width;
    this.toneUniformData[5] = this.height;
    this.device.queue.writeBuffer(
      this.toneUniformBuffer, 0,
      this.toneUniformData as unknown as GPUAllowSharedBufferSource
    );

    if (!this.tonePipeline) {
      this.createTonePipeline(screenFormat);
    }

    if (!this.toneBindGroup) {
      this.toneBindGroup = this.device.createBindGroup({
        layout: this.tonePipeline!.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.toneUniformBuffer } },
          { binding: 1, resource: this.colorTarget.view },
          { binding: 2, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
        ],
      });
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: screenView,
        loadOp: "load",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.tonePipeline!);
    pass.setBindGroup(0, this.toneBindGroup!);
    pass.draw(3);
    pass.end();
  }

  private createTonePipeline(screenFormat: GPUTextureFormat): void {
    const code = `
struct Uniforms {
  exposure: f32,
  gamma: f32,
  mode: f32,
  pad: f32,
  screenWidth: f32,
  screenHeight: f32,
  pad2: f32,
  pad3: f32,
};
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var hdrTex: texture_2d<f32>;
@group(0) @binding(2) var hdrSampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3)
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

fn acesToneMap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn filmicToneMap(x: vec3<f32>) -> vec3<f32> {
  let A = 0.22; let B = 0.30; let C = 0.10; let D = 0.20; let E = 0.01; let F = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy / vec2<f32>(u.screenWidth, u.screenHeight);
  var color = textureSample(hdrTex, hdrSampler, uv).rgb * u.exposure;

  let mode = i32(u.mode);
  if (mode == 1) {
    color = acesToneMap(color);
  } else if (mode == 2) {
    color = filmicToneMap(color);
  } else if (mode == 3) {
    color = color / (color + vec3<f32>(1.0));
  }

  color = pow(color, vec3<f32>(1.0 / u.gamma));
  return vec4<f32>(color, 1.0);
}`;
    const module = this.device.createShaderModule({ code });

    this.tonePipeline = this.device.createRenderPipeline({
      label: "tone-mapping",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: screenFormat }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  destroy(): void {
    this.colorTarget?.destroy();
    this.depthTarget?.destroy();
    this.msaaColorTexture?.destroy();
    this.msaaDepthTexture?.destroy();
    this.resolveTexture?.destroy();
    this.toneUniformBuffer?.destroy();
  }
}
