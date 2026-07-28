const bloomDownsampleShader = `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(pos[vi], 0.0, 1.0);
  out.uv = pos[vi] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(srcTex));
  let texel = 1.0 / texSize;

  let a = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>(-2.0, -2.0)).rgb;
  let b = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 0.0, -2.0)).rgb;
  let c = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 2.0, -2.0)).rgb;
  let d = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>(-2.0,  0.0)).rgb;
  let e = textureSample(srcTex, srcSampler, in.uv).rgb;
  let f = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 2.0,  0.0)).rgb;
  let g = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>(-2.0,  2.0)).rgb;
  let h = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 0.0,  2.0)).rgb;
  let i = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 2.0,  2.0)).rgb;
  let j = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>(-1.0, -1.0)).rgb;
  let k = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 1.0, -1.0)).rgb;
  let l = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>(-1.0,  1.0)).rgb;
  let m = textureSample(srcTex, srcSampler, in.uv + texel * vec2<f32>( 1.0,  1.0)).rgb;

  var color = (a + c + g + i) * 0.03125;
  color += (b + d + f + h) * 0.0625;
  color += (j + k + l + m) * 0.125;
  color += e * 0.125;

  let brightness = max(color.r, max(color.g, color.b));
  let threshold = 0.8;
  let contribution = max(brightness - threshold, 0.0) / max(brightness, 0.001);

  return vec4<f32>(color * contribution, 1.0);
}
`;

const bloomBlurShader = `
struct BlurParams {
  direction: vec2<f32>,
  mipLevel: f32,
  pad: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(pos[vi], 0.0, 1.0);
  out.uv = pos[vi] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(srcTex));
  let texel = params.direction / texSize;

  let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

  var color = textureSample(srcTex, srcSampler, in.uv).rgb * weights[0];
  for (var i = 1; i < 5; i++) {
    let offset = texel * f32(i) * (1.0 + params.mipLevel * 0.5);
    color += textureSample(srcTex, srcSampler, in.uv + offset).rgb * weights[i];
    color += textureSample(srcTex, srcSampler, in.uv - offset).rgb * weights[i];
  }

  return vec4<f32>(color, 1.0);
}
`;

const bloomCombineShader = `
struct CombineParams {
  bloomIntensity: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;
@group(0) @binding(3) var<uniform> params: CombineParams;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(pos[vi], 0.0, 1.0);
  out.uv = pos[vi] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let scene = textureSample(sceneTex, texSampler, in.uv).rgb;
  let bloom = textureSample(bloomTex, texSampler, in.uv).rgb;
  return vec4<f32>(scene + bloom * params.bloomIntensity, 1.0);
}
`;

const BLUR_SLOT = 256;

export class BloomPass {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private downsamplePipeline!: GPURenderPipeline;
  private blurPipeline!: GPURenderPipeline;
  private combinePipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;
  private blurParamBuffer!: GPUBuffer;
  private combineParamBuffer!: GPUBuffer;
  private pingPong: GPUTexture[] = [];
  private pingPongViews: GPUTextureView[] = [];
  private mipLevels = 5;
  private blurParamData: Float32Array;
  private combineParamData = new Float32Array(4);
  private cachedSceneTexture: GPUTexture | null = null;
  private cachedSceneView: GPUTextureView | null = null;
  private cachedMipViews: GPUTextureView[] = [];
  private cachedMipTextures: GPUTexture[] = [];
  private lastBloomIntensity = -1;

  bloomIntensity = 0.6;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
    this.blurParamData = new Float32Array(this.mipLevels * 2 * (BLUR_SLOT / 4));
    for (let i = 0; i < this.mipLevels; i++) {
      const hBase = i * 2 * (BLUR_SLOT / 4);
      this.blurParamData[hBase + 0] = 1.0;
      this.blurParamData[hBase + 1] = 0.0;
      this.blurParamData[hBase + 2] = i;
      this.blurParamData[hBase + 3] = 0.0;
      const vBase = (i * 2 + 1) * (BLUR_SLOT / 4);
      this.blurParamData[vBase + 0] = 0.0;
      this.blurParamData[vBase + 1] = 1.0;
      this.blurParamData[vBase + 2] = i;
      this.blurParamData[vBase + 3] = 0.0;
    }
    this.init();
  }

  private init() {
    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.blurParamBuffer = this.device.createBuffer({
      label: "blur-params",
      size: this.mipLevels * 2 * BLUR_SLOT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.combineParamBuffer = this.device.createBuffer({
      label: "combine-params",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const dsModule = this.device.createShaderModule({ code: bloomDownsampleShader });
    this.downsamplePipeline = this.device.createRenderPipeline({
      label: "bloom-downsample",
      layout: "auto",
      vertex: { module: dsModule, entryPoint: "vs_main" },
      fragment: { module: dsModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    const blurModule = this.device.createShaderModule({ code: bloomBlurShader });
    const blurBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
      ],
    });
    this.blurPipeline = this.device.createRenderPipeline({
      label: "bloom-blur",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [blurBGL] }),
      vertex: { module: blurModule, entryPoint: "vs_main" },
      fragment: { module: blurModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    const combineModule = this.device.createShaderModule({ code: bloomCombineShader });
    this.combinePipeline = this.device.createRenderPipeline({
      label: "bloom-combine",
      layout: "auto",
      vertex: { module: combineModule, entryPoint: "vs_main" },
      fragment: { module: combineModule, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  private getSceneView(tex: GPUTexture): GPUTextureView {
    if (this.cachedSceneTexture === tex && this.cachedSceneView) return this.cachedSceneView;
    this.cachedSceneTexture = tex;
    this.cachedSceneView = tex.createView();
    return this.cachedSceneView;
  }

  private syncMipViews(mipTargets: GPUTexture[]) {
    for (let i = 0; i < this.mipLevels; i++) {
      if (this.cachedMipTextures[i] !== mipTargets[i]) {
        this.cachedMipTextures[i] = mipTargets[i];
        this.cachedMipViews[i] = mipTargets[i].createView();
      }
    }
  }

  execute(
    encoder: GPUCommandEncoder,
    sceneTexture: GPUTexture,
    mipTargets: GPUTexture[],
    outputTarget: GPUTextureView
  ) {
    const sceneView = this.getSceneView(sceneTexture);
    this.syncMipViews(mipTargets);

    this.device.queue.writeBuffer(this.blurParamBuffer, 0, this.blurParamData as unknown as GPUAllowSharedBufferSource);

    if (this.bloomIntensity !== this.lastBloomIntensity) {
      this.lastBloomIntensity = this.bloomIntensity;
      this.combineParamData[0] = this.bloomIntensity;
      this.device.queue.writeBuffer(this.combineParamBuffer, 0, this.combineParamData as unknown as GPUAllowSharedBufferSource);
    }

    // Downsample scene to mip0
    {
      const bg = this.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sceneView },
          { binding: 1, resource: this.sampler },
        ],
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.cachedMipViews[0],
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.downsamplePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // Progressive downsample through mip chain
    for (let i = 1; i < this.mipLevels; i++) {
      const bg = this.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.cachedMipViews[i - 1] },
          { binding: 1, resource: this.sampler },
        ],
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.cachedMipViews[i],
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.downsamplePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    // Blur each mip with dynamic offset into shared param buffer
    const blurLayout = this.blurPipeline.getBindGroupLayout(0);
    for (let i = 0; i < this.mipLevels; i++) {
      const src = mipTargets[i];
      const pp = this.getPingPong(i, src.width, src.height);
      const ppView = this.pingPongViews[i];

      const bgH = this.device.createBindGroup({
        layout: blurLayout,
        entries: [
          { binding: 0, resource: this.cachedMipViews[i] },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.blurParamBuffer, size: BLUR_SLOT } },
        ],
      });

      // Horizontal: src -> pingpong
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: ppView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass.setPipeline(this.blurPipeline);
        pass.setBindGroup(0, bgH, [i * 2 * BLUR_SLOT]);
        pass.draw(3);
        pass.end();
      }

      const bgV = this.device.createBindGroup({
        layout: blurLayout,
        entries: [
          { binding: 0, resource: ppView },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.blurParamBuffer, size: BLUR_SLOT } },
        ],
      });

      // Vertical: pingpong -> src
      {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.cachedMipViews[i],
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass.setPipeline(this.blurPipeline);
        pass.setBindGroup(0, bgV, [(i * 2 + 1) * BLUR_SLOT]);
        pass.draw(3);
        pass.end();
      }
    }

    // Combine: scene + bloom
    {
      const bg = this.device.createBindGroup({
        layout: this.combinePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sceneView },
          { binding: 1, resource: this.cachedMipViews[0] },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: { buffer: this.combineParamBuffer } },
        ],
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: outputTarget,
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.combinePipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }
  }

  private getPingPong(mipIndex: number, width: number, height: number): GPUTexture {
    const existing = this.pingPong[mipIndex];
    if (existing && existing.width === width && existing.height === height) {
      return existing;
    }
    existing?.destroy();
    this.pingPong[mipIndex] = this.device.createTexture({
      label: `bloom-pingpong-${mipIndex}`,
      size: [width, height],
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.pingPongViews[mipIndex] = this.pingPong[mipIndex].createView();
    return this.pingPong[mipIndex];
  }

  destroy() {
    this.blurParamBuffer.destroy();
    this.combineParamBuffer.destroy();
    for (const t of this.pingPong) t.destroy();
  }
}
