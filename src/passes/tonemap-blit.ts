// Minimal HDR → swapchain resolve: exposure, ACES filmic tonemap, gamma.
//
// The deferred lighting pass outputs linear HDR (rgba16float). Something has to
// bring that back into display range, and the full PostProcessPass (fog, light
// shafts, dust, depth input) is far too heavy for a demo that just wants a
// picture on screen. This is the "plain ending" of the pipeline; swap in
// PostProcessPass when a demo wants the fancy one.

const TONEMAP_BLIT_WGSL = `
struct Cfg {
  exposure: f32,
  gamma: f32,
  /** 0 = ACES filmic, 1 = clamp only (keeps flat toon colors unmuddied). */
  mode: f32,
  _pad0: f32,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var<uniform> cfg: Cfg;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  return vec4<f32>(p[vi], 0.0, 1.0);
}

// Narkowicz ACES approximation.
fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(srcTex));
  let uv = fragCoord.xy / dims;
  let hdr = max(textureSampleLevel(srcTex, srcSampler, uv, 0.0).rgb * cfg.exposure, vec3<f32>(0.0));
  // ACES rolls highlights off beautifully for PBR, but it desaturates the
  // deliberately flat colors of a cel-shaded character. Mode 1 skips it.
  let mapped = select(acesTonemap(hdr), clamp(hdr, vec3<f32>(0.0), vec3<f32>(1.0)), cfg.mode > 0.5);
  let sdr = pow(mapped, vec3<f32>(1.0 / max(cfg.gamma, 0.01)));
  return vec4<f32>(sdr, 1.0);
}
`;

export class TonemapBlitPass {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline;
  private bindGroupLayout: GPUBindGroupLayout;
  private sampler: GPUSampler;
  private cfgBuffer: GPUBuffer;
  private cachedSourceView: GPUTextureView | null = null;
  private bindGroup: GPUBindGroup | null = null;

  exposure = 1.0;
  gamma = 2.2;
  /** 0 = ACES filmic (realistic), 1 = clamp only (toon-safe). */
  mode = 0;

  constructor(device: GPUDevice, outputFormat: GPUTextureFormat) {
    this.device = device;

    this.sampler = device.createSampler({
      label: "tonemap-blit-sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.cfgBuffer = device.createBuffer({
      label: "tonemap-blit-cfg",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.bindGroupLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });

    const module = device.createShaderModule({ label: "tonemap-blit", code: TONEMAP_BLIT_WGSL });
    this.pipeline = device.createRenderPipeline({
      label: "tonemap-blit-pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: outputFormat }] },
      primitive: { topology: "triangle-list" },
    });
  }

  execute(encoder: GPUCommandEncoder, sourceView: GPUTextureView, targetView: GPUTextureView): void {
    this.device.queue.writeBuffer(
      this.cfgBuffer,
      0,
      new Float32Array([this.exposure, this.gamma, this.mode, 0]) as unknown as GPUAllowSharedBufferSource,
    );

    if (!this.bindGroup || this.cachedSourceView !== sourceView) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.bindGroupLayout,
        entries: [
          { binding: 0, resource: sourceView },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.cfgBuffer } },
        ],
      });
      this.cachedSourceView = sourceView;
    }

    const pass = encoder.beginRenderPass({
      label: "tonemap-blit",
      colorAttachments: [{
        view: targetView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this.cfgBuffer?.destroy();
  }
}
