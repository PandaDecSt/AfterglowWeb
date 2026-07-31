// EEVEE 3.6 bloom pyramid: blit (Karis prefilter) → 13-tap downsamples → 9-tap tent upsamples → combine.
// Mirrors source/blender/draw/engines/eevee/shaders/effect_bloom_frag.glsl.

const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`;

const bloomBlitShader = /* wgsl */ `${FULLSCREEN_VS}
@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> prefilter: vec4<f32>;
@group(0) @binding(2) var maskTex: texture_2d<f32>;

fn luminance(c: vec3f) -> f32 {
  return dot(max(c, vec3f(0.0)), vec3f(0.2126, 0.7152, 0.0722));
}

fn fetch(c: vec2<i32>, clampV: f32) -> vec3f {
  let d = vec2<i32>(textureDimensions(hdrTex));
  let cc = clamp(c, vec2<i32>(0), d - vec2<i32>(1));
  let s = textureLoad(hdrTex, cc, 0).rgb;
  let mask = textureLoad(maskTex, cc, 0).r;
  let masked = s * mask;
  return select(masked, min(masked, vec3f(clampV)), clampV > 0.0);
}

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let dst = vec2<i32>(p.xy - vec2f(0.5));
  let base = dst * 2;
  let clampV = prefilter.z;
  let a = fetch(base + vec2<i32>(0, 0), clampV);
  let b = fetch(base + vec2<i32>(1, 0), clampV);
  let c = fetch(base + vec2<i32>(0, 1), clampV);
  let d = fetch(base + vec2<i32>(1, 1), clampV);
  let wa = 1.0 / (1.0 + luminance(a));
  let wb = 1.0 / (1.0 + luminance(b));
  let wc = 1.0 / (1.0 + luminance(c));
  let wd = 1.0 / (1.0 + luminance(d));
  let avg = (a * wa + b * wb + c * wc + d * wd) / max(wa + wb + wc + wd, 1e-6);
  let bright = max(avg.r, max(avg.g, avg.b));
  let soft = clamp(bright - prefilter.x + prefilter.y, 0.0, 2.0 * prefilter.y);
  let q = (soft * soft) / (4.0 * max(prefilter.y, 1e-4) + 1e-6);
  let contrib = max(q, bright - prefilter.x) / max(bright, 1e-4);
  return vec4f(max(avg * contrib, vec3f(0.0)), 1.0);
}
`;

const bloomDownsampleShader = /* wgsl */ `${FULLSCREEN_VS}
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

fn samp(uv: vec2f, off: vec2f) -> vec3f {
  return textureSampleLevel(srcTex, srcSamp, uv + off, 0.0).rgb;
}

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let srcDims = vec2f(textureDimensions(srcTex));
  let t = 1.0 / srcDims;
  let dstDims = srcDims * 0.5;
  let uv = p.xy / max(dstDims, vec2f(1.0));
  let A = samp(uv, t * vec2f(-2.0, -2.0));
  let B = samp(uv, t * vec2f( 0.0, -2.0));
  let C = samp(uv, t * vec2f( 2.0, -2.0));
  let D = samp(uv, t * vec2f(-1.0, -1.0));
  let E = samp(uv, t * vec2f( 1.0, -1.0));
  let F = samp(uv, t * vec2f(-2.0,  0.0));
  let G = samp(uv, t * vec2f( 0.0,  0.0));
  let H = samp(uv, t * vec2f( 2.0,  0.0));
  let I = samp(uv, t * vec2f(-1.0,  1.0));
  let J = samp(uv, t * vec2f( 1.0,  1.0));
  let K = samp(uv, t * vec2f(-2.0,  2.0));
  let L = samp(uv, t * vec2f( 0.0,  2.0));
  let M = samp(uv, t * vec2f( 2.0,  2.0));
  var o = (D + E + I + J) * (0.5 / 4.0);
  o = o + (A + B + G + F) * (0.125 / 4.0);
  o = o + (B + C + H + G) * (0.125 / 4.0);
  o = o + (F + G + L + K) * (0.125 / 4.0);
  o = o + (G + H + M + L) * (0.125 / 4.0);
  return vec4f(o, 1.0);
}
`;

const bloomUpsampleShader = /* wgsl */ `${FULLSCREEN_VS}
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var baseTex: texture_2d<f32>;
@group(0) @binding(2) var srcSamp: sampler;
@group(0) @binding(3) var<uniform> upU: vec4<f32>;

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let srcDims = vec2f(textureDimensions(srcTex));
  let baseDims = vec2f(textureDimensions(baseTex));
  let uv = p.xy / max(baseDims, vec2f(1.0));
  let t = upU.x / srcDims;
  var o = textureSampleLevel(srcTex, srcSamp, uv + t * vec2f(-1.0, -1.0), 0.0).rgb * 1.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f( 0.0, -1.0), 0.0).rgb * 2.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f( 1.0, -1.0), 0.0).rgb * 1.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f(-1.0,  0.0), 0.0).rgb * 2.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f( 0.0,  0.0), 0.0).rgb * 4.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f( 1.0,  0.0), 0.0).rgb * 2.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f(-1.0,  1.0), 0.0).rgb * 1.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f( 0.0,  1.0), 0.0).rgb * 2.0;
  o = o + textureSampleLevel(srcTex, srcSamp, uv + t * vec2f( 1.0,  1.0), 0.0).rgb * 1.0;
  o = o * (1.0 / 16.0);
  let base = textureSampleLevel(baseTex, srcSamp, uv, 0.0).rgb;
  return vec4f(o + base, 1.0);
}
`;

const bloomCombineShader = /* wgsl */ `${FULLSCREEN_VS}
@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;
@group(0) @binding(2) var texSamp: sampler;
@group(0) @binding(3) var<uniform> combineParams: vec4<f32>;

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(sceneTex));
  let uv = p.xy / dims;
  let scene = textureSampleLevel(sceneTex, texSamp, uv, 0.0).rgb;
  let bloom = textureSampleLevel(bloomTex, texSamp, uv, 0.0).rgb;
  return vec4f(scene + bloom * combineParams.x, 1.0);
}
`;

export class BloomPass {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private blitPipeline!: GPURenderPipeline;
  private downsamplePipeline!: GPURenderPipeline;
  private upsamplePipeline!: GPURenderPipeline;
  private combinePipeline!: GPURenderPipeline;
  private linearSampler!: GPUSampler;
  private blitUBO!: GPUBuffer;
  private upsampleUBO!: GPUBuffer;
  private combineUBO!: GPUBuffer;
  private bloomDown: GPUTexture[] = [];
  private bloomDownViews: GPUTextureView[] = [];
  private bloomUp: GPUTexture[] = [];
  private bloomUpViews: GPUTextureView[] = [];
  private mipCount = 0;
  private cachedSceneTex: GPUTexture | null = null;
  private cachedSceneView: GPUTextureView | null = null;
  private whiteTexView!: GPUTextureView;

  bloomIntensity = 0.05;
  threshold = 0.5;
  knee = 0.5;
  radius = 4.0;
  clamp = 0.0;

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;
    this.init();
  }

  private init() {
    this.linearSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.blitUBO = this.device.createBuffer({
      label: "bloom-blit-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.upsampleUBO = this.device.createBuffer({
      label: "bloom-upsample-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.combineUBO = this.device.createBuffer({
      label: "bloom-combine-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const whiteTex = this.device.createTexture({ label: "bloom-white-mask", size: [1, 1], format: "rg8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    this.device.queue.writeTexture({ texture: whiteTex }, new Uint8Array([255, 255, 0, 0]), { bytesPerRow: 4 }, [1, 1]);
    this.whiteTexView = whiteTex.createView();

    const blitModule = this.device.createShaderModule({ code: bloomBlitShader });
    this.blitPipeline = this.device.createRenderPipeline({
      label: "bloom-blit",
      layout: "auto",
      vertex: { module: blitModule, entryPoint: "vs" },
      fragment: { module: blitModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    const dsModule = this.device.createShaderModule({ code: bloomDownsampleShader });
    this.downsamplePipeline = this.device.createRenderPipeline({
      label: "bloom-downsample",
      layout: "auto",
      vertex: { module: dsModule, entryPoint: "vs" },
      fragment: { module: dsModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    const upModule = this.device.createShaderModule({ code: bloomUpsampleShader });
    this.upsamplePipeline = this.device.createRenderPipeline({
      label: "bloom-upsample",
      layout: "auto",
      vertex: { module: upModule, entryPoint: "vs" },
      fragment: { module: upModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });

    const combineModule = this.device.createShaderModule({ code: bloomCombineShader });
    this.combinePipeline = this.device.createRenderPipeline({
      label: "bloom-combine",
      layout: "auto",
      vertex: { module: combineModule, entryPoint: "vs" },
      fragment: { module: combineModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureTextures(w: number, h: number) {
    const shortSide = Math.min(w, h);
    const newMipCount = Math.min(5, Math.max(2, Math.floor(Math.log2(shortSide)) - 1));
    if (newMipCount === this.mipCount && this.bloomDown.length > 0 &&
        this.bloomDown[0].width === Math.floor(w / 2)) return;

    for (const t of this.bloomDown) t.destroy();
    for (const t of this.bloomUp) t.destroy();
    this.bloomDown = [];
    this.bloomUp = [];
    this.bloomDownViews = [];
    this.bloomUpViews = [];
    this.mipCount = newMipCount;

    let mw = Math.floor(w / 2);
    let mh = Math.floor(h / 2);
    for (let i = 0; i < this.mipCount; i++) {
      const tex = this.device.createTexture({
        label: `bloom-down-${i}`,
        size: [Math.max(1, mw), Math.max(1, mh)],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.bloomDown.push(tex);
      this.bloomDownViews.push(tex.createView());
      mw = Math.floor(mw / 2);
      mh = Math.floor(mh / 2);
    }

    mw = Math.floor(w / 2);
    mh = Math.floor(h / 2);
    for (let i = 0; i < this.mipCount - 1; i++) {
      const tex = this.device.createTexture({
        label: `bloom-up-${i}`,
        size: [Math.max(1, mw), Math.max(1, mh)],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.bloomUp.push(tex);
      this.bloomUpViews.push(tex.createView());
      mw = Math.floor(mw / 2);
      mh = Math.floor(mh / 2);
    }
  }

  private getSceneView(tex: GPUTexture): GPUTextureView {
    if (this.cachedSceneTex === tex && this.cachedSceneView) return this.cachedSceneView;
    this.cachedSceneTex = tex;
    this.cachedSceneView = tex.createView();
    return this.cachedSceneView;
  }

  execute(encoder: GPUCommandEncoder, sceneTexture: GPUTexture, outputTarget: GPUTextureView, maskTexture?: GPUTexture) {
    const w = sceneTexture.width;
    const h = sceneTexture.height;
    this.ensureTextures(w, h);

    const sceneView = this.getSceneView(sceneTexture);

    const blitData = new Float32Array([this.threshold, this.knee * 0.5, this.clamp, 0.0]);
    this.device.queue.writeBuffer(this.blitUBO, 0, blitData as unknown as GPUAllowSharedBufferSource);

    const upData = new Float32Array([Math.max(0.5, this.radius), 0.0, 0.0, 0.0]);
    this.device.queue.writeBuffer(this.upsampleUBO, 0, upData as unknown as GPUAllowSharedBufferSource);

    const combineData = new Float32Array([this.bloomIntensity, 0.0, 0.0, 0.0]);
    this.device.queue.writeBuffer(this.combineUBO, 0, combineData as unknown as GPUAllowSharedBufferSource);

    {
      const blitEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: sceneView },
        { binding: 1, resource: { buffer: this.blitUBO } },
      ];
      if (maskTexture) {
        blitEntries.push({ binding: 2, resource: maskTexture.createView() });
      } else {
        blitEntries.push({ binding: 2, resource: this.whiteTexView });
      }
      const bg = this.device.createBindGroup({
        layout: this.blitPipeline.getBindGroupLayout(0),
        entries: blitEntries,
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomDownViews[0],
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.blitPipeline);
      pass.setBindGroup(0, bg);
      pass.draw(3);
      pass.end();
    }

    for (let i = 1; i < this.mipCount; i++) {
      const bg = this.device.createBindGroup({
        layout: this.downsamplePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.bloomDownViews[i - 1] },
          { binding: 1, resource: this.linearSampler },
        ],
      });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomDownViews[i],
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

    const upSteps = this.mipCount - 1;
    if (upSteps > 0) {
      const topIdx = this.mipCount - 1;
      for (let k = 0; k < upSteps; k++) {
        const outIdx = upSteps - 1 - k;
        const srcView = k === 0 ? this.bloomDownViews[topIdx] : this.bloomUpViews[outIdx + 1];
        const baseView = this.bloomDownViews[outIdx];
        const bg = this.device.createBindGroup({
          layout: this.upsamplePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: srcView },
            { binding: 1, resource: baseView },
            { binding: 2, resource: this.linearSampler },
            { binding: 3, resource: { buffer: this.upsampleUBO } },
          ],
        });
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.bloomUpViews[outIdx],
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          }],
        });
        pass.setPipeline(this.upsamplePipeline);
        pass.setBindGroup(0, bg);
        pass.draw(3);
        pass.end();
      }
    }

    const bloomResultView = upSteps > 0 ? this.bloomUpViews[0] : this.bloomDownViews[0];
    {
      const bg = this.device.createBindGroup({
        layout: this.combinePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: sceneView },
          { binding: 1, resource: bloomResultView },
          { binding: 2, resource: this.linearSampler },
          { binding: 3, resource: { buffer: this.combineUBO } },
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

  destroy() {
    this.blitUBO.destroy();
    this.upsampleUBO.destroy();
    this.combineUBO.destroy();
    for (const t of this.bloomDown) t.destroy();
    for (const t of this.bloomUp) t.destroy();
  }
}
