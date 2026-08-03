const NEUTRAL = 0.5;

export class PMXTonemapper {
  private device: GPUDevice;
  private pipeline: GPURenderPipeline | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private toneUBO!: GPUBuffer;
  private gradeUBO!: GPUBuffer;
  private grade2UBO!: GPUBuffer;
  private grade3UBO!: GPUBuffer;
  private bloomParamsUBO!: GPUBuffer;
  private sampler!: GPUSampler;
  private filmicLUT: GPUTexture | null = null;
  private filmicLUTView: GPUTextureView | null = null;
  private prevSceneView: GPUTextureView | null = null;
  private prevBloomView: GPUTextureView | null = null;
  private blackTex: GPUTexture | null = null;
  private blackTexView: GPUTextureView | null = null;

  exposure = 1.0;
  gamma = 2.2;
  contrast = 1.0;
  saturation = 1.0;
  tonemapEnabled = true;
  gradeEnabled = true;

  shadows: [number, number, number] = [NEUTRAL, NEUTRAL, NEUTRAL];
  midtones: [number, number, number] = [NEUTRAL, NEUTRAL, NEUTRAL];
  highlights: [number, number, number] = [NEUTRAL, NEUTRAL, NEUTRAL];

  constructor(device: GPUDevice) {
    this.device = device;
  }

  apply(
    encoder: GPUCommandEncoder,
    screenView: GPUTextureView,
    screenFormat: GPUTextureFormat,
    sceneView: GPUTextureView,
    bloomView: GPUTextureView | null,
    bloomIntensity: number,
  ): void {
    if (!this.pipeline) {
      const lut = this.buildFilmicLUT();
      const code = `
struct Params { exposure: f32, gamma: f32, contrast: f32, flags: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var srcSampler: sampler;
@group(0) @binding(3) var filmicLut: texture_2d<f32>;
@group(0) @binding(4) var<uniform> grade: vec4<f32>;
@group(0) @binding(5) var<uniform> grade2: vec4<f32>;
@group(0) @binding(6) var bloomTex: texture_2d<f32>;
@group(0) @binding(7) var<uniform> bloomParams: vec4<f32>;
@group(0) @binding(8) var<uniform> grade3: vec4<f32>;

fn filmicLUT(x: f32) -> f32 {
  let t = clamp(log2(max(x, 1e-10)) + 10.0, 0.0, 13.0);
  let idx = u32(t * 255.0 / 13.0 + 0.5);
  return textureLoad(filmicLut, vec2u(min(idx, 255u), 0u), 0).r;
}

fn gradeColor(c: vec3f) -> vec3f {
  let slope = grade3.xyz;
  let offset = grade.xyz;
  let power = grade2.xyz;
  var x = pow(max(c * slope + offset, vec3f(0.0)), power);
  x = (x - vec3f(0.5)) * p.contrast + vec3f(0.5);
  let luma = dot(x, vec3f(0.2126, 0.7152, 0.0722));
  return max(mix(vec3f(luma), x, grade2.w), vec3f(0.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3));
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(srcTex));
  let uv = pos.xy / dims;
  let scene = textureSample(srcTex, srcSampler, uv).rgb;
  let bloom = textureSample(bloomTex, srcSampler, uv).rgb;
  var color = (scene + bloom * bloomParams.x) * p.exposure;
  let doTonemap = (p.flags & 1u) != 0u;
  let doGrade = (p.flags & 2u) != 0u;
  if (doTonemap) {
    color = vec3f(filmicLUT(color.r), filmicLUT(color.g), filmicLUT(color.b));
  } else {
    color = clamp(color, vec3f(0.0), vec3f(1.0));
  }
  if (doGrade) {
    color = gradeColor(color);
  }
  color = pow(max(color, vec3f(0.0)), vec3f(1.0 / p.gamma));
  return vec4<f32>(color, 1.0);
}`;
      const module = this.device.createShaderModule({ code });
      this.pipeline = this.device.createRenderPipeline({
        label: "pmx-tonemap",
        layout: "auto",
        vertex: { module, entryPoint: "vs_main" },
        fragment: { module, entryPoint: "fs_main", targets: [{ format: screenFormat }] },
        primitive: { topology: "triangle-list" },
      });
    }

    if (!this.toneUBO) {
      this.toneUBO = this.device.createBuffer({ label: "tone-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!this.gradeUBO) {
      this.gradeUBO = this.device.createBuffer({ label: "grade-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.grade2UBO = this.device.createBuffer({ label: "grade2-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.grade3UBO = this.device.createBuffer({ label: "grade3-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!this.bloomParamsUBO) {
      this.bloomParamsUBO = this.device.createBuffer({ label: "bloom-params-ubo", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }
    if (!this.blackTex) {
      this.blackTex = this.device.createTexture({ label: "tonemap-black", size: [1, 1], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      this.device.queue.writeTexture({ texture: this.blackTex }, new Uint8Array([0, 0, 0, 0]), { bytesPerRow: 4 }, [1, 1]);
      this.blackTexView = this.blackTex.createView();
    }

    const off = (c: number) => (c - NEUTRAL) * 0.5;
    const pow_ = (c: number) => Math.max(0.05, 1 - (c - NEUTRAL) * 1.5);
    const slope = (c: number) => Math.max(0, 1 + (c - NEUTRAL) * 1.5);

    const flags = (this.tonemapEnabled ? 1 : 0) | (this.gradeEnabled ? 2 : 0);
    const data = new ArrayBuffer(16);
    const f32 = new Float32Array(data);
    const u32 = new Uint32Array(data);
    f32[0] = this.exposure; f32[1] = this.gamma; f32[2] = this.contrast; u32[3] = flags;
    this.device.queue.writeBuffer(this.toneUBO, 0, data);

    this.device.queue.writeBuffer(this.gradeUBO, 0, new Float32Array([off(this.shadows[0]), off(this.shadows[1]), off(this.shadows[2]), 0]) as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.grade2UBO, 0, new Float32Array([pow_(this.midtones[0]), pow_(this.midtones[1]), pow_(this.midtones[2]), this.saturation]) as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.grade3UBO, 0, new Float32Array([slope(this.highlights[0]), slope(this.highlights[1]), slope(this.highlights[2]), 0]) as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.bloomParamsUBO, 0, new Float32Array([bloomIntensity, 0, 0, 0]) as unknown as GPUAllowSharedBufferSource);

    if (!this.sampler) {
      this.sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear" });
    }

    const effectiveBloomView = bloomView ?? this.blackTexView!;
    if (!this.bindGroup || sceneView !== this.prevSceneView || effectiveBloomView !== this.prevBloomView) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.toneUBO } },
          { binding: 1, resource: sceneView },
          { binding: 2, resource: this.sampler },
          { binding: 3, resource: this.filmicLUTView! },
          { binding: 4, resource: { buffer: this.gradeUBO } },
          { binding: 5, resource: { buffer: this.grade2UBO } },
          { binding: 6, resource: effectiveBloomView },
          { binding: 7, resource: { buffer: this.bloomParamsUBO } },
          { binding: 8, resource: { buffer: this.grade3UBO } },
        ],
      });
      this.prevSceneView = sceneView;
      this.prevBloomView = effectiveBloomView;
    }

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: screenView, loadOp: "clear", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
  }

  private buildFilmicLUT(): GPUTexture {
    if (this.filmicLUT) return this.filmicLUT;
    const LUT_W = 256;
    const data = new Float32Array(LUT_W * 4);
    const A = 0.22, B = 0.30, C = 0.10, D = 0.20, E = 0.01, F = 0.30;
    const filmicWhite = ((11.2 * (A * 11.2 + C * B) + D * E) / (11.2 * (A * 11.2 + B) + D * F)) - E / F;
    const whiteScale = 1.0 / filmicWhite;
    for (let i = 0; i < LUT_W; i++) {
      const logX = (i / (LUT_W - 1)) * 13.0 - 10.0;
      const x = Math.pow(2, logX);
      const filmic = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
      const v = Math.max(0, filmic * whiteScale);
      data[i * 4 + 0] = v;
      data[i * 4 + 1] = v;
      data[i * 4 + 2] = v;
      data[i * 4 + 3] = 1.0;
    }
    this.filmicLUT = this.device.createTexture({
      label: "filmic-lut",
      size: [LUT_W, 1],
      format: "rgba32float",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.device.queue.writeTexture({ texture: this.filmicLUT }, data as unknown as GPUAllowSharedBufferSource, { bytesPerRow: LUT_W * 16 }, [LUT_W, 1]);
    this.filmicLUTView = this.filmicLUT.createView();
    return this.filmicLUT;
  }

  destroy(): void {
    this.toneUBO?.destroy();
    this.gradeUBO?.destroy();
    this.grade2UBO?.destroy();
    this.grade3UBO?.destroy();
    this.bloomParamsUBO?.destroy();
    this.filmicLUT?.destroy();
    this.blackTex?.destroy();
  }
}
