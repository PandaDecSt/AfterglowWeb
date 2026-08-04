// TAA: temporal anti-aliasing with motion-vector reprojection,
// 3x3 neighborhood color clamping and exponential history blending.
// Rendered into a ping-pong history pair; the output of the current frame
// becomes the history of the next frame.

export class TAAPass {
  private device: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private history: [GPUTexture, GPUTexture] | null = null;
  private frameIndex = 0;
  private width = 0;
  private height = 0;
  private format: GPUTextureFormat;
  alpha = 0.08;
  debugMode = 0; // 0 = normal, 1 = show motion vectors, 2 = no reprojection, 3 = no history

  constructor(device: GPUDevice, format: GPUTextureFormat = "rgba16float") {
    this.device = device;
    this.format = format;

    const module = this.device.createShaderModule({
      label: "taa",
      code: `
struct TAAUniforms {
  texelSize: vec2<f32>,
  alpha: f32,
  mode: f32, // 0 = normal, 1 = show motion vectors, 2 = disable reprojection, 3 = no history
};

@group(0) @binding(0) var curTex: texture_2d<f32>;
@group(0) @binding(1) var motionTex: texture_2d<f32>;
@group(0) @binding(2) var histTex: texture_2d<f32>;
@group(0) @binding(3) var texSampler: sampler;
@group(0) @binding(4) var<uniform> u: TAAUniforms;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1, -1), vec2<f32>(3, -1), vec2<f32>(-1, 3)
  );
  return vec4<f32>(pos[vi], 0.0, 1.0);
}

fn sampleAt(uv: vec2<f32>) -> vec3<f32> {
  return textureSample(curTex, texSampler, uv).rgb;
}

fn rgb2YCoCg(c: vec3<f32>) -> vec3<f32> {
  let y = (c.r + 2.0 * c.g + c.b) * 0.25;
  return vec3<f32>(y, (c.r - c.b) * 0.5, (c.g - y));
}

fn yCoCg2rgb(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

fn clampAABB(uv: vec2<f32>, hist: vec3<f32>, center: vec3<f32>) -> vec3<f32> {
  let t = u.texelSize;
  // AABB clamp in YCoCg space: luminance separated from chroma,
  // so moving-object color spill is clamped tightly without blowing up HDR
  let n00 = rgb2YCoCg(sampleAt(uv + vec2<f32>(-t.x, -t.y)));
  let n10 = rgb2YCoCg(sampleAt(uv + vec2<f32>( 0.0, -t.y)));
  let n20 = rgb2YCoCg(sampleAt(uv + vec2<f32>( t.x, -t.y)));
  let n01 = rgb2YCoCg(sampleAt(uv + vec2<f32>(-t.x,  0.0)));
  let n11 = rgb2YCoCg(sampleAt(uv + vec2<f32>( 0.0,  0.0)));
  let n21 = rgb2YCoCg(sampleAt(uv + vec2<f32>( t.x,  0.0)));
  let n02 = rgb2YCoCg(sampleAt(uv + vec2<f32>(-t.x,  t.y)));
  let n12 = rgb2YCoCg(sampleAt(uv + vec2<f32>( 0.0,  t.y)));
  let n22 = rgb2YCoCg(sampleAt(uv + vec2<f32>( t.x,  t.y)));

  var mn = min(min(min(n00, n10), min(n20, n01)), min(min(n11, n21), min(n02, n12)));
  var mx = max(max(max(n00, n10), max(n20, n01)), max(max(n11, n21), max(n02, n12)));
  mn = min(mn, n22);
  mx = max(mx, n22);

  // keep luminance box loose but pull chroma box in: kills color ghosting fast
  var cx = (mn.x + mx.x) * 0.5;
  var sx = (mx.x - mn.x) * 0.5;
  mn.x = cx - sx * 1.5;
  mx.x = cx + sx * 1.5;

  let histY = rgb2YCoCg(hist);
  let clampedY = clamp(histY, mn, mx);
  let histHDR = yCoCg2rgb(clampedY);

  // luminance clamping: a history pixel cannot be brighter than the current frame at this pixel
  let curRaw = sampleAt(uv);
  let lumaCur = max(curRaw.r, max(curRaw.g, curRaw.b));
  let lumaHist = max(histHDR.r, max(histHDR.g, histHDR.b));
  let lumScale = select(1.0, lumaCur / max(lumaHist, 1e-4), lumaHist > lumaCur);
  return histHDR * lumScale;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = pos.xy * u.texelSize;
  let cur = sampleAt(uv);
  let motion = textureSample(motionTex, texSampler, uv).xy;

  // debug: visualize motion vectors (motion in uv units, typically < 1/100)
  // neutral (0.5,0.5,0.5) = zero motion; red = +x, green = +y
  if (u.mode == 1.0) {
    let mv = motion * 800.0;
    return vec4<f32>(mv.x * 0.5 + 0.5, mv.y * 0.5 + 0.5, 0.5, 1.0);
  }

  // debug 3: bypass history entirely (current frame only, no blending)
  if (u.mode == 3.0) {
    return vec4<f32>(cur, 1.0);
  }

  // reproject: find this pixel's location in the previous frame
  var histUV = uv - motion;
  let motionPx = length(motion / u.texelSize);
  // motion-adaptive blend: fast motion trusts the current frame more, reducing ghosting
  let alpha = clamp(u.alpha + motionPx * 0.02, u.alpha, 0.5);
  if (u.mode != 2.0) {
    histUV = clamp(histUV, u.texelSize * 0.5, vec2<f32>(1.0) - u.texelSize * 0.5);
    let hist = textureSample(histTex, texSampler, histUV).rgb;
    let clamped = clampAABB(uv, hist, cur);
    return vec4<f32>(mix(clamped, cur, alpha), 1.0);
  }

  // mode 2: no reprojection (history sampled at same pixel) - isolates jitter-only artifacts
  histUV = clamp(histUV, u.texelSize * 0.5, vec2<f32>(1.0) - u.texelSize * 0.5);
  let histNoReproj = textureSample(histTex, texSampler, histUV).rgb;
  let clampedNoReproj = clampAABB(uv, histNoReproj, cur);
  return vec4<f32>(mix(clampedNoReproj, cur, alpha), 1.0);
}
`,
    });

    const bgl = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      label: "taa-pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureHistory(w: number, h: number): void {
    if (this.history && this.width === w && this.height === h) return;
    this.history?.[0].destroy();
    this.history?.[1].destroy();
    this.width = w;
    this.height = h;
    this.history = [
      this.device.createTexture({
        label: "taa-history-0",
        size: [w, h],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
      this.device.createTexture({
        label: "taa-history-1",
        size: [w, h],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    ];
    this.frameIndex = 0;
  }

  execute(
    encoder: GPUCommandEncoder,
    sceneTexture: GPUTexture,
    motionView: GPUTextureView,
    width: number,
    height: number,
  ): { texture: GPUTexture; view: GPUTextureView } {
    this.ensureHistory(width, height);

    // read previous history, write current frame into the other buffer
    const readIdx = this.frameIndex & 1;
    const writeIdx = readIdx ^ 1;
    const histTex = this.history![readIdx];
    const outTex = this.history![writeIdx];

    const data = new Float32Array(4);
    data[0] = 1 / width;
    data[1] = 1 / height;
    data[2] = this.frameIndex === 0 ? 1.0 : this.alpha;
    data[3] = this.debugMode;
    const ubo = this.device.createBuffer({
      label: "taa-uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(ubo, 0, data as unknown as GPUAllowSharedBufferSource);

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sceneTexture.createView() },
        { binding: 1, resource: motionView },
        { binding: 2, resource: histTex.createView() },
        { binding: 3, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
        { binding: 4, resource: { buffer: ubo } },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: "taa",
      colorAttachments: [{
        view: outTex.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();

    this.frameIndex++;
    return { texture: outTex, view: outTex.createView() };
  }

  reset(): void {
    this.frameIndex = 0;
  }

  destroy(): void {
    this.history?.[0].destroy();
    this.history?.[1].destroy();
    this.history = null;
  }
}
