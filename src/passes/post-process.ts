const postProcessShader = `
struct PostParams {
  resolution: vec2<f32>,
  time: f32,
  chromaticStrength: f32,
  fogDensity: f32,
  vignetteStrength: f32,
  exposure: f32,
  saturation: f32,
  fogColor: vec3<f32>,
  cameraPos: vec3<f32>,
  camPad: f32,
  invVP: mat4x4<f32>,
  lightDir: vec3<f32>,
  dustIntensity: f32,
  lightShaftIntensity: f32,
  lightShaftLayers: f32,
  heightFogDensity: f32,
  heightFogBlend: f32,
  bloomEnabled: f32,
  lutStrength: f32,
  contrast: f32,
  colorTemp: f32,
  colorTint: vec3<f32>,
};

@group(0) @binding(0) var sceneTex: texture_2d<f32>;
@group(0) @binding(1) var depthTex: texture_depth_2d;
@group(0) @binding(2) var texSampler: sampler;
@group(0) @binding(3) var depthSampler: sampler;
@group(0) @binding(4) var<uniform> pp: PostParams;

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

fn acesTonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSRGB(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  let s = step(vec3<f32>(0.0031308), c);
  return mix(lo, hi, s);
}

fn reconstructWorldPos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
  let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
  let worldPos = pp.invVP * ndc;
  return worldPos.xyz / worldPos.w;
}

fn hash31(p: vec3<f32>) -> f32 {
  return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn hash21(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn heightFog(worldHeight: f32, sceneDepth: f32) -> f32 {
  let heightDiff = max(pp.cameraPos.z - worldHeight + sceneDepth * 0.25, 0.0);
  let fog = heightDiff * heightDiff * 0.0001 * pp.heightFogDensity;
  return saturate(1.0 - exp(-2.0 * fog));
}

fn atmosphericDust(uv: vec2<f32>, sceneDepth: f32) -> vec3<f32> {
  let timeSeed = fract(floor(pp.time * 20.0) * 0.05);
  let noise = hash31(vec3<f32>(uv * 100.0, timeSeed));
  let dust = noise * min(sceneDepth * 0.001, 0.1);
  return vec3<f32>(dust) * pp.dustIntensity;
}

fn underwaterMask(uv: vec2<f32>) -> f32 {
  // Reconstruct near plane world position
  let ndc = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
  let nearWorldPos = pp.invVP * ndc;
  let nearWorld = nearWorldPos.xyz / nearWorldPos.w;

  // Simple water height comparison (water at y=0)
  let waterHeight = 0.0;
  let heightDiff = waterHeight - nearWorld.z + 0.03;
  return smoothstep(0.0, 0.01, max(heightDiff, 0.0));
}

// Procedural LUT mapping (simulates color grading)
fn lutMapping(color: vec3<f32>) -> vec3<f32> {
  // Contrast
  var c = (color - 0.5) * pp.contrast + 0.5;

  // Color temperature (warm/cool shift)
  c.r += pp.colorTemp * 0.1;
  c.b -= pp.colorTemp * 0.1;

  // Color tint
  c += pp.colorTint * 0.1;

  // Clamp
  c = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));

  // Blend with original based on LUT strength
  return mix(color, c, pp.lutStrength);
}

fn skylightShaft(bloomColor: vec3<f32>, uv: vec2<f32>, sceneDepth: f32) -> vec3<f32> {
  let lightDirCS = (pp.invVP * vec4<f32>(pp.lightDir, 0.0)).xyz;
  let invLayers = 1.0 / max(pp.lightShaftLayers, 1.0);
  let layerInterval = 0.32 * invLayers;
  let hashScale = 0.16 * invLayers;
  let screenLightDir = normalize(vec2<f32>(lightDirCS.x, lightDirCS.y));
  var radialBlur = vec3<f32>(0.0);
  let radialOffset = vec2<f32>(layerInterval, -layerInterval) * lightDirCS.xy * (1.0 - lightDirCS.z);

  for (var i = 1; i <= 8; i++) {
    if (f32(i) > pp.lightShaftLayers) { break; }
    var radialUV = uv + f32(i) * radialOffset;
    let hash = hashScale * (hash21(uv + f32(i)) * 2.0 - 1.0) * screenLightDir;
    radialUV += hash;
    radialUV = clamp(radialUV, vec2<f32>(0.001), vec2<f32>(0.999));
    let depthSample = textureSampleLevel(depthTex, depthSampler, radialUV, 0);
    radialBlur += bloomColor * (1.0 - smoothstep(0.0, 0.001, depthSample - sceneDepth));
  }
  return radialBlur * invLayers * pp.lightShaftIntensity * min(sceneDepth * 0.5, 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  var uv = in.uv;

  // Chromatic aberration
  let dir = uv - 0.5;
  let dist = length(dir);
  let offset = dir * dist * pp.chromaticStrength * 4.0;
  let r = textureSample(sceneTex, texSampler, uv + offset).r;
  let g = textureSample(sceneTex, texSampler, uv).g;
  let b = textureSample(sceneTex, texSampler, uv - offset).b;
  var color = vec3<f32>(r, g, b);

  // Exposure
  color *= pp.exposure;

  // Reconstruct world position
  let depth = textureSampleLevel(depthTex, depthSampler, uv, 0);
  let worldPos = reconstructWorldPos(uv, depth);
  let sceneDepth = length(worldPos - pp.cameraPos);

  // Height fog (world-space)
  let hfog = heightFog(worldPos.z, sceneDepth);
  color = mix(color, pp.fogColor, clamp(hfog * pp.heightFogBlend, 0.0, 1.0));

  // ACES Tonemapping
  color = acesTonemap(color);

  // Saturation
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  color = mix(vec3<f32>(luma), color, pp.saturation);

  // Skylight shaft (god rays) - before bloom for better blending
  // (requires bloom texture, applied externally if enabled)

  // Atmospheric dust
  color += atmosphericDust(uv, sceneDepth);

  // Underwater effect
  let underwater = underwaterMask(uv);
  let underwaterColor = vec3<f32>(0.02, 0.05, 0.12) + pp.fogColor * 0.05;
  color = mix(color, underwaterColor, saturate(1.0 - exp(-sceneDepth * 0.05 - 1.0)) * underwater);

  // LUT mapping (color grading)
  color = lutMapping(color);

  // Vignette
  let vignette = 1.0 - pp.vignetteStrength * dist * dist * 2.0;
  color *= vignette;

  // Gamma
  color = linearToSRGB(color);

  // Subtle film grain
  let grain = fract(sin(dot(uv * pp.resolution + pp.time * 100.0, vec2<f32>(12.9898, 78.233))) * 43758.5453);
  color += (grain - 0.5) * 0.015;

  return vec4<f32>(color, 1.0);
}
`;

export interface PostProcessParams {
  chromaticStrength: number;
  fogDensity: number;
  fogColor: [number, number, number];
  vignetteStrength: number;
  exposure: number;
  saturation: number;
  dustIntensity: number;
  lightShaftIntensity: number;
  lightShaftLayers: number;
  heightFogDensity: number;
  heightFogBlend: number;
  lutStrength: number;
  contrast: number;
  colorTemp: number;
  colorTint: [number, number, number];
}

export class PostProcessPass {
  private device: GPUDevice;
  private pipeline!: GPURenderPipeline;
  private paramBuffer!: GPUBuffer;
  private sampler!: GPUSampler;
  private depthSampler!: GPUSampler;
  private paramData = new Float32Array(64);
  private cachedSceneTex: GPUTexture | null = null;
  private cachedSceneView: GPUTextureView | null = null;
  private cachedDepthView: GPUTextureView | null = null;
  private cachedBindGroup: GPUBindGroup | null = null;
  params: PostProcessParams = {
    chromaticStrength: 0.003,
    fogDensity: 0.02,
    fogColor: [0.4, 0.5, 0.6],
    vignetteStrength: 0.4,
    exposure: 1.2,
    saturation: 1.1,
    dustIntensity: 0.3,
    lightShaftIntensity: 0.25,
    lightShaftLayers: 8,
    heightFogDensity: 0.4,
    heightFogBlend: 1.0,
    lutStrength: 0.5,
    contrast: 1.1,
    colorTemp: 0.0,
    colorTint: [0, 0, 0],
  };


  // External inputs for camera/light
  cameraPos = [0, 2, 6] as [number, number, number];
  invVP = new Float32Array(16);
  lightDir = [0.4, 0.8, 0.3] as [number, number, number];

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    this.depthSampler = device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });
    this.paramBuffer = device.createBuffer({
      label: "postprocess-params",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const module = device.createShaderModule({ code: postProcessShader });
    this.pipeline = device.createRenderPipeline({
      label: "postprocess",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private ensureBindGroup(sceneTexture: GPUTexture, depthView: GPUTextureView) {
    if (this.cachedSceneTex === sceneTexture && this.cachedDepthView === depthView && this.cachedBindGroup) {
      return;
    }
    this.cachedSceneTex = sceneTexture;
    this.cachedDepthView = depthView;
    this.cachedSceneView = sceneTexture.createView();
    this.cachedBindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.cachedSceneView },
        { binding: 1, resource: this.cachedDepthView },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.depthSampler },
        { binding: 4, resource: { buffer: this.paramBuffer } },
      ],
    });
  }

  execute(
    encoder: GPUCommandEncoder,
    sceneTexture: GPUTexture,
    depthView: GPUTextureView,
    outputView: GPUTextureView,
    resolution: [number, number],
    time: number
  ) {
    const p = this.params;
    const d = this.paramData;
    // PostParams layout:
    // 0-1: resolution, 2: time, 3: chromaticStrength
    // 4: fogDensity, 5: vignetteStrength, 6: exposure, 7: saturation
    // 8-10: fogColor, 11: pad
    // 12-14: cameraPos, 15: camPad
    // 16-31: invVP
    // 32-34: lightDir, 35: dustIntensity
    // 36: lightShaftIntensity, 37: lightShaftLayers, 38: heightFogDensity, 39: heightFogBlend
    d[0] = resolution[0];
    d[1] = resolution[1];
    d[2] = time;
    d[3] = p.chromaticStrength;
    d[4] = p.fogDensity;
    d[5] = p.vignetteStrength;
    d[6] = p.exposure;
    d[7] = p.saturation;
    d[8] = p.fogColor[0];
    d[9] = p.fogColor[1];
    d[10] = p.fogColor[2];
    d[11] = 0;
    // Camera position
    d[12] = this.cameraPos[0];
    d[13] = this.cameraPos[1];
    d[14] = this.cameraPos[2];
    d[15] = 0;
    // Inverse VP matrix
    d.set(this.invVP, 16);
    // Light direction
    d[32] = this.lightDir[0];
    d[33] = this.lightDir[1];
    d[34] = this.lightDir[2];
    d[35] = p.dustIntensity;
    d[36] = p.lightShaftIntensity;
    d[37] = p.lightShaftLayers;
    d[38] = p.heightFogDensity;
    d[39] = p.heightFogBlend;
    d[40] = 1.0; // bloomEnabled
    d[41] = p.lutStrength;
    d[42] = p.contrast;
    d[43] = p.colorTemp;
    d[44] = p.colorTint[0];
    d[45] = p.colorTint[1];
    d[46] = p.colorTint[2];
    d[47] = 0;
    this.device.queue.writeBuffer(this.paramBuffer, 0, d as unknown as GPUAllowSharedBufferSource);


    this.ensureBindGroup(sceneTexture, depthView);

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: outputView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.cachedBindGroup!);
    pass.draw(3);
    pass.end();
  }

  destroy() {
    this.paramBuffer.destroy();
  }
}
