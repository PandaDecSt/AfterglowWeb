import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";

const NOISE_SIZE = 256;

const computeShader = `
const SEED: f32 = 1312.21551;

fn hash2d(xy: vec2<f32>, seed: f32) -> vec2<f32> {
  return fract(sin(vec2<f32>(
    dot(xy, vec2<f32>(46.531, 91.4653)),
    dot(xy, vec2<f32>(64.4634, 49.4349))
  )) * 13127.643 + seed);
}

fn snorm2(v: vec2<f32>) -> vec2<f32> {
  return v * 2.0 - 1.0;
}

fn simplexNoise2d(xy: vec2<f32>, scale: vec2<f32>, seed: f32) -> f32 {
  let scaledXY = xy * scale;
  let F = 0.366025404;
  let G = 0.211324865;

  let skewedID = floor(scaledXY + (scaledXY.x + scaledXY.y) * F);
  let skewedToStdOriginDir = scaledXY - (skewedID - (skewedID.x + skewedID.y) * G);

  var halfSign: vec2<f32>;
  if (skewedToStdOriginDir.x > skewedToStdOriginDir.y) {
    halfSign = vec2<f32>(1.0, 0.0);
  } else {
    halfSign = vec2<f32>(0.0, 1.0);
  }

  let skewedToStdAdjacentDir = skewedToStdOriginDir - halfSign + G;
  let skewedToStdOppositeDir = skewedToStdOriginDir - 1.0 + 2.0 * G;

  let barycenter = vec3<f32>(
    dot(skewedToStdOriginDir, skewedToStdOriginDir),
    dot(skewedToStdAdjacentDir, skewedToStdAdjacentDir),
    dot(skewedToStdOppositeDir, skewedToStdOppositeDir)
  );

  let invHalfBarycenter = max(vec3<f32>(0.5) - barycenter, vec3<f32>(0.0));

  let hashOri = dot(skewedToStdOriginDir, snorm2(hash2d(skewedID, seed)));
  let hashAdj = dot(skewedToStdAdjacentDir, snorm2(hash2d(skewedID + halfSign, seed)));
  let hashOpp = dot(skewedToStdOppositeDir, snorm2(hash2d(skewedID + 1.0, seed)));

  let doubleInv = invHalfBarycenter * invHalfBarycenter;
  let hashGradiant = doubleInv * doubleInv * vec3<f32>(hashOri, hashAdj, hashOpp);

  let result = dot(vec3<f32>(70.0), hashGradiant);
  return result * 0.5 + 0.5;
}

fn worleyNoise2d(xy: vec2<f32>, scale: vec2<f32>, seed: f32) -> f32 {
  let scaledXY = xy * scale;
  let scaledFloorXY = floor(scaledXY);

  var minDist = 1.0;
  for (var offsetY = -1; offsetY <= 1; offsetY++) {
    for (var offsetX = -1; offsetX <= 1; offsetX++) {
      let tilePos = scaledFloorXY + vec2<f32>(f32(offsetX), f32(offsetY));
      let pointPos = vec2<f32>(f32(offsetX), f32(offsetY)) + hash2d(tilePos, seed) + scaledFloorXY;
      let d = distance(scaledXY, pointPos);
      minDist = min(d, minDist);
    }
  }
  return minDist;
}

@group(0) @binding(0) var simplexTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var worleyTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let texRes = vec2<u32>(${NOISE_SIZE}, ${NOISE_SIZE});
  if (gid.x >= texRes.x || gid.y >= texRes.y) {
    return;
  }

  let uv = vec2<f32>(gid.xy) * (1.0 / f32(${NOISE_SIZE}));

  var simplexValue = 0.0;
  let fractalCount = 8u;
  for (var i = 0u; i < fractalCount; i++) {
    let weight = (2.0 * f32(fractalCount - i) / f32(fractalCount)) * (1.0 / f32(fractalCount));
    let freq = pow(2.0, f32(i) + 2.0);
    simplexValue += simplexNoise2d(uv, vec2<f32>(freq), SEED) * weight;
  }
  simplexValue = pow(simplexValue, 2.0);

  var worleyValue = 0.0;
  let worleyFractalCount = 4u;
  for (var i = 0u; i < worleyFractalCount; i++) {
    let weight = (2.0 * f32(worleyFractalCount - i) / f32(worleyFractalCount)) * (1.0 / f32(worleyFractalCount));
    let freq = pow(2.0, f32(i) + 4.0);
    worleyValue += worleyNoise2d(uv, vec2<f32>(freq), SEED) * weight;
  }
  worleyValue = pow(worleyValue * 1.4, 2.0);

  textureStore(simplexTex, gid.xy, vec4<f32>(simplexValue, simplexValue, simplexValue, 1.0));
  textureStore(worleyTex, gid.xy, vec4<f32>(worleyValue, worleyValue, worleyValue, 1.0));
}
`;

const displayShader = `
struct DisplayUniforms {
  time: f32,
  mode: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> du: DisplayUniforms;
@group(0) @binding(1) var simplexTex: texture_2d<f32>;
@group(0) @binding(2) var worleyTex: texture_2d<f32>;
@group(0) @binding(3) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  var out: VSOut;
  out.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
  out.uv = positions[vertexIndex] * 0.5 + 0.5;
  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let s = textureSample(simplexTex, texSampler, in.uv).r;
  let w = textureSample(worleyTex, texSampler, in.uv).r;

  let mode = i32(du.mode) % 4;
  var color: vec3<f32>;
  if (mode == 0) {
    color = vec3<f32>(s);
  } else if (mode == 1) {
    color = vec3<f32>(w);
  } else if (mode == 2) {
    color = vec3<f32>(s, w, s * w);
  } else {
    let n = normalize(vec3<f32>(s - 0.5, w - 0.5, 0.2));
    color = n * 0.5 + 0.5;
  }

  return vec4<f32>(color, 1.0);
}
`;

export class FractalNoiseDemo implements Demo {
  label = "FractalNoise";

  private device!: GPUDevice;
  private format!: GPUTextureFormat;
  private computePipeline!: GPUComputePipeline;
  private displayPipeline!: GPURenderPipeline;
  private simplexTex!: GPUTexture;
  private worleyTex!: GPUTexture;
  private sampler!: GPUSampler;
  private uniformBuffer!: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private displayBindGroup!: GPUBindGroup;

  mode = 2;

  init(ctx: GPUContext) {
    this.device = ctx.device;
    this.format = ctx.format;

    this.simplexTex = this.device.createTexture({
      label: "simplex-noise",
      size: [NOISE_SIZE, NOISE_SIZE],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.worleyTex = this.device.createTexture({
      label: "worley-noise",
      size: [NOISE_SIZE, NOISE_SIZE],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "noise-display-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: "fractal-noise-compute",
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: computeShader }),
        entryPoint: "cs_main",
      },
    });

    this.displayPipeline = this.device.createRenderPipeline({
      label: "fractal-noise-display",
      layout: "auto",
      vertex: {
        module: this.device.createShaderModule({ code: displayShader }),
        entryPoint: "vs_main",
      },
      fragment: {
        module: this.device.createShaderModule({ code: displayShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.computeBindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.simplexTex.createView() },
        { binding: 1, resource: this.worleyTex.createView() },
      ],
    });

    this.displayBindGroup = this.device.createBindGroup({
      layout: this.displayPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: this.simplexTex.createView() },
        { binding: 2, resource: this.worleyTex.createView() },
        { binding: 3, resource: this.sampler },
      ],
    });

    this.runCompute();
  }

  private runCompute() {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.computeBindGroup);
    pass.dispatchWorkgroups(NOISE_SIZE / 8, NOISE_SIZE / 8);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  update(time: number) {
    const data = new Float32Array([time, this.mode, 0, 0]);
    this.device.queue.writeBuffer(this.uniformBuffer, 0, data);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(this.displayPipeline);
    pass.setBindGroup(0, this.displayBindGroup);
    pass.draw(6);
    pass.end();
  }

  stats() {
    return {
      drawCalls: 1,
      computeDispatches: 1,
      custom: { "Texture Size": `${NOISE_SIZE}x${NOISE_SIZE}`, "Mode": ["Simplex", "Worley", "Combined", "Normal"][this.mode] ?? "Unknown" },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "mode", { Simplex: 0, Worley: 1, Combined: 2, Normal: 3 }).name("Noise Mode");
  }

  destroy() {
    this.simplexTex.destroy();
    this.worleyTex.destroy();
    this.uniformBuffer.destroy();
  }
}
