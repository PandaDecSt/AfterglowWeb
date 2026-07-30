import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import type { Demo } from "./types";
import type { RenderPass } from "../core/renderer";

const acesShader = `
struct Params {
  resolution: vec2<f32>,
  time: f32,
  exposure: f32,
  whiteClip: f32,
  shoulder: f32,
  mode: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var texSampler: sampler;

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

const D65_TO_D60 = mat3x3<f32>(
  vec3<f32>(1.0130349146, 0.0061052578, -0.0149709436),
  vec3<f32>(0.0076982301, 0.9981633521, -0.0050320385),
  vec3<f32>(-0.0028413174, 0.0046851567, 0.9245061375),
);

const D60_TO_D65 = mat3x3<f32>(
  vec3<f32>(0.9872240087, -0.0061132286, 0.0159532883),
  vec3<f32>(-0.0075983718, 1.0018614847, 0.0053300358),
  vec3<f32>(0.0030725771, -0.0050959615, 1.0816806031),
);

fn ACES_Narkowicz(x: vec3<f32>) -> vec3<f32> {
  let a = x * (x * (x * 60.14595 + 14.22784) + 0.7068982513);
  let b = x * (x * (x * 10.882106 + 56.82012) + 329.7445) + 436.4901;
  return a / b;
}

fn ACES_Hill(x: vec3<f32>) -> vec3<f32> {
  let a = x * (x * (2.51 * x + 0.03) + 0.43);
  let b = x * (x * (2.43 * x + 0.59) + 0.14);
  return a / b;
}

fn ACES_Simplified(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn LinearToSRGB(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  var result: vec3<f32>;
  result.r = select(hi.r, lo.r, c.r < 0.0031308);
  result.g = select(hi.g, lo.g, c.g < 0.0031308);
  result.b = select(hi.b, lo.b, c.b < 0.0031308);
  return result;
}

fn generateScene(uv: vec2<f32>, time: f32) -> vec3<f32> {
  var color = mix(vec3<f32>(0.01, 0.02, 0.04), vec3<f32>(0.1, 0.15, 0.25), uv.y);
  let center = vec2<f32>(0.5 + sin(time * 0.3) * 0.2, 0.5 + cos(time * 0.5) * 0.2);
  let dist = distance(uv, center);
  color += vec3<f32>(50.0, 40.0, 30.0) * exp(-dist * 10.0);
  let center2 = vec2<f32>(0.3, 0.7);
  let dist2 = distance(uv, center2);
  color += vec3<f32>(20.0, 30.0, 50.0) * exp(-dist2 * 15.0);
  let box = smoothstep(0.02, 0.0, abs(uv.x - 0.7)) * smoothstep(0.02, 0.0, abs(uv.y - 0.3));
  color += vec3<f32>(5.0) * box;
  return color;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let hdrColor = generateScene(in.uv, params.time);
  let exposed = hdrColor * params.exposure;

  var result: vec3<f32>;
  let mode = i32(params.mode);

  if (mode == 0) { result = ACES_Simplified(exposed); }
  else if (mode == 1) { result = ACES_Narkowicz(exposed); }
  else if (mode == 2) { result = ACES_Hill(exposed); }
  else { result = exposed / (exposed + vec3<f32>(1.0)); }

  result = D65_TO_D60 * result;
  result = D60_TO_D65 * result;
  result = LinearToSRGB(result);

  return vec4<f32>(clamp(result, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
`;

export class ACESPipelineDemo implements Demo {
  label = "ACES Pipeline";
  private ctx!: GPUContext;
  private camera!: Camera;
  private device!: GPUDevice;
  private format!: GPUTextureFormat;

  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroup!: GPUBindGroup;

  private exposure = 1.0;
  private toneMode = 0;

  async init(ctx: GPUContext, camera: Camera) {
    this.ctx = ctx;
    this.camera = camera;
    this.device = ctx.device;
    this.format = ctx.format;

    this.uniformBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Create a 1x1 dummy texture
    const dummyTex = this.device.createTexture({
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING,
    });

    const bgLayout = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.bindGroup = this.device.createBindGroup({
      layout: bgLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: dummyTex.createView() },
        { binding: 2, resource: this.device.createSampler({ magFilter: "nearest", minFilter: "nearest" }) },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgLayout] }),
      vertex: {
        module: this.device.createShaderModule({ code: acesShader }),
        entryPoint: "vs_main",
      },
      fragment: {
        module: this.device.createShaderModule({ code: acesShader }),
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  update(time: number) {
    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;

    const params = new Float32Array(8);
    params[0] = w;
    params[1] = h;
    params[2] = time;
    params[3] = this.exposure;
    params[4] = 1.0;
    params[5] = 0.5;
    params[6] = this.toneMode;
    params[7] = 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, params);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(3);

        pass.end();
      },
    }];
  }

  destroy() {}

  registerGUI(gui: any) {
    const folder = gui.addFolder("ACES Pipeline");
    folder.add(this, "exposure", 0.1, 5).name("Exposure");
    folder.add(this, "toneMode", { Simplified: 0, Narkowicz: 1, Hill: 2, None: 3 }).name("Tone Map");
  }
}
