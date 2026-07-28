import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";

const METEO_SIZE = 128;

const meteorographInitShader = `
struct Params {
  size: f32,
  time: f32,
  deltaTime: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<vec4<f32>>;

fn hash21(p: vec2<f32>) -> f32 {
  return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn toSnorm(v: f32) -> f32 {
  return v * 2.0 - 1.0;
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x + gid.y * u32(params.size);
  if (id >= u32(params.size * params.size)) { return; }

  let fx = f32(gid.x);
  let fy = f32(gid.y);
  let seed = params.time * 0.1;

  // Initialize with random wind and climate
  let windX = toSnorm(hash21(vec2<f32>(fx + seed, fy))) * 2.0;
  let windY = toSnorm(hash21(vec2<f32>(fx, fy + seed))) * 2.0;
  let humidity = 0.3 + hash21(vec2<f32>(fx * 0.5, fy * 0.5)) * 0.4;
  let temperature = 10.0 + hash21(vec2<f32>(fx * 0.3, fy * 0.7)) * 20.0;

  data[id] = vec4<f32>(windX, windY, humidity, temperature);
}
`;

const meteorographUpdateShader = `
struct Params {
  size: f32,
  time: f32,
  deltaTime: f32,
  windSourceIntensity: f32,
  evaporationRate: f32,
  rainCapacity: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> dataIn: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dataOut: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pluviometer: array<vec4<f32>>;

const SIZE: f32 = ${METEO_SIZE}.0;

fn worldPosFromTileID(id: vec2<f32>) -> vec2<f32> {
  return (id / SIZE - 0.5) * 24.0;
}

fn sampleMeteo(xy: vec2<i32>) -> vec4<f32> {
  let wrapped = (xy + i32(SIZE)) % i32(SIZE);
  return dataIn[u32(wrapped.x) + u32(wrapped.y) * u32(SIZE)];
}

fn hash31(p: vec3<f32>) -> f32 {
  return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453);
}

fn toSnorm(v: f32) -> f32 {
  return v * 2.0 - 1.0;
}

fn perlinNoise(p: vec3<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let a = hash31(i + vec3<f32>(0.0, 0.0, 0.0));
  let b = hash31(i + vec3<f32>(1.0, 0.0, 0.0));
  let c = hash31(i + vec3<f32>(0.0, 1.0, 0.0));
  let d = hash31(i + vec3<f32>(1.0, 1.0, 0.0));
  let e = hash31(i + vec3<f32>(0.0, 0.0, 1.0));
  let ff = hash31(i + vec3<f32>(1.0, 0.0, 1.0));
  let g = hash31(i + vec3<f32>(0.0, 1.0, 1.0));
  let h = hash31(i + vec3<f32>(1.0, 1.0, 1.0));

  return mix(mix(mix(a, b, u.x), mix(c, d, u.x), u.y),
             mix(mix(e, ff, u.x), mix(g, h, u.x), u.y), u.z);
}

fn airResistance(speed: f32) -> f32 {
  return 0.0001 * speed * speed;
}

fn deltaEvaporation(windSpeed: f32, humiture: vec2<f32>, soilMoisture: f32) -> f32 {
  let temp = clamp(humiture.y, 0.0, 100.0);
  let vaporPressure = 610.78 * exp((17.27 * temp) / (temp + 237.3));
  let vpd = max(vaporPressure - humiture.x * vaporPressure, 0.0);
  return 1.2e-4 * max(windSpeed, 0.2) * vpd * soilMoisture * params.deltaTime;
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let id = gid.x + gid.y * u32(SIZE);
  if (id >= u32(SIZE * SIZE)) { return; }

  let xy = vec2<i32>(i32(gid.x), i32(gid.y));
  let worldPos = worldPosFromTileID(vec2<f32>(f32(gid.x), f32(gid.y)));

  // Sample neighbors
  let metT = sampleMeteo(xy + vec2<i32>(0, 1));
  let metB = sampleMeteo(xy + vec2<i32>(0, -1));
  let metR = sampleMeteo(xy + vec2<i32>(1, 0));
  let metL = sampleMeteo(xy + vec2<i32>(-1, 0));
  let met = dataIn[id];

  // Wind iteration from pressure (temperature gradient)
  let deltaTempT = metT.w - met.w;
  let deltaTempB = metB.w - met.w;
  let deltaTempR = metR.w - met.w;
  let deltaTempL = metL.w - met.w;
  let pressureWind = clamp(
    vec2<f32>(deltaTempR + deltaTempL, deltaTempT + deltaTempB) * 0.005,
    vec2<f32>(-0.005), vec2<f32>(0.005)
  );

  // Random wind source
  let windNoiseUV = vec3<f32>(f32(gid.x) + params.time * 2.0, f32(gid.y), params.time) / SIZE;
  let randomWind = vec2<f32>(
    perlinNoise(windNoiseUV * 10.0),
    perlinNoise(windNoiseUV * 10.0 + vec3<f32>(50.0))
  ) * params.windSourceIntensity;

  // Wind update
  var wind = mix(met.xy, vec2<f32>(toSnorm(randomWind.x), toSnorm(randomWind.y)) * 0.5 + 0.5, params.deltaTime * 0.2);
  wind += (vec2<f32>(-metT.y + metB.y, -metR.x + metL.x) + pressureWind) * params.deltaTime;

  // Humidity transport
  var humiture = met.zw;
  let humFlowT = -metT.y * metT.zw;
  let humFlowB = metB.y * metB.zw;
  let humFlowR = -metR.x * metR.zw;
  let humFlowL = metL.x * metL.zw;
  humiture += (humFlowT + humFlowB + humFlowR + humFlowL) * params.deltaTime;

  // Diffusion
  let diffuseWeight = 0.01 * params.deltaTime;
  humiture = humiture * (1.0 - diffuseWeight * 4.0) +
    (metT.zw + metB.zw + metR.zw + metL.zw) * diffuseWeight;

  // Evaporation
  humiture.x += deltaEvaporation(length(wind), humiture, 0.2);

  // Rainfall when humidity saturated
  let rainFall = max(0.0, humiture.x - params.rainCapacity) * 0.5 * params.deltaTime;
  humiture.x -= rainFall;

  // Temperature recovery
  let surfaceTemp = 10.0 + worldPos.y * 0.5;
  humiture.y = mix(humiture.y, surfaceTemp, params.deltaTime * 0.1);

  // Clamp
  humiture = max(humiture, vec2<f32>(0.0, -273.15));

  dataOut[id] = vec4<f32>(wind, humiture);
  pluviometer[id] = vec4<f32>(rainFall, 0.0, 0.0, 0.0);
}
`;

const visualizeShader = `
struct Params {
  size: f32,
  time: f32,
  displayMode: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> data: array<vec4<f32>>;
@group(0) @binding(2) var texSampler: sampler;
@group(0) @binding(3) var outputTex: texture_storage_2d<rgba16float, write>;

const SIZE: f32 = ${METEO_SIZE}.0;

fn toSnorm(v: f32) -> f32 {
  return v * 2.0 - 1.0;
}

@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u32(SIZE) || gid.y >= u32(SIZE)) { return; }

  let id = gid.x + gid.y * u32(SIZE);
  let met = data[id];

  var color: vec3<f32>;
  let mode = i32(params.displayMode);

  if (mode == 0) {
    // Wind field: arrows visualization
    let wind = met.xy;
    let windLen = length(wind);
    let windDir = normalize(wind + vec2<f32>(0.001));
    // Simple arrow: line in wind direction
    let fx = f32(gid.x) / SIZE;
    let fy = f32(gid.y) / SIZE;
    let localPos = vec2<f32>(fract(fx * 8.0), fract(fy * 8.0)) * 2.0 - 1.0;
    let arrowDist = abs(localPos.x - windDir.x * localPos.y) * 0.5;
    let arrowMask = smoothstep(0.1, 0.05, arrowDist) * smoothstep(0.0, 0.8, 1.0 - abs(localPos.y));
    color = vec3<f32>(arrowMask * windLen * 2.0);
  } else if (mode == 1) {
    // Humidity: blue gradient
    color = vec3<f32>(0.1, 0.2, 0.5) * met.z;
    color += vec3<f32>(0.8, 0.9, 1.0) * (1.0 - met.z) * 0.3;
  } else if (mode == 2) {
    // Temperature: hot/cold gradient
    let temp = (met.w + 10.0) / 40.0; // Normalize to [0,1] for -10C to 30C
    color = mix(vec3<f32>(0.2, 0.3, 0.8), vec3<f32>(0.9, 0.3, 0.1), clamp(temp, 0.0, 1.0));
  } else {
    // Combined: humidity * temperature
    color = vec3<f32>(met.z * met.w * 0.02);
  }

  textureStore(outputTex, gid.xy, vec4<f32>(color, 1.0));
}
`;

const fullscreenVS = `
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
`;

const visualizeFS = `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return textureSample(tex, texSampler, in.uv);
}
`;

export class MeteorographDemo implements Demo {
  label = "Meteorograph";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;

  private initPipeline!: GPUComputePipeline;
  private updatePipeline!: GPUComputePipeline;
  private visualizePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;

  private meteoBufferA!: GPUBuffer;
  private meteoBufferB!: GPUBuffer;
  private pluviometerBuffer!: GPUBuffer;
  private paramBuffer!: GPUBuffer;
  private outputTexture!: GPUTexture;
  private outputView!: GPUTextureView;

  private initBindGroup!: GPUBindGroup;
  private updateBindGroupA!: GPUBindGroup;
  private updateBindGroupB!: GPUBindGroup;
  private visualizeBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;

  private sampler!: GPUSampler;
  private frameCount = 0;
  private useBufferA = true;

  displayMode = 0;
  windSourceIntensity = 4.0;
  evaporationRate = 0.00012;
  rainCapacity = 1.0;
  simulationSpeed = 1.0;

  init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;

    const size = METEO_SIZE;
    const bufferSize = size * size * 16; // vec4 = 16 bytes

    // Create buffers
    this.meteoBufferA = this.device.createBuffer({
      label: "meteo-a",
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.meteoBufferB = this.device.createBuffer({
      label: "meteo-b",
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.pluviometerBuffer = this.device.createBuffer({
      label: "pluviometer",
      size: bufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.paramBuffer = this.device.createBuffer({
      label: "meteo-params",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Output texture for visualization
    this.outputTexture = this.device.createTexture({
      label: "meteo-output",
      size: [size, size],
      format: "rgba16float",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.outputView = this.outputTexture.createView();

    this.sampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // BGL for init/update
    const storageBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });

    // Init pipeline
    const initModule = this.device.createShaderModule({ code: meteorographInitShader });
    const initBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.initPipeline = this.device.createComputePipeline({
      label: "meteo-init",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [initBGL] }),
      compute: { module: initModule, entryPoint: "cs_main" },
    });
    this.initBindGroup = this.device.createBindGroup({
      layout: initBGL,
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: { buffer: this.meteoBufferA } },
      ],
    });

    // Update pipeline
    const updateModule = this.device.createShaderModule({ code: meteorographUpdateShader });
    this.updatePipeline = this.device.createComputePipeline({
      label: "meteo-update",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [storageBGL] }),
      compute: { module: updateModule, entryPoint: "cs_main" },
    });
    this.updateBindGroupA = this.device.createBindGroup({
      layout: storageBGL,
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: { buffer: this.meteoBufferA } },
        { binding: 2, resource: { buffer: this.meteoBufferB } },
        { binding: 3, resource: { buffer: this.pluviometerBuffer } },
      ],
    });
    this.updateBindGroupB = this.device.createBindGroup({
      layout: storageBGL,
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: { buffer: this.meteoBufferB } },
        { binding: 2, resource: { buffer: this.meteoBufferA } },
        { binding: 3, resource: { buffer: this.pluviometerBuffer } },
      ],
    });

    // Visualize pipeline
    const visModule = this.device.createShaderModule({ code: visualizeShader });
    const visBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: "rgba16float" } },
      ],
    });
    this.visualizePipeline = this.device.createComputePipeline({
      label: "meteo-vis",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [visBGL] }),
      compute: { module: visModule, entryPoint: "cs_main" },
    });
    this.visualizeBindGroup = this.device.createBindGroup({
      layout: visBGL,
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: { buffer: this.meteoBufferA } },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.outputView },
      ],
    });

    // Render pipeline (fullscreen quad)
    const fsModule = this.device.createShaderModule({ code: visualizeFS });
    const vsModule = this.device.createShaderModule({ code: fullscreenVS });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "meteo-render",
      layout: "auto",
      vertex: { module: vsModule, entryPoint: "vs_main" },
      fragment: {
        module: fsModule,
        entryPoint: "fs_main",
        targets: [{ format: this.ctx.format }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.outputView },
        { binding: 1, resource: this.sampler },
      ],
    });

    // Initialize
    this.dispatchInit();
  }

  private dispatchInit() {
    const encoder = this.device.createCommandEncoder();
    const d = new Float32Array(8);
    d[0] = METEO_SIZE;
    d[1] = performance.now() / 1000;
    this.device.queue.writeBuffer(this.paramBuffer, 0, d);

    const pass = encoder.beginComputePass();
    pass.setPipeline(this.initPipeline);
    pass.setBindGroup(0, this.initBindGroup);
    pass.dispatchWorkgroups(Math.ceil(METEO_SIZE / 16), Math.ceil(METEO_SIZE / 16));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  update(time: number) {
    this.frameCount++;
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    const deltaTime = 1 / 60;
    const d = new Float32Array(8);
    d[0] = METEO_SIZE;
    d[1] = performance.now() / 1000;
    d[2] = deltaTime * this.simulationSpeed;
    d[3] = this.windSourceIntensity;
    d[4] = this.evaporationRate;
    d[5] = this.rainCapacity;

    // Multiple simulation steps per frame
    const steps = Math.max(1, Math.round(this.simulationSpeed));
    for (let i = 0; i < steps; i++) {
      this.device.queue.writeBuffer(this.paramBuffer, 0, d);

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.updatePipeline);
      pass.setBindGroup(0, this.useBufferA ? this.updateBindGroupA : this.updateBindGroupB);
      pass.dispatchWorkgroups(Math.ceil(METEO_SIZE / 16), Math.ceil(METEO_SIZE / 16));
      pass.end();
      this.useBufferA = !this.useBufferA;
    }

    // Update visualize bind group to read from current buffer
    const currentBuffer = this.useBufferA ? this.meteoBufferA : this.meteoBufferB;
    this.visualizeBindGroup = this.device.createBindGroup({
      layout: this.visualizePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: { buffer: currentBuffer } },
        { binding: 2, resource: this.sampler },
        { binding: 3, resource: this.outputView },
      ],
    });

    // Visualize
    const visPass = encoder.beginComputePass();
    visPass.setPipeline(this.visualizePipeline);
    visPass.setBindGroup(0, this.visualizeBindGroup);
    visPass.dispatchWorkgroups(Math.ceil(METEO_SIZE / 16), Math.ceil(METEO_SIZE / 16));
    visPass.end();

    // Render to screen
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroup);
    renderPass.draw(3);
    renderPass.end();
  }

  stats() {
    return {
      drawCalls: 2,
      triangles: 1,
      custom: {
        "Resolution": `${METEO_SIZE}x${METEO_SIZE}`,
        "Display": ["Wind", "Humidity", "Temperature", "Combined"][this.displayMode],
        "Steps/Frame": Math.max(1, Math.round(this.simulationSpeed)),
      },
    };
  }

  registerGUI(gui: any) {
    gui.add(this, "displayMode", { Wind: 0, Humidity: 1, Temperature: 2, Combined: 3 }).name("Display Mode");
    gui.add(this, "simulationSpeed", 0.1, 5, 0.1).name("Simulation Speed");
    gui.add(this, "windSourceIntensity", 0, 10, 0.1).name("Wind Intensity");
    gui.add(this, "rainCapacity", 0.5, 2, 0.1).name("Rain Capacity");
    gui.add(this, "evaporationRate", 0.00005, 0.001, 0.00001).name("Evaporation Rate");
  }

  destroy() {
    this.meteoBufferA.destroy();
    this.meteoBufferB.destroy();
    this.pluviometerBuffer.destroy();
    this.paramBuffer.destroy();
    this.outputTexture.destroy();
  }
}
