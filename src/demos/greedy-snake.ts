import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo } from "./types";

const GRID_W = 32;
const GRID_H = 32;

// GreedySnake: CS-driven snake game
// Snake state stored in storage buffer, updated by compute shader
const snakeComputeShader = `
struct SnakeUniforms {
  time: f32,
  deltaTime: f32,
  gridW: f32,
  gridH: f32,
};

struct SnakeState {
  // Grid cells: 0=empty, 1=snake body, 2=snake head, 3=food
  cells: array<u32>,
};

struct SnakeMeta {
  headX: u32,
  headY: u32,
  direction: u32, // 0=up, 1=right, 2=down, 3=left
  length: u32,
  foodX: u32,
  foodY: u32,
  moveTimer: f32,
  score: u32,
};

@group(0) @binding(0) var<uniform> u: SnakeUniforms;
@group(0) @binding(1) var<storage, read_write> grid: array<u32>;
@group(0) @binding(2) var<storage, read_write> snakeMeta: array<u32>;

fn hash1d(x: f32, seed: f32) -> f32 {
  return fract(sin(x * 127.1 + seed) * 43758.5453);
}

@compute @workgroup_size(1)
fn cs_main() {
  let gw = u32(u.gridW);
  let gh = u32(u.gridH);

  var headX = snakeMeta[0];
  var headY = snakeMeta[1];
  var direction = snakeMeta[2];
  var length = snakeMeta[3];
  var foodX = snakeMeta[4];
  var foodY = snakeMeta[5];
  var moveTimer = bitcast<f32>(snakeMeta[6]);
  var score = snakeMeta[7];

  // Move timer
  moveTimer += u.deltaTime;
  let moveInterval = 0.15;

  if (moveTimer >= moveInterval) {
    moveTimer = 0.0;

    // Auto-steer: simple AI that moves toward food
    let dx = i32(foodX) - i32(headX);
    let dy = i32(foodY) - i32(headY);

    // Simple greedy direction
    if (abs(dx) > abs(dy)) {
      if (dx > 0) { direction = 1u; } else { direction = 3u; }
    } else {
      if (dy > 0) { direction = 2u; } else { direction = 0u; }
    }

    // Compute new head position
    var newHeadX = headX;
    var newHeadY = headY;
    if (direction == 0u) { newHeadY = headY - 1u; }
    else if (direction == 1u) { newHeadX = headX + 1u; }
    else if (direction == 2u) { newHeadY = headY + 1u; }
    else { newHeadX = headX - 1u; }

    // Wrap around
    newHeadX = newHeadX % gw;
    newHeadY = newHeadY % gh;

    // Check if ate food
    let ateFood = (newHeadX == foodX && newHeadY == foodY);

    // Move snake: shift body
    if (!ateFood) {
      // Find tail and clear it
      // Simple approach: just move head, clear old head
      grid[headY * gw + headX] = 1u; // old head becomes body
    }

    // Set new head
    headX = newHeadX;
    headY = newHeadY;
    grid[headY * gw + headX] = 2u;

    if (ateFood) {
      length += 1u;
      score += 1u;
      // Spawn new food
      let rndX = hash1d(f32(score) * 1.1, u.time);
      let rndY = hash1d(f32(score) * 2.3, u.time + 1.0);
      foodX = u32(rndX * f32(gw - 1u));
      foodY = u32(rndY * f32(gh - 1u));
      grid[foodY * gw + foodX] = 3u;
    }
  }

  snakeMeta[0] = headX;
  snakeMeta[1] = headY;
  snakeMeta[2] = direction;
  snakeMeta[3] = length;
  snakeMeta[4] = foodX;
  snakeMeta[5] = foodY;
  snakeMeta[6] = bitcast<u32>(moveTimer);
  snakeMeta[7] = score;
}
`;

// Render: display grid as colored quads
const snakeRenderShader = `
struct RenderUniforms {
  gridW: f32,
  gridH: f32,
  time: f32,
  pad: f32,
};

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> grid: array<u32>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  let gw = u32(u.gridW);
  let gh = u32(u.gridH);

  // Each cell = 6 verts (2 triangles)
  let cellIndex = vertexIndex / 6u;
  let vertInCell = vertexIndex % 6u;

  let cellX = cellIndex % gw;
  let cellY = cellIndex / gw;

  let cellValue = grid[cellIndex];

  // Quad corners
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
  );

  let cellSize = vec2<f32>(2.0 / f32(gw), 2.0 / f32(gh));
  let cellOrigin = vec2<f32>(-1.0 + f32(cellX) * cellSize.x, -1.0 + f32(cellY) * cellSize.y);
  let pos = cellOrigin + corners[vertInCell] * cellSize * 0.95;

  var out: VSOut;
  out.position = vec4<f32>(pos, 0.0, 1.0);

  // Color by cell type
  if (cellValue == 0u) {
    out.color = vec3<f32>(0.05, 0.05, 0.08); // empty
  } else if (cellValue == 1u) {
    out.color = vec3<f32>(0.2, 0.8, 0.3); // body
  } else if (cellValue == 2u) {
    out.color = vec3<f32>(0.1, 1.0, 0.4); // head
  } else {
    out.color = vec3<f32>(1.0, 0.3, 0.2); // food
  }

  return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
`;

export class GreedySnakeDemo implements Demo {
  label = "GreedySnake";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;
  private format!: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;
  private gridBuffer!: GPUBuffer;
  private metaBuffer!: GPUBuffer;
  private uniformBuffer!: GPUBuffer;
  private renderUniformBuffer!: GPUBuffer;
  private computeBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;

  private uniformData = new Float32Array(4);
  private renderUniformData = new Float32Array(4);
  private initialized = false;

  async init(ctx: GPUContext, camera: Camera) {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;
    this.format = ctx.format;

    const cellCount = GRID_W * GRID_H;

    this.gridBuffer = this.device.createBuffer({
      label: "snake-grid",
      size: cellCount * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // snakeMeta: headX, headY, direction, length, foodX, foodY, moveTimer(f32 as u32), score
    this.metaBuffer = this.device.createBuffer({
      label: "snake-snakeMeta",
      size: 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.uniformBuffer = this.device.createBuffer({
      label: "snake-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.renderUniformBuffer = this.device.createBuffer({
      label: "snake-render-ubo",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Compute pipeline
    const computeModule = this.device.createShaderModule({ code: snakeComputeShader });
    const computeBGL = this.device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    this.computePipeline = this.device.createComputePipeline({
      label: "snake-compute",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeBGL] }),
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Render pipeline
    const renderModule = this.device.createShaderModule({ code: snakeRenderShader });
    this.renderPipeline = this.device.createRenderPipeline({
      label: "snake-render",
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs_main" },
      fragment: {
        module: renderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.computeBindGroup = this.device.createBindGroup({
      layout: computeBGL,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.gridBuffer } },
        { binding: 2, resource: { buffer: this.metaBuffer } },
      ],
    });

    this.renderBindGroup = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderUniformBuffer } },
        { binding: 1, resource: { buffer: this.gridBuffer } },
      ],
    });
  }

  private initGame() {
    // Initialize grid: snake at center, food random
    const cellCount = GRID_W * GRID_H;
    const grid = new Uint32Array(cellCount);

    // Snake body: 5 cells at center
    const startX = Math.floor(GRID_W / 2);
    const startY = Math.floor(GRID_H / 2);
    for (let i = 0; i < 5; i++) {
      grid[startY * GRID_W + startX - i] = i === 0 ? 2 : 1;
    }

    // Food
    const foodX = Math.floor(Math.random() * GRID_W);
    const foodY = Math.floor(Math.random() * GRID_H);
    grid[foodY * GRID_W + foodX] = 3;

    this.device.queue.writeBuffer(this.gridBuffer, 0, grid);

    // snakeMeta: headX, headY, direction(1=right), length(5), foodX, foodY, moveTimer(0), score(0)
    const snakeMeta = new Uint32Array([startX, startY, 1, 5, foodX, foodY, 0, 0]);
    this.device.queue.writeBuffer(this.metaBuffer, 0, snakeMeta);

    this.initialized = true;
  }

  update(time: number, deltaTime: number) {
    const dt = Math.min(deltaTime, 0.05);
    this.uniformData[0] = time;
    this.uniformData[1] = dt;
    this.uniformData[2] = GRID_W;
    this.uniformData[3] = GRID_H;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformData as unknown as GPUAllowSharedBufferSource);

    this.renderUniformData[0] = GRID_W;
    this.renderUniformData[1] = GRID_H;
    this.renderUniformData[2] = time;
    this.renderUniformData[3] = 0;
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, this.renderUniformData as unknown as GPUAllowSharedBufferSource);
  }

  render(encoder: GPUCommandEncoder, view: GPUTextureView) {
    if (!this.initialized) {
      this.initGame();
    }

    // Compute: update snake
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.computeBindGroup);
    computePass.dispatchWorkgroups(1);
    computePass.end();

    // Render: display grid
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0.02, g: 0.02, b: 0.04, a: 1 },
        loadOp: "clear",
        storeOp: "store",
      }],
    });
    renderPass.setPipeline(this.renderPipeline);
    renderPass.setBindGroup(0, this.renderBindGroup);
    renderPass.draw(GRID_W * GRID_H * 6);
    renderPass.end();
  }

  stats() {
    return {
      drawCalls: 1,
      computeDispatches: 1,
      custom: {
        "Grid": `${GRID_W}x${GRID_H}`,
        "AI": "Greedy (toward food)",
        "Drive": "Compute Shader",
      },
    };
  }

  registerGUI(gui: any) {}

  destroy() {
    this.gridBuffer.destroy();
    this.metaBuffer.destroy();
    this.uniformBuffer.destroy();
    this.renderUniformBuffer.destroy();
  }
}

