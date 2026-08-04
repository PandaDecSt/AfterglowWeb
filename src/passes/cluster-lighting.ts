import { LightScene, LightType, MAX_LIGHTS, CLUSTER_SIZE_X, CLUSTER_SIZE_Y, CLUSTER_SIZE_Z } from "../scene/light";
import { mat4, type Mat4 } from "wgpu-matrix";

export class ClusterLighting {
  private device: GPUDevice;
  private lightScene: LightScene;

  private clusterCount: number;
  private maxLightsPerCluster = 32;

  private clusterBoundsBuffer!: GPUBuffer;
  private clusterLightsBuffer!: GPUBuffer;
  private clusterLightIndicesBuffer!: GPUBuffer;
  private lightGridBuffer!: GPUBuffer;
  private lightIndexBuffer!: GPUBuffer;

  private assignPipeline: GPUComputePipeline | null = null;
  private assignBindGroup: GPUBindGroup | null = null;

  private uniformBuffer!: GPUBuffer;

  constructor(device: GPUDevice, lightScene: LightScene) {
    this.device = device;
    this.lightScene = lightScene;
    this.clusterCount = CLUSTER_SIZE_X * CLUSTER_SIZE_Y * CLUSTER_SIZE_Z;

    const totalClusterIndices = this.clusterCount * this.maxLightsPerCluster;

    this.uniformBuffer = device.createBuffer({
      label: "cluster-uniforms",
      size: 128,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.lightGridBuffer = device.createBuffer({
      label: "cluster-light-grid",
      size: this.clusterCount * 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.lightIndexBuffer = device.createBuffer({
      label: "cluster-light-indices",
      size: totalClusterIndices * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.clusterBoundsBuffer = device.createBuffer({
      label: "cluster-bounds",
      size: this.clusterCount * 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  update(
    viewProj: Mat4,
    invViewProj: Mat4,
    screenWidth: number,
    screenHeight: number,
    near: number,
    far: number,
  ): void {
    const ubo = new Float32Array(32);
    ubo[0] = screenWidth;
    ubo[1] = screenHeight;
    ubo[3] = near;
    ubo[4] = far;
    ubo.set(invViewProj as unknown as ArrayLike<number>, 6);
    ubo[16] = CLUSTER_SIZE_X;
    ubo[17] = CLUSTER_SIZE_Y;
    ubo[18] = CLUSTER_SIZE_Z;
    ubo[19] = this.maxLightsPerCluster;
    ubo[20] = this.lightScene.count;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, ubo as unknown as GPUAllowSharedBufferSource);

    this.buildClusterBounds(near, far, invViewProj, screenWidth, screenHeight);
  }

  private buildClusterBounds(
    near: number,
    far: number,
    invViewProj: Mat4,
    screenWidth: number,
    screenHeight: number,
  ): void {
    const boundsData = new Float32Array(this.clusterCount * 8);
    const tilePixelsX = screenWidth / CLUSTER_SIZE_X;
    const tilePixelsY = screenHeight / CLUSTER_SIZE_Y;

    for (let z = 0; z < CLUSTER_SIZE_Z; z++) {
      const zNear = near * Math.pow(far / near, z / CLUSTER_SIZE_Z);
      const zFar = near * Math.pow(far / near, (z + 1) / CLUSTER_SIZE_Z);

      for (let y = 0; y < CLUSTER_SIZE_Y; y++) {
        for (let x = 0; x < CLUSTER_SIZE_X; x++) {
          const clusterIdx = z * CLUSTER_SIZE_Y * CLUSTER_SIZE_X + y * CLUSTER_SIZE_X + x;
          const offset = clusterIdx * 8;

          const minScreen = [(x * tilePixelsX) / screenWidth, (y * tilePixelsY) / screenHeight];
          const maxScreen = [((x + 1) * tilePixelsX) / screenWidth, ((y + 1) * tilePixelsY) / screenHeight];

          boundsData[offset + 0] = minScreen[0] * 2 - 1;
          boundsData[offset + 1] = 1 - minScreen[1] * 2;
          boundsData[offset + 2] = zNear;
          boundsData[offset + 3] = 0;
          boundsData[offset + 4] = maxScreen[0] * 2 - 1;
          boundsData[offset + 5] = 1 - maxScreen[1] * 2;
          boundsData[offset + 6] = zFar;
          boundsData[offset + 7] = 0;
        }
      }
    }

    this.device.queue.writeBuffer(this.clusterBoundsBuffer, 0, boundsData as unknown as GPUAllowSharedBufferSource);
  }

  assignLightsCPU(): void {
    const gridData = new Int32Array(this.clusterCount * 2);
    const indexData = new Int32Array(this.clusterCount * this.maxLightsPerCluster);
    let globalIndexOffset = 0;

    const lights = this.lightScene.lights;
    const nonDirLights = lights.filter(l => l.type !== LightType.Directional);

    for (let z = 0; z < CLUSTER_SIZE_Z; z++) {
      for (let y = 0; y < CLUSTER_SIZE_Y; y++) {
        for (let x = 0; x < CLUSTER_SIZE_X; x++) {
          const clusterIdx = z * CLUSTER_SIZE_Y * CLUSTER_SIZE_X + y * CLUSTER_SIZE_X + x;
          let count = 0;

          for (let li = 0; li < nonDirLights.length && count < this.maxLightsPerCluster; li++) {
            const light = nonDirLights[li];
            const originalIdx = lights.indexOf(light);

            if (light.type === LightType.Point) {
              const pl = light;
              if (this.isPointLightInCluster(pl, clusterIdx)) {
                indexData[globalIndexOffset + count] = originalIdx;
                count++;
              }
            } else if (light.type === LightType.Spot) {
              const sl = light;
              if (this.isSpotLightInCluster(sl, clusterIdx)) {
                indexData[globalIndexOffset + count] = originalIdx;
                count++;
              }
            }
          }

          gridData[clusterIdx * 2] = globalIndexOffset;
          gridData[clusterIdx * 2 + 1] = count;
          globalIndexOffset += count;
        }
      }
    }

    this.device.queue.writeBuffer(this.lightGridBuffer, 0, gridData as unknown as GPUAllowSharedBufferSource);
    this.device.queue.writeBuffer(this.lightIndexBuffer, 0, indexData as unknown as GPUAllowSharedBufferSource);
  }

  private isPointLightInCluster(light: { position: [number, number, number]; range: number }, _clusterIdx: number): boolean {
    return light.range > 0;
  }

  private isSpotLightInCluster(light: { position: [number, number, number]; range: number }, _clusterIdx: number): boolean {
    return light.range > 0;
  }

  get gridBuffer(): GPUBuffer { return this.lightGridBuffer; }
  get indexBuffer(): GPUBuffer { return this.lightIndexBuffer; }
  get boundsBuffer(): GPUBuffer { return this.clusterBoundsBuffer; }
  get clusterCountValue(): number { return this.clusterCount; }
  get maxPerCluster(): number { return this.maxLightsPerCluster; }

  destroy(): void {
    this.uniformBuffer?.destroy();
    this.lightGridBuffer?.destroy();
    this.lightIndexBuffer?.destroy();
    this.clusterBoundsBuffer?.destroy();
  }
}

export const CLUSTER_LIGHTING_WGSL = `
struct ClusterLightGrid {
  offset: u32,
  count: u32,
};

fn getClusterIndex(
  screenPos: vec2<f32>,
  viewZ: f32,
  screenSize: vec2<f32>,
  near: f32,
  far: f32,
  clusterDim: vec3<u32>,
) -> u32 {
  let tileX = u32(screenPos.x / screenSize.x * f32(clusterDim.x));
  let tileY = u32(screenPos.y / screenSize.y * f32(clusterDim.y));
  let zSlice = u32(log2(max(viewZ, near) / near) / log2(far / near) * f32(clusterDim.z));
  let cx = min(tileX, clusterDim.x - 1u);
  let cy = min(tileY, clusterDim.y - 1u);
  let cz = min(zSlice, clusterDim.z - 1u);
  return cz * clusterDim.y * clusterDim.x + cy * clusterDim.x + cx;
}
`;