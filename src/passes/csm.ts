import { mat4, vec3, type Mat4, type Vec3 } from "wgpu-matrix";

export const CSM_CASCADE_COUNT = 4;

export interface CSMSplit {
  near: number;
  far: number;
  vp: Mat4;
  orthoSize: number;
}

function vec3MulScalar(v: Vec3, s: number): Vec3 {
  return vec3.create(v[0] * s, v[1] * s, v[2] * s);
}

function mat4TransformPoint(m: Mat4, p: [number, number, number]): Vec3 {
  const x = p[0], y = p[1], z = p[2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15] || 1;
  return vec3.create(
    (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
    (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
    (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
  );
}

export class CascadedShadowMap {
  private device: GPUDevice;
  private cascadeTextures: GPUTexture[] = [];
  private cascadeViews: GPUTextureView[] = [];
  private shadowSampler!: GPUSampler;
  private uniformBuffer!: GPUBuffer;

  size: number;
  format: GPUTextureFormat = "depth32float";
  generation = 0;

  lightDirection: Vec3 = vec3.create(0.5, 1.0, 0.3);
  lightColor: Vec3 = vec3.create(1, 1, 1);
  intensity = 3.0;

  cascadeSplits: number[] = [0.05, 0.15, 0.35, 1.0];
  cascadeVPs: Mat4[] = [];
  cascadeOrthoSizes: number[] = [];

  bias = 0.001;
  normalBias = 0.08;
  blendZone = 0.1;

  private uniformData!: Float32Array;

  constructor(device: GPUDevice, size = 2048) {
    this.device = device;
    this.size = size;
    this.createResources();
  }

  private createResources(): void {
    for (let i = 0; i < CSM_CASCADE_COUNT; i++) {
      const tex = this.device.createTexture({
        label: `csm-cascade-${i}`,
        size: [this.size, this.size],
        format: this.format,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.cascadeTextures.push(tex);
      this.cascadeViews.push(tex.createView());
    }

    this.shadowSampler = this.device.createSampler({
      label: "csm-comparison",
      compare: "less",
      magFilter: "linear",
      minFilter: "linear",
    });

    this.uniformData = new Float32Array(16 * CSM_CASCADE_COUNT + CSM_CASCADE_COUNT + 4);
    this.uniformBuffer = this.device.createBuffer({
      label: "csm-uniforms",
      size: this.uniformData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  get views(): GPUTextureView[] { return this.cascadeViews; }
  get sampler(): GPUSampler { return this.shadowSampler; }
  get ubo(): GPUBuffer { return this.uniformBuffer; }

  beginCascadePass(encoder: GPUCommandEncoder, cascadeIndex: number): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      label: `csm-pass-${cascadeIndex}`,
      colorAttachments: [],
      depthStencilAttachment: {
        view: this.cascadeViews[cascadeIndex],
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });
  }

  updateCascadeVPs(
    cameraViewProj: Mat4,
    cameraInvViewProj: Mat4,
    cameraNear: number,
    cameraFar: number,
    sceneCenter: Vec3 = vec3.create(0, 0, 0),
    sceneExtent = 20.0,
  ): void {
    this.cascadeVPs = [];
    this.cascadeOrthoSizes = [];

    const dir = vec3.normalize(this.lightDirection);
    const up = Math.abs(dir[1]) > 0.99 ? vec3.create(0, 0, -1) : vec3.create(0, 1, 0);

    for (let i = 0; i < CSM_CASCADE_COUNT; i++) {
      const splitNear = cameraNear + this.cascadeSplits[i] * (cameraFar - cameraNear);
      const splitFar = cameraNear + this.cascadeSplits[i + 1 < CSM_CASCADE_COUNT ? i + 1 : i] * (cameraFar - cameraNear);

      const corners = this.getFrustumCorners(cameraInvViewProj, splitNear, splitFar);
      const center = this.getFrustumCenter(corners);
      const radius = this.getFrustumRadius(corners, center);

      const lightView = mat4.lookAt(
        vec3.add(center, vec3MulScalar(dir, radius * 2)),
        center,
        up,
      );

      const orthoSize = radius * 2.0;
      const lightProj = mat4.ortho(
        -orthoSize, orthoSize,
        -orthoSize, orthoSize,
        0.1, radius * 4.0,
      );

      const remap = mat4.identity(mat4.create());
      remap[10] = 0.5;
      remap[14] = 0.5;

      const vp = mat4.multiply(remap, mat4.multiply(lightProj, lightView));
      this.cascadeVPs.push(vp);
      this.cascadeOrthoSizes.push(orthoSize);
    }

    this.uploadUniforms(cameraNear, cameraFar);
  }

  private getFrustumCorners(invViewProj: Mat4, near: number, far: number): Vec3[] {
    const ndcCorners: [number, number][] = [
      [-1, -1], [1, -1], [1, 1], [-1, 1],
    ];

    const corners: Vec3[] = [];
    for (const [x, y] of ndcCorners) {
      const nearPt = mat4TransformPoint(invViewProj, [x, y, -1]);
      const farPt = mat4TransformPoint(invViewProj, [x, y, 1]);
      const nearCorner = vec3.add(vec3MulScalar(nearPt, 1 - near), vec3MulScalar(farPt, near));
      const farCorner = vec3.add(vec3MulScalar(nearPt, 1 - far), vec3MulScalar(farPt, far));
      corners.push(nearCorner, farCorner);
    }
    return corners;
  }

  private getFrustumCenter(corners: Vec3[]): Vec3 {
    let cx = 0, cy = 0, cz = 0;
    for (const c of corners) { cx += c[0]; cy += c[1]; cz += c[2]; }
    const n = corners.length;
    return vec3.create(cx / n, cy / n, cz / n);
  }

  private getFrustumRadius(corners: Vec3[], center: Vec3): number {
    let maxDist = 0;
    for (const c of corners) {
      const d = vec3.distance(c, center);
      if (d > maxDist) maxDist = d;
    }
    return maxDist;
  }

  private uploadUniforms(cameraNear: number, cameraFar: number): void {
    const data = this.uniformData;
    let offset = 0;

    for (let i = 0; i < CSM_CASCADE_COUNT; i++) {
      data.set(this.cascadeVPs[i] as unknown as ArrayLike<number>, offset);
      offset += 16;
    }

    for (let i = 0; i < CSM_CASCADE_COUNT; i++) {
      const splitNear = cameraNear + this.cascadeSplits[i] * (cameraFar - cameraNear);
      data[offset + i] = splitNear;
    }
    offset += CSM_CASCADE_COUNT;

    data[offset + 0] = this.bias;
    data[offset + 1] = this.normalBias;
    data[offset + 2] = this.blendZone;
    data[offset + 3] = CSM_CASCADE_COUNT;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data as unknown as GPUAllowSharedBufferSource);
  }

  resize(size: number): void {
    if (size === this.size) return;
    for (const t of this.cascadeTextures) t.destroy();
    this.cascadeTextures = [];
    this.cascadeViews = [];
    this.size = size;
    this.generation++;
    this.createResources();
  }

  destroy(): void {
    for (const t of this.cascadeTextures) t.destroy();
    this.uniformBuffer?.destroy();
  }
}

export const CSM_WGSL = `
const CSM_CASCADE_COUNT: u32 = ${CSM_CASCADE_COUNT};

fn selectCSMCascade(viewZ: f32, splitDistances: array<f32, ${CSM_CASCADE_COUNT}>) -> i32 {
  for (var i = 0; i < ${CSM_CASCADE_COUNT}; i++) {
    if (viewZ < splitDistances[i]) {
      return i;
    }
  }
  return ${CSM_CASCADE_COUNT} - 1;
}

fn sampleCSMPCF(
  shadowTextures: array<texture_depth_2d, ${CSM_CASCADE_COUNT}>,
  shadowSampler: sampler_comparison,
  cascadeVPs: array<mat4x4<f32>, ${CSM_CASCADE_COUNT}>,
  worldPos: vec3<f32>,
  normal: vec3<f32>,
  lightDir: vec3<f32>,
  cascadeIndex: i32,
  normalBias: f32,
  shadowBias: f32,
) -> f32 {
  if (dot(normal, lightDir) <= 0.0) { return 1.0; }

  let biasedPos = worldPos + normal * normalBias;
  let lclip = cascadeVPs[cascadeIndex] * vec4<f32>(biasedPos, 1.0);
  let ndc = lclip.xyz / max(lclip.w, 1e-6);
  let suv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let cmpZ = ndc.z - shadowBias;

  let tex = shadowTextures[cascadeIndex];
  let ts = 1.0 / f32(textureDimensions(tex).x);

  let s00 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(tex, shadowSampler, suv, cmpZ);
  let s21 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(tex, shadowSampler, suv + vec2f( ts,  ts), cmpZ);

  var vis = (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);

  let inZ = select(0.0, 1.0, ndc.z > 0.0 && ndc.z < 1.0);
  let frustum = (1.0 - smoothstep(0.88, 0.96, abs(ndc.x)))
              * (1.0 - smoothstep(0.88, 0.96, abs(ndc.y))) * inZ;
  return mix(1.0, vis, frustum);
}
`;