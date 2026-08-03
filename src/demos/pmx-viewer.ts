import { GPUContext } from "../core/device";
import { Camera } from "../scene/camera";
import { Demo, ShaderStageDesc } from "./types";
import type { EngineContext } from "../core/engine";
import type { RenderPass } from "../core/renderer";
import { Skeleton, type BoneDesc } from "../scene/skeleton";
import { Skinning } from "../scene/skinning";
import { AnimationPlayer } from "../scene/animation-player";
import { loadPMX, type PMXModel, type PMXMaterial } from "../utils/pmx-loader";
import { ShadowMap } from "../passes/shadow";
import { BloomPass } from "../passes/bloom";
import { HDRRenderTarget } from "../passes/hdr";
import { mat4, quat, vec3 } from "wgpu-matrix";
import { BrdfLut } from "../passes/brdf-lut";
import { loadVMD, vmdDuration, type VMDCameraFrame } from "../utils/vmd-loader";
import { buildIKChains, solveIK, type IKChain } from "../scene/ik-solver";
import { MMDPhysics } from "../scene/physics";
import { buildPhysicsRigidbodies, buildPhysicsJoints } from "../scene/pmx-physics-bridge";
import { GPUComputeSkinning } from "../scene/gpu-skinning";
import { GPUComputeMorph } from "../scene/gpu-morph";
import { CameraAnimation } from "../scene/camera-animation";
import { MmdCoord } from "../scene/mmd-coord";
import { MorphDeformer } from "../scene/morph-deformer";
import { SCENE_VS } from "../shader/pmx/scene-vs";
import { buildMainFS } from "../shader/pmx/main-fs";
import { SHADOW_VS } from "../shader/pmx/shadow-vs";
import { OUTLINE_VS } from "../shader/pmx/outline-vs";
import { OUTLINE_FS } from "../shader/pmx/outline-fs";
import { RenderClass, PresetConfig, PRESETS, detectPreset } from "../scene/pmx-preset";
import { PMXTonemapper } from "../passes/pmx-tonemapper";
import { create1x1Texture, createToonRampTexture, loadTextureImage } from "../utils/texture-utils";
import { IKDebugRenderer } from "../debug/ik-debug-renderer";

const HDR_FORMAT = "rgba16float";

interface MatRenderData {
  indexOffset: number;
  indexCount: number;
  mainBG: GPUBindGroup;
  shadowBG: GPUBindGroup;
  shadowGen: number;
  outlineBG: GPUBindGroup | null;
  isTransparent: boolean;
  hasEdge: boolean;
  renderClass: RenderClass;
  castsShadow: boolean;
}

export class PMXDemo implements Demo {
  label = "PMX Viewer";

  private device!: GPUDevice;
  private ctx!: GPUContext;
  private camera!: Camera;

  private mainPipeline!: GPURenderPipeline;
  private eyePipeline!: GPURenderPipeline;
  private hairPipeline!: GPURenderPipeline;
  private hairOverEyesPipeline!: GPURenderPipeline;
  private shadowPipeline!: GPURenderPipeline;
  private outlinePipeline!: GPURenderPipeline;
  private sceneBuffer!: GPUBuffer;
  private sceneData = new Float32Array(80);
  private sceneDataU32 = new Uint32Array(this.sceneData.buffer);
  private vertexBuffer!: GPUBuffer;

  private morphDeformer: MorphDeformer | null = null;
  private indexBuffer!: GPUBuffer;
  private totalIndexCount = 0;
  private use32bit = false;

  private shadowMap!: ShadowMap;
  private shadowBGLayout!: GPUBindGroupLayout;
  private hdrTarget!: HDRRenderTarget;
  private bloom!: BloomPass;
  private brdfLut!: BrdfLut;

  private bloomMaskTex: GPUTexture | null = null;
  private bloomMaskView: GPUTextureView | null = null;
  private bloomOutput: GPUTexture | null = null;
  private bloomOutputView: GPUTextureView | null = null;
  private tonemapper!: PMXTonemapper;

  private matRenders: MatRenderData[] = [];
  private opaqueOrder: MatRenderData[] = [];
  private gpuTextures: GPUTexture[] = [];

  private skeleton: Skeleton | null = null;
  private skinning: Skinning | null = null;
  private animPlayer: AnimationPlayer | null = null;
  private ikChains: IKChain[] = [];

  private skinMatrixStorageBuffer!: GPUBuffer;
  private skinBGL!: GPUBindGroupLayout;
  private skinBG!: GPUBindGroup;

  private shadowSceneBuffer!: GPUBuffer;
  private shadowSceneData = new Float32Array(36);
  private shadowSceneDataU32 = new Uint32Array(this.shadowSceneData.buffer);
  private shadowSceneBG!: GPUBindGroup;

  bloomEnabled = true;
  stencilEnabled = true;
  get tonemapEnabled() { return this.tonemapper?.tonemapEnabled ?? true; }
  set tonemapEnabled(v: boolean) { if (this.tonemapper) this.tonemapper.tonemapEnabled = v; }
  get gradeEnabled() { return this.tonemapper?.gradeEnabled ?? true; }
  set gradeEnabled(v: boolean) { if (this.tonemapper) this.tonemapper.gradeEnabled = v; }
  debugIK = false;
  animPaused = false;
  physicsEnabled = true;
  gpuSkinningEnabled = false;
  lightX = 5;
  lightY = 10;
  lightZ = 8;
  shadowRes = 2048;

  private physics: MMDPhysics | null = null;
  private gpuSkinning: GPUComputeSkinning | null = null;
  private gpuMorph: GPUComputeMorph | null = null;
  private gpuMorphIndices: number[] = [];
  private gpuMorphWeights: Float32Array | null = null;
  gpuMorphEnabled = false;
  private vmdCameraAnimation: CameraAnimation | null = null;
  private orbitCameraAnimation: CameraAnimation | null = null;
  cameraAnimEnabled = false;
  cameraAnimSource = "none";
  faceLockEnabled = false;
  private _headBoneIdx = -1;
  private _headPos = new Float32Array(3);

  private loaded = false;
  private _modelCenterY = 10;
  private _modelRadius = 20;

  private _depthTex: GPUTexture | null = null;
  private _depthW = 0;
  private _depthH = 0;

  async init(ctx: GPUContext, camera: Camera, engine?: EngineContext): Promise<void> {
    this.device = ctx.device;
    this.ctx = ctx;
    this.camera = camera;

    try {
      const pmx = await loadPMX("/model.pmx");
      console.log(`[PMXDemo] Loaded: ${pmx.name}, V:${pmx.vertices.length} I:${pmx.indices.length} M:${pmx.materials.length} B:${pmx.bones.length} T:${pmx.textures.length}`);
      await this.setupFromPMX(pmx);
      this.loaded = true;
    } catch (e) {
      console.error("[PMXDemo] Load failed:", e);
    }
  }

  private async setupFromPMX(pmx: PMXModel): Promise<void> {
    const vertexCount = pmx.vertices.length;
    const vertexStride = 56;
    const vertexBuf = new ArrayBuffer(vertexCount * vertexStride);
    const dv = new DataView(vertexBuf);
    let minPos = [Infinity, Infinity, Infinity];
    let maxPos = [-Infinity, -Infinity, -Infinity];

    for (let i = 0; i < vertexCount; i++) {
      const v = pmx.vertices[i];
      const off = i * vertexStride;
      dv.setFloat32(off, v.position[0], true);
      dv.setFloat32(off + 4, v.position[1], true);
      dv.setFloat32(off + 8, v.position[2], true);
      dv.setFloat32(off + 12, v.normal[0], true);
      dv.setFloat32(off + 16, v.normal[1], true);
      dv.setFloat32(off + 20, v.normal[2], true);
      dv.setFloat32(off + 24, v.uv[0], true);
      dv.setFloat32(off + 28, v.uv[1], true);
      const bj = v.boneIndices;
      const bw = v.boneWeights;
      dv.setUint16(off + 32, bj.length > 0 ? Math.max(0, Math.min(65535, bj[0])) : 0, true);
      dv.setUint16(off + 34, bj.length > 1 ? Math.max(0, Math.min(65535, bj[1])) : 0, true);
      dv.setUint16(off + 36, bj.length > 2 ? Math.max(0, Math.min(65535, bj[2])) : 0, true);
      dv.setUint16(off + 38, bj.length > 3 ? Math.max(0, Math.min(65535, bj[3])) : 0, true);
      dv.setFloat32(off + 40, bw.length > 0 ? bw[0] : 0, true);
      dv.setFloat32(off + 44, bw.length > 1 ? bw[1] : 0, true);
      dv.setFloat32(off + 48, bw.length > 2 ? bw[2] : 0, true);
      dv.setFloat32(off + 52, bw.length > 3 ? bw[3] : 0, true);
      for (let k = 0; k < 3; k++) { if (v.position[k] < minPos[k]) minPos[k] = v.position[k]; if (v.position[k] > maxPos[k]) maxPos[k] = v.position[k]; }
    }

    const center = [(minPos[0] + maxPos[0]) / 2, (minPos[1] + maxPos[1]) / 2, (minPos[2] + maxPos[2]) / 2];
    const extent = [maxPos[0] - minPos[0], maxPos[1] - minPos[1], maxPos[2] - minPos[2]];
    const radius = Math.sqrt(extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2) / 2;
    this.camera.orbit(vec3.create(center[0], center[1], center[2]), radius * 2.5, radius * 0.01, radius * 20, Math.PI / 2, 0);
    this._modelCenterY = center[1];
    this._modelRadius = radius;
    console.log(`[PMXDemo] Bounds: center=[${center.map(v => v.toFixed(2))}] radius=${radius.toFixed(2)} orbitDist=${(radius * 2.5).toFixed(2)}`);

    this.vertexBuffer = this.device.createBuffer({ label: "pmx-vb", size: vertexBuf.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(this.vertexBuffer.getMappedRange()).set(new Uint8Array(vertexBuf));
    this.vertexBuffer.unmap();

    const baseVertices = new Float32Array(vertexBuf);
    this.morphDeformer = new MorphDeformer(this.device, this.vertexBuffer, baseVertices, pmx.morphs);
    console.log(`[PMXDemo] Vertex morphs: ${this.morphDeformer.morphCount} / ${pmx.morphs.length} total`);

    this.use32bit = vertexCount > 65535;
    this.totalIndexCount = pmx.indices.length;
    const indexSize = this.use32bit ? 4 : 2;
    this.indexBuffer = this.device.createBuffer({ label: "pmx-ib", size: this.totalIndexCount * indexSize, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    if (this.use32bit) { new Int32Array(this.indexBuffer.getMappedRange()).set(pmx.indices); }
    else { new Uint16Array(this.indexBuffer.getMappedRange()).set(pmx.indices); }
    this.indexBuffer.unmap();

    this.shadowMap = new ShadowMap(this.device, 2048);
    this.shadowMap.orthoSize = 64;
    this.shadowMap.near = 1;
    this.shadowMap.far = 140;

    this.buildPipelines();

    this.sceneBuffer = this.device.createBuffer({ label: "pmx-scene-ubo", size: this.sceneData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    this.shadowSceneBuffer = this.device.createBuffer({ label: "pmx-shadow-scene-ubo", size: this.shadowSceneData.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const shadowGroup0Layout = this.shadowPipeline.getBindGroupLayout(0);
    this.shadowSceneBG = this.device.createBindGroup({
      label: "pmx-shadow-scene-bg",
      layout: shadowGroup0Layout,
      entries: [{ binding: 0, resource: { buffer: this.shadowSceneBuffer } }],
    });

    const skinBoneCount = Math.max(1, pmx.bones.length);
    this.skinMatrixStorageBuffer = this.device.createBuffer({
      label: "pmx-skin-storage",
      size: skinBoneCount * 16 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (pmx.bones.length === 0) {
      const identity = new Float32Array(16);
      identity[0] = 1; identity[5] = 1; identity[10] = 1; identity[15] = 1;
      this.device.queue.writeBuffer(this.skinMatrixStorageBuffer, 0, identity as unknown as GPUAllowSharedBufferSource);
    }
    this.skinBG = this.device.createBindGroup({
      label: "pmx-skin-bg",
      layout: this.skinBGL,
      entries: [{ binding: 0, resource: { buffer: this.skinMatrixStorageBuffer } }],
    });

    this.brdfLut = new BrdfLut();
    this.brdfLut.bake(this.device);

    const defaultTex = create1x1Texture(this.device, 255, 255, 255, 255, "default-white");
    const toonRampTex = createToonRampTexture(this.device);
    this.gpuTextures.push(defaultTex, toonRampTex);

    const loadedTextures: (GPUTexture | null)[] = [defaultTex];
    for (let i = 0; i < pmx.textures.length; i++) {
      let texPath = pmx.textures[i].path.replace(/\\/g, "/");
      if (!texPath.startsWith("/")) texPath = "/" + texPath;
      if (texPath.startsWith("//")) texPath = texPath.slice(1);
      const tex = await loadTextureImage(this.device, texPath, `pmx-tex-${i}`);
      if (i < 3) console.log(`[PMXDemo] tex ${i}: path="${pmx.textures[i].path}" url="${texPath}" loaded=${tex !== null}`);
      if (tex) this.gpuTextures.push(tex);
      loadedTextures.push(tex);
    }

    const sampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "repeat" });
    const mainBGL = this.mainPipeline.getBindGroupLayout(0);
    const outlineBGL = this.outlinePipeline.getBindGroupLayout(0);

    let indexOffset = 0;
    for (let mi = 0; mi < pmx.materials.length; mi++) {
      const m = pmx.materials[mi];
      const matIndexCount = m.faceCount;
      const isTransparent = m.diffuse[3] < 1.0 - 0.001;
      const hasEdge = (m.flag & 0x10) !== 0 && m.edgeScale > 0;

      const matBuf = this.device.createBuffer({ label: `pmx-mat-${mi}`, size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const preset = detectPreset(m.name, isTransparent);
      const matData = new Float32Array(24);
      matData[0] = m.diffuse[0]; matData[1] = m.diffuse[1]; matData[2] = m.diffuse[2]; matData[3] = m.diffuse[3];
      matData[4] = m.ambient[0]; matData[5] = m.ambient[1]; matData[6] = m.ambient[2]; matData[7] = m.specularPower;
      matData[8] = m.specular[0]; matData[9] = m.specular[1]; matData[10] = m.specular[2]; matData[11] = m.sphereMode;
      matData[12] = preset.metallic; matData[13] = preset.roughness; matData[14] = preset.emissionStrength; matData[15] = preset.nprMix;
      matData[16] = preset.rimColor[0]; matData[17] = preset.rimColor[1]; matData[18] = preset.rimColor[2]; matData[19] = preset.rimStrength;
      matData[20] = preset.rimPower;
      matData[21] = preset.alphaMode ?? 0;
      this.device.queue.writeBuffer(matBuf, 0, matData as unknown as GPUAllowSharedBufferSource);

      const diffuseTex = (m.textureIndex >= 0 && m.textureIndex + 1 < loadedTextures.length && loadedTextures[m.textureIndex + 1]) ? loadedTextures[m.textureIndex + 1]! : defaultTex;
      const sphereTex = (m.sphereTextureIndex >= 0 && m.sphereTextureIndex + 1 < loadedTextures.length && loadedTextures[m.sphereTextureIndex + 1]) ? loadedTextures[m.sphereTextureIndex + 1]! : defaultTex;
      const toonTex = (m.toonSharing === 0 && m.toonTextureIndex >= 0 && m.toonTextureIndex + 1 < loadedTextures.length && loadedTextures[m.toonTextureIndex + 1])
        ? loadedTextures[m.toonTextureIndex + 1]! : toonRampTex;

      const mainBG = this.device.createBindGroup({
        label: `pmx-main-bg-${mi}`, layout: mainBGL,
        entries: [
          { binding: 0, resource: { buffer: this.sceneBuffer } },
          { binding: 1, resource: { buffer: matBuf } },
          { binding: 2, resource: diffuseTex.createView() },
          { binding: 3, resource: sphereTex.createView() },
          { binding: 4, resource: toonTex.createView() },
          { binding: 5, resource: sampler },
          { binding: 6, resource: this.brdfLut.view },
        ],
      });

      const shadowBG = this.device.createBindGroup({
        label: `pmx-shadow-bg-${mi}`, layout: this.shadowBGLayout,
        entries: [
          { binding: 0, resource: this.shadowMap.view },
          { binding: 1, resource: this.shadowMap.sampler },
          { binding: 2, resource: { buffer: this.shadowMap.getVPBuffer() } },
        ],
      });

      let outlineBG: GPUBindGroup | null = null;
      if (hasEdge) {
        const edgeBuf = this.device.createBuffer({ label: `pmx-edge-${mi}`, size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const edgeData = new Float32Array(8);
        edgeData[0] = m.edgeColor[0]; edgeData[1] = m.edgeColor[1]; edgeData[2] = m.edgeColor[2]; edgeData[3] = m.edgeColor[3];
        edgeData[4] = m.edgeScale;
        this.device.queue.writeBuffer(edgeBuf, 0, edgeData as unknown as GPUAllowSharedBufferSource);
        outlineBG = this.device.createBindGroup({
          label: `pmx-outline-bg-${mi}`, layout: outlineBGL,
          entries: [
            { binding: 0, resource: { buffer: this.sceneBuffer } },
            { binding: 1, resource: { buffer: edgeBuf } },
            { binding: 2, resource: diffuseTex.createView() },
            { binding: 5, resource: sampler },
          ],
        });
      }

      const castsShadow = (m.flag & 0x04) !== 0;
      this.matRenders.push({ indexOffset, indexCount: matIndexCount, mainBG, shadowBG, shadowGen: this.shadowMap.generation, outlineBG, isTransparent, hasEdge, renderClass: preset.renderClass, castsShadow });
      indexOffset += matIndexCount;
    }

    const rcRank = (rc: RenderClass) => rc === "eye" ? 1 : rc === "hair" ? 2 : 0;
    this.opaqueOrder = this.matRenders
      .filter(mr => !mr.isTransparent)
      .sort((a, b) => rcRank(a.renderClass) - rcRank(b.renderClass));

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    this.hdrTarget = new HDRRenderTarget(this.device, HDR_FORMAT, "depth24plus-stencil8");
    this.hdrTarget.toneMapping = "filmic";
    this.hdrTarget.resize(w, h);
    this.bloom = new BloomPass(this.device, this.ctx.supportsRG11B10 ? "rg11b10ufloat" as GPUTextureFormat : "rgba16float" as GPUTextureFormat);
    this.tonemapper = new PMXTonemapper(this.device);
    this.ikDebugRenderer = new IKDebugRenderer(this.device, this.ctx.format);

    if (pmx.bones.length > 0) {
      const boneDescs: BoneDesc[] = pmx.bones.map((b, i) => {
        let px = b.position[0], py = b.position[1], pz = b.position[2];
        if (b.parentIndex >= 0 && b.parentIndex < pmx.bones.length) {
          const parent = pmx.bones[b.parentIndex];
          px -= parent.position[0];
          py -= parent.position[1];
          pz -= parent.position[2];
        }
        return {
          name: b.name,
          parentIndex: b.parentIndex,
          position: vec3.create(px, py, pz),
          rotation: quat.identity(quat.create()),
          scale: vec3.create(1, 1, 1),
          appendParentIndex: b.appendParentIndex,
          appendRatio: b.appendRatio,
          appendRotate: b.appendRotate,
          appendMove: b.appendMove,
        };
      });
      this.skeleton = new Skeleton(boneDescs);
      for (const name of ["頭", "頭部", "head", "Head"]) {
        this._headBoneIdx = this.skeleton.getBoneIndex(name);
        if (this._headBoneIdx >= 0) break;
      }
      if (this._headBoneIdx < 0) this._headBoneIdx = this.skeleton.getBoneIndex("両目") ?? -1;
      console.log(`[PMXDemo] Head bone idx: ${this._headBoneIdx}${this._headBoneIdx >= 0 ? ' (' + this.skeleton.boneNames[this._headBoneIdx] + ')' : ' (not found)'}`);
      const joints = new Uint16Array(vertexCount * 4);
      const weights = new Float32Array(vertexCount * 4);
      for (let i = 0; i < vertexCount; i++) { const v = pmx.vertices[i]; for (let j = 0; j < 4; j++) { joints[i * 4 + j] = v.boneIndices.length > j ? v.boneIndices[j] : 0; weights[i * 4 + j] = v.boneWeights.length > j ? v.boneWeights[j] : 0; } }
      this.skinning = new Skinning(vertexCount, 4, joints, weights, pmx.bones.length);
      this.skeleton.updateWorldMatrices();
      this.skeleton.computeSkinMatrices(this.skinning.skinMatrixData);
      this.device.queue.writeBuffer(this.skinMatrixStorageBuffer, 0, this.skinning.skinMatrixData as unknown as GPUAllowSharedBufferSource);
      this.animPlayer = new AnimationPlayer(this.skeleton, pmx.morphs.length);
      this.ikChains = buildIKChains(pmx.bones);
      console.log(`[PMXDemo] IK chains: ${this.ikChains.length}`);
      for (const c of this.ikChains) {
        const linkNames = c.links.map(l => `${pmx.bones[l.index].name}${l.hasLimit ? `[${l.limitMin[0].toFixed(1)},${l.limitMax[0].toFixed(1)}]x[${l.limitMin[1].toFixed(1)},${l.limitMax[1].toFixed(1)}]y[${l.limitMin[2].toFixed(1)},${l.limitMax[2].toFixed(1)}]z` : ""}`).join(" <- ");
        console.log(`  IK: ${pmx.bones[c.targetIndex].name} -> effector=${pmx.bones[c.effectorIndex].name} iter=${c.iterations} maxAngle=${c.maxAngle.toFixed(3)} links: ${linkNames}`);
      }

      try {
        const boneNames = pmx.bones.map(b => b.name);
        const morphNames = pmx.morphs.map(m => m.name);

        const vmd = await loadVMD("/motions.vmd");
        this.animPlayer.playVMD(vmd, boneNames, morphNames, { loop: true });
        console.log(`[PMXDemo] VMD loaded: "${vmd.name}", duration=${vmdDuration(vmd).toFixed(2)}s, bones=${vmd.boneFrames.size}, morphs=${vmd.morphFrames.size}, cameras=${vmd.cameraFrames.length}`);

        try {
          const vmd2 = await loadVMD("/motion.vmd");
          if (vmd2.boneFrames.size > 0 || vmd2.morphFrames.size > 0) {
            this.animPlayer.playVMD(vmd2, boneNames, morphNames, { loop: true });
            console.log(`[PMXDemo] Extra VMD: "${vmd2.name}", bones=${vmd2.boneFrames.size}, morphs=${vmd2.morphFrames.size}, cameras=${vmd2.cameraFrames.length}`);
          }
        } catch { }

        this.orbitCameraAnimation = this.createOrbitCameraTrack(vmdDuration(vmd), this._modelCenterY, this._modelRadius);
        if (vmd.cameraFrames.length > 0) {
          this.vmdCameraAnimation = new CameraAnimation(vmd.cameraFrames);
          this.cameraAnimSource = `VMD (${vmd.cameraFrames.length} frames)`;
          console.log(`[PMXDemo] VMD camera: ${vmd.cameraFrames.length} frames`);
        } else {
          this.vmdCameraAnimation = null;
          this.cameraAnimSource = "no VMD camera data";
          console.warn(`[PMXDemo] VMD has no camera frames`);
        }
        this.cameraAnimEnabled = false;
        console.log(`[PMXDemo] Camera animation: source=${this.cameraAnimSource}, orbit track ready (${this.orbitCameraAnimation.frameCount} frames)`);
      } catch (e) {
        console.warn("[PMXDemo] VMD load failed:", e);
      }
    }

    if (pmx.rigidbodies.length > 0) {
      try {
        const physRbs = buildPhysicsRigidbodies(pmx.rigidbodies);
        const physJoints = buildPhysicsJoints(pmx.joints);
        this.physics = new MMDPhysics(physRbs, physJoints);
        console.log(`[PMXDemo] Physics initialized: ${pmx.rigidbodies.length} rigidbodies, ${pmx.joints.length} joints`);
      } catch (e) {
        console.warn("[PMXDemo] Physics init failed:", e);
      }
    }

    if (this.skeleton && this.skinning) {
      try {
        const boneIndicesData = new Uint32Array(vertexCount * 4);
        const boneWeightsData = new Float32Array(vertexCount * 4);
        for (let i = 0; i < vertexCount; i++) {
          const v = pmx.vertices[i];
          const bj = v.boneIndices;
          const bw = v.boneWeights;
          const off4 = i * 4;
          boneIndicesData[off4 + 0] = bj.length > 0 ? bj[0] : 0;
          boneIndicesData[off4 + 1] = bj.length > 1 ? bj[1] : 0;
          boneIndicesData[off4 + 2] = bj.length > 2 ? bj[2] : 0;
          boneIndicesData[off4 + 3] = bj.length > 3 ? bj[3] : 0;
          boneWeightsData[off4 + 0] = bw.length > 0 ? bw[0] : 0;
          boneWeightsData[off4 + 1] = bw.length > 1 ? bw[1] : 0;
          boneWeightsData[off4 + 2] = bw.length > 2 ? bw[2] : 0;
          boneWeightsData[off4 + 3] = bw.length > 3 ? bw[3] : 0;
        }
        this.gpuSkinning = new GPUComputeSkinning(this.device);
        this.gpuSkinning.setup(
          vertexCount,
          pmx.bones.length,
          14,
          new Float32Array(vertexBuf),
          this.skinning.skinMatrixData,
          boneIndicesData,
          boneWeightsData,
        );
        console.log(`[PMXDemo] GPU Compute Skinning ready`);
      } catch (e) {
        console.warn("[PMXDemo] GPU skinning init failed:", e);
        this.gpuSkinning = null;
      }
    }

    if (this.morphDeformer) {
      try {
        const vertexMorphs: { pmxIndex: number; offsets: { vertexIndex: number; position: Float32Array }[] }[] = [];
        for (let i = 0; i < pmx.morphs.length; i++) {
          if (pmx.morphs[i].type === 1) vertexMorphs.push({ pmxIndex: i, offsets: pmx.morphs[i].offsets });
        }
        const vertexCount = pmx.vertices.length;
        const morphCount = vertexMorphs.length;
        const deltas = new Float32Array(morphCount * vertexCount * 3);
        for (let mi = 0; mi < morphCount; mi++) {
          const base = mi * vertexCount * 3;
          for (const off of vertexMorphs[mi].offsets) {
            const d3 = base + off.vertexIndex * 3;
            deltas[d3] = off.position[0];
            deltas[d3 + 1] = off.position[1];
            deltas[d3 + 2] = off.position[2];
          }
        }
        this.gpuMorphIndices = vertexMorphs.map(m => m.pmxIndex);
        this.gpuMorphWeights = new Float32Array(morphCount);
        this.gpuMorph = new GPUComputeMorph(this.device);
        this.gpuMorph.setup(vertexCount, morphCount, 14, new Float32Array(vertexBuf), deltas);
        console.log(`[PMXDemo] GPU Compute Morph ready: ${morphCount} morphs, deltas=${(deltas.byteLength / 1024).toFixed(0)}KB`);
      } catch (e) {
        console.warn("[PMXDemo] GPU morph init failed:", e);
        this.gpuMorph = null;
      }
    }
  }

  private buildPipelines(): void {
    const vsModule = this.device.createShaderModule({ code: SCENE_VS });
    const fsModule = this.device.createShaderModule({ code: buildMainFS() });
    const shadowVSModule = this.device.createShaderModule({ code: SHADOW_VS });
    const outVSModule = this.device.createShaderModule({ code: OUTLINE_VS });

    vsModule.getCompilationInfo().then(info => {
      for (const msg of info.messages) console.log(`[PMXDemo] VS compile: ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}`);
    });
    fsModule.getCompilationInfo().then(info => {
      for (const msg of info.messages) console.log(`[PMXDemo] FS compile: ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}`);
    });
    const outFSModule = this.device.createShaderModule({ code: OUTLINE_FS });

    const vertexLayout = {
      arrayStride: 56,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" as const },
        { shaderLocation: 1, offset: 12, format: "float32x3" as const },
        { shaderLocation: 2, offset: 24, format: "float32x2" as const },
        { shaderLocation: 3, offset: 32, format: "uint16x4" as const },
        { shaderLocation: 4, offset: 40, format: "float32x4" as const },
      ],
    };

    const blendState = {
      color: { srcFactor: "src-alpha" as const, dstFactor: "one-minus-src-alpha" as const, operation: "add" as const },
      alpha: { srcFactor: "one" as const, dstFactor: "one-minus-src-alpha" as const, operation: "add" as const },
    };

    this.shadowBGLayout = this.device.createBindGroupLayout({
      label: "shadow-bg-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });

    const mainGroup0 = this.device.createBindGroupLayout({
      label: "main-group0",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    });

    this.skinBGL = this.device.createBindGroupLayout({
      label: "skin-bg-layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    const DS_FORMAT: GPUTextureFormat = "depth24plus-stencil8";
    const mainLayout = this.device.createPipelineLayout({ bindGroupLayouts: [mainGroup0, this.shadowBGLayout, this.skinBGL] });
    const mainTargets = [{ format: HDR_FORMAT as GPUTextureFormat, blend: blendState }, { format: "rg8unorm" as GPUTextureFormat }];

    this.mainPipeline = this.device.createRenderPipeline({
      label: "pmx-main",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: { format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });

    this.eyePipeline = this.device.createRenderPipeline({
      label: "pmx-eye",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets, constants: { IS_EYE: 1 } },
      primitive: { topology: "triangle-list", cullMode: "back", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less", depthBias: -1, depthBiasSlopeScale: 0.0,
        stencilFront: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilBack: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilReadMask: 0xff, stencilWriteMask: 0xff,
      },
    });

    this.hairPipeline = this.device.createRenderPipeline({
      label: "pmx-hair",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less",
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff, stencilWriteMask: 0,
      },
    });

    this.hairOverEyesPipeline = this.device.createRenderPipeline({
      label: "pmx-hair-over-eyes",
      layout: mainLayout,
      vertex: { module: vsModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: fsModule, entryPoint: "fs_main", targets: mainTargets, constants: { IS_OVER_EYES: 1 } },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: false, depthCompare: "less-equal",
        stencilFront: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff, stencilWriteMask: 0,
      },
    });
    console.log(`[PMXDemo] mainPipeline valid=${this.mainPipeline !== null}`);

    const shadowGroup0 = this.device.createBindGroupLayout({
      label: "shadow-group0",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });

    this.shadowPipeline = this.device.createRenderPipeline({
      label: "pmx-shadow",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [shadowGroup0, this.skinBGL] }),
      vertex: { module: shadowVSModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      primitive: { topology: "triangle-list", cullMode: "none", frontFace: "cw" },
      depthStencil: { format: this.shadowMap.format, depthWriteEnabled: true, depthCompare: "less", depthBias: 2, depthBiasSlopeScale: 1.5 },
    });

    const outlineGroup0 = this.device.createBindGroupLayout({
      label: "outline-group0",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });

    this.outlinePipeline = this.device.createRenderPipeline({
      label: "pmx-outline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [outlineGroup0, this.shadowBGLayout, this.skinBGL] }),
      vertex: { module: outVSModule, entryPoint: "vs_main", buffers: [vertexLayout] },
      fragment: { module: outFSModule, entryPoint: "fs_main", targets: [{ format: HDR_FORMAT, blend: blendState }, { format: "rg8unorm" }] },
      primitive: { topology: "triangle-list", cullMode: "front", frontFace: "cw" },
      depthStencil: {
        format: DS_FORMAT, depthWriteEnabled: true, depthCompare: "less-equal", depthBias: 4, depthBiasSlopeScale: 1,
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff, stencilWriteMask: 0,
      },
    });
  }

  update(time: number, deltaTime: number): void {
    if (!this.loaded) return;

    if (this.animPlayer && !this.animPaused) {
      this.animPlayer.update(deltaTime);
      this.applyMorphDeform();
      this.skeleton!.updateWorldMatrices();
      if (this.ikChains.length > 0) {
        solveIK(this.skeleton!, this.ikChains);

      }
      this.skeleton!.computeSkinMatrices(this.skinning!.skinMatrixData);

      if (this.physicsEnabled && this.physics && this.skeleton) {
        this.physics.step(deltaTime, this.skeleton.worldMatrices, this.skeleton.inverseBindMatrices);
        this.skeleton!.computeSkinMatrices(this.skinning!.skinMatrixData);

      }
    }

    if (this.cameraAnimEnabled && !this.animPaused) {
      const anim = this.vmdCameraAnimation ?? this.orbitCameraAnimation;
      if (anim) {
        const animTime = this.animPlayer ? this.animPlayer.currentTime : 0;
        const pose = anim.sample(animTime);
        if (pose) {
          this.camera.setMode("vmd");
          this.camera.setVmdPose(pose);
        }
      }
    } else {
      this.camera.setMode("orbit");
      if (this.faceLockEnabled && this.skeleton && this._headBoneIdx >= 0) {
        this.camera.target = MmdCoord.worldPos(this.skeleton.worldMatrices, this._headBoneIdx);
      }
    }

    const w = this.ctx.canvas.width;
    const h = this.ctx.canvas.height;
    if (w !== this.hdrTarget.w || h !== this.hdrTarget.h) {
      this.hdrTarget.resize(w, h);

    }

    const viewProj = this.camera.getViewProjectionMatrix(w / h);
    const model = mat4.scaling(vec3.create(1, 1, -1));

    this.sceneData.set(viewProj as unknown as ArrayLike<number>, 0);
    this.sceneData.set(model as unknown as ArrayLike<number>, 16);
    const lx = this.lightX, ly = this.lightY, lz = this.lightZ;
    const len = Math.sqrt(lx * lx + ly * ly + lz * lz) || 1;
    this.sceneData[32] = lx / len; this.sceneData[33] = ly / len; this.sceneData[34] = lz / len; this.sceneData[35] = 0;
    this.sceneData[36] = 2.0; this.sceneData[37] = 2.0; this.sceneData[38] = 2.0; this.sceneData[39] = 0;
    this.sceneData[40] = this.camera.position[0]; this.sceneData[41] = this.camera.position[1]; this.sceneData[42] = this.camera.position[2]; this.sceneData[43] = 0;
    this.sceneDataU32[44] = this.gpuSkinningEnabled ? 1 : 0;

    this.device.queue.writeBuffer(this.sceneBuffer, 0, this.sceneData as unknown as GPUAllowSharedBufferSource);

    const slx = this.lightX, sly = this.lightY, slz = this.lightZ;
    const slen = Math.sqrt(slx * slx + sly * sly + slz * slz) || 1;
    const sdx = slx / slen, sdy = sly / slen, sdz = slz / slen;
    const shadowTarget = vec3.create(0, 11, 0);
    const shadowEye = vec3.create(shadowTarget[0] + sdx * 72, shadowTarget[1] + sdy * 72, shadowTarget[2] + sdz * 72);
    this.shadowMap.lightPosition = shadowEye;
    this.shadowMap.lightTarget = shadowTarget;
    this.shadowMap.updateLightVP();

    this.shadowSceneData.set(this.shadowMap.lightVP as unknown as ArrayLike<number>, 0);
    this.shadowSceneData.set(model as unknown as ArrayLike<number>, 16);
    this.shadowSceneDataU32[32] = this.gpuSkinningEnabled ? 1 : 0;
    this.device.queue.writeBuffer(this.shadowSceneBuffer, 0, this.shadowSceneData as unknown as GPUAllowSharedBufferSource);
  }

  createPasses(): RenderPass[] {
    return [{
      label: this.label,
      execute: (encoder: GPUCommandEncoder, view: GPUTextureView) => {
        if (!this.loaded) return;
        const w = this.ctx.canvas.width;
        const h = this.ctx.canvas.height;
        if (this.hdrTarget.w !== w || this.hdrTarget.h !== h) { this.hdrTarget.resize(w, h); }
        if (!this.bloomMaskTex || this.bloomMaskTex.width !== w || this.bloomMaskTex.height !== h) {
          this.bloomMaskTex?.destroy();
          this.bloomMaskTex = this.device.createTexture({ label: "bloom-mask", size: [w, h], format: "rg8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
          this.bloomMaskView = this.bloomMaskTex.createView();
        }

        if (this.skinning) {

          this.device.queue.writeBuffer(this.skinMatrixStorageBuffer, 0, this.skinning.skinMatrixData as unknown as GPUAllowSharedBufferSource);
        }

        let activeVB = this.vertexBuffer;

        if (this.gpuMorphEnabled && this.gpuMorph) {
          const allWeights = this.animPlayer?.getMorphWeights();
          if (allWeights && this.gpuMorphWeights) {
            const indices = this.gpuMorphIndices;
            const out = this.gpuMorphWeights;
            for (let i = 0; i < indices.length; i++) {
              out[i] = allWeights[indices[i]];
            }
            this.gpuMorph.updateWeights(out);
          }
          this.gpuMorph.dispatch(encoder);
          const morphVB = this.gpuMorph.getMorphedVertexBuffer();
          if (morphVB) {
            if (this.gpuSkinningEnabled && this.gpuSkinning) {
              this.gpuSkinning.setSourceBuffer(morphVB);
            } else {
              activeVB = morphVB;
            }
          }
        }

        if (this.gpuSkinningEnabled && this.gpuSkinning && this.skinning) {
          this.gpuSkinning.updateSkinMatrices(this.skinning.skinMatrixData);
          this.gpuSkinning.dispatch(encoder);
          const gpuVB = this.gpuSkinning.getSkinnedVertexBuffer();
          if (gpuVB) activeVB = gpuVB;
        }

        const shadowPass = this.shadowMap.beginShadowPass(encoder);
        shadowPass.setPipeline(this.shadowPipeline);
        shadowPass.setBindGroup(0, this.shadowSceneBG);
        shadowPass.setBindGroup(1, this.skinBG);
        shadowPass.setVertexBuffer(0, activeVB);
        shadowPass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");
        for (const mr of this.matRenders) {
          if (!mr.castsShadow) continue;
          shadowPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
        }
        shadowPass.end();

        const mainPass = encoder.beginRenderPass({
          colorAttachments: [
            { view: this.hdrTarget.colorTarget.view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
            { view: this.bloomMaskView!, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          ],
          depthStencilAttachment: {
            view: this.hdrTarget.depthTarget.view,
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
            stencilClearValue: 0,
            stencilLoadOp: "clear",
            stencilStoreOp: "store",
          },
        });
        mainPass.setVertexBuffer(0, activeVB);
        mainPass.setIndexBuffer(this.indexBuffer, this.use32bit ? "uint32" : "uint16");
        mainPass.setStencilReference(1);

        const curGen = this.shadowMap.generation;
        if (this.matRenders.length > 0 && this.matRenders[0].shadowGen !== curGen) {
          for (const mr of this.matRenders) {
            if (mr.shadowGen === curGen) continue;
            mr.shadowBG = this.device.createBindGroup({
              layout: this.shadowBGLayout,
              entries: [
                { binding: 0, resource: this.shadowMap.view },
                { binding: 1, resource: this.shadowMap.sampler },
                { binding: 2, resource: { buffer: this.shadowMap.getVPBuffer() } },
              ],
            });
            mr.shadowGen = curGen;
          }
        }

        for (const mr of this.opaqueOrder) {
          if (mr.hasEdge && mr.outlineBG && mr.renderClass !== "eye") {
            mainPass.setPipeline(this.outlinePipeline);
            mainPass.setBindGroup(0, mr.outlineBG);
            mainPass.setBindGroup(1, mr.shadowBG);
            mainPass.setBindGroup(2, this.skinBG);
            mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
          }
          const pipeline = this.stencilEnabled
            ? (mr.renderClass === "eye" ? this.eyePipeline
              : mr.renderClass === "hair" ? this.hairPipeline
              : this.mainPipeline)
            : this.mainPipeline;
          mainPass.setPipeline(pipeline);
          mainPass.setBindGroup(0, mr.mainBG);
          mainPass.setBindGroup(1, mr.shadowBG);
          mainPass.setBindGroup(2, this.skinBG);
          mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
        }

        for (const mr of this.matRenders) {
          if (!mr.isTransparent) continue;
          if (mr.hasEdge && mr.outlineBG) {
            mainPass.setPipeline(this.outlinePipeline);
            mainPass.setBindGroup(0, mr.outlineBG);
            mainPass.setBindGroup(1, mr.shadowBG);
            mainPass.setBindGroup(2, this.skinBG);
            mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
          }
          mainPass.setPipeline(this.mainPipeline);
          mainPass.setBindGroup(0, mr.mainBG);
          mainPass.setBindGroup(1, mr.shadowBG);
          mainPass.setBindGroup(2, this.skinBG);
          mainPass.drawIndexed(mr.indexCount, 1, mr.indexOffset);
        }
        mainPass.end();

        const hdrTex = this.hdrTarget.colorTarget.texture;
        const hdrView = this.hdrTarget.colorTarget.view;

        if (this.bloomEnabled) {
          const bloomResult = this.bloom.execute(encoder, hdrTex, this.bloomMaskView!);
          this.tonemapper.apply(encoder, view, this.ctx.format, hdrView, bloomResult.view, this.bloom.bloomIntensity);
        } else {
          this.tonemapper.apply(encoder, view, this.ctx.format, hdrView, null, 0);
        }

        if (this.debugIK && this.ikChains.length > 0 && this.skeleton) {
          this.ikDebugRenderer.draw(
            encoder, view, this.hdrTarget.depthTarget.view,
            this.camera.getViewProjectionMatrix(w / h),
            this.skeleton.worldMatrices, this.skeleton.parentIndices,
            this.skeleton.boneCount, this.ikChains, w, h,
          );
        }
      },
    }];
  }

  private ikDebugRenderer!: IKDebugRenderer;

  private setShadowResolution(size: number): void {
    if (size === this.shadowMap.size) return;
    this.shadowMap.resize(size);
    for (const mr of this.matRenders) {
      mr.shadowBG = this.device.createBindGroup({
        layout: this.shadowBGLayout,
        entries: [
          { binding: 0, resource: this.shadowMap.view },
          { binding: 1, resource: this.shadowMap.sampler },
          { binding: 2, resource: { buffer: this.shadowMap.getVPBuffer() } },
        ],
      });
    }
  }

  registerGUI(gui: any) {
    gui.add(this, "animPaused").name("Pause Animation");
    gui.add(this, "physicsEnabled").name("Physics");
    gui.add(this, "gpuSkinningEnabled").name("GPU Skinning");
    gui.add(this, "gpuMorphEnabled").name("GPU Morph");
    gui.add(this, "cameraAnimEnabled").name("Camera Animation");
    gui.add(this, "cameraAnimSource").name("Camera Source").disable(true);
    gui.add(this, "faceLockEnabled").name("Face Lock");
    gui.add(this, "debugIK").name("Debug Skeleton");
    gui.add(this, "bloomEnabled").name("Bloom");
    gui.add(this, "tonemapEnabled").name("Tone Mapping");
    gui.add(this, "gradeEnabled").name("Color Grading");
    gui.add(this, "stencilEnabled").name("Eye Stencil");
    gui.add(this, "shadowRes", [1024, 2048, 4096]).name("Shadow Res").onChange((v: number) => this.setShadowResolution(v));
    const camFolder = gui.addFolder("Camera");
    camFolder.add(this.camera, "distance", 1, 80, 0.5).name("Distance");
    camFolder.add(this.camera, "fov", 20, 120, 1).name("FOV");
    camFolder.add(this.camera.target, "0", -20, 20, 0.1).name("Target X");
    camFolder.add(this.camera.target, "1", -5, 30, 0.1).name("Target Y");
    camFolder.add(this.camera.target, "2", -20, 20, 0.1).name("Target Z");
    gui.add(this.bloom, "bloomIntensity", 0, 1, 0.01).name("Bloom Intensity");
    gui.add(this.bloom, "threshold", 0, 2, 0.01).name("Bloom Threshold");
    gui.add(this.bloom, "knee", 0, 1, 0.01).name("Bloom Knee");
    gui.add(this.bloom, "radius", 0.5, 10, 0.1).name("Bloom Radius");
    gui.add(this.bloom, "maxMips", [2, 3, 4, 5]).name("Bloom Mips");
    const toneFolder = gui.addFolder("Tone Mapping");
    toneFolder.add(this.tonemapper, "exposure", 0.1, 3, 0.01).name("Exposure");
    toneFolder.add(this.tonemapper, "gamma", 1.0, 3.0, 0.01).name("Gamma");
    const gradeFolder = gui.addFolder("Color Grading");
    gradeFolder.add(this.tonemapper, "slope", 0.5, 2.0, 0.01).name("Slope");
    gradeFolder.add(this.tonemapper, "offset", -0.5, 0.5, 0.01).name("Offset");
    gradeFolder.add(this.tonemapper, "power", 0.5, 2.0, 0.01).name("Power");
    gradeFolder.add(this.tonemapper, "saturation", 0, 2, 0.01).name("Saturation");
    gradeFolder.add(this.tonemapper, "contrast", 0.5, 2.0, 0.01).name("Contrast");
    const lightFolder = gui.addFolder("Light");
    lightFolder.add(this, "lightX", -30, 30, 0.5).name("X");
    lightFolder.add(this, "lightY", -30, 30, 0.5).name("Y");
    lightFolder.add(this, "lightZ", -30, 30, 0.5).name("Z");
  }

  private createOrbitCameraTrack(durationSec: number, centerY: number, radius: number): CameraAnimation {
    const frames: VMDCameraFrame[] = [];
    const totalFrames = Math.ceil(durationSec * 30);
    const orbitFrames = Math.min(totalFrames, 300);
    const dist = -(radius * 2.5);
    const cy = centerY;
    const fov = 30;
    for (let i = 0; i <= orbitFrames; i++) {
      const angle = (i / orbitFrames) * Math.PI * 2;
      const frame = Math.round((i / orbitFrames) * totalFrames);
      const ip = new Uint8Array(24);
      for (let j = 0; j < 24; j++) ip[j] = 20;
      frames.push({
        frame,
        distance: dist,
        target: [0, cy, 0],
        rotation: [0, angle, 0],
        fov,
        interpolation: ip,
      });
    }
    return new CameraAnimation(frames);
  }

  destroy(): void {
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.sceneBuffer?.destroy();
    this.shadowSceneBuffer?.destroy();
    this.skinMatrixStorageBuffer?.destroy();
    this.skinning?.destroy();
    this.shadowMap?.destroy();
    this.hdrTarget?.destroy();
    this.bloom?.destroy();

    this.bloomMaskTex?.destroy();
    this.bloomOutput?.destroy();
    this.tonemapper?.destroy();
    this.ikDebugRenderer?.destroy();
    this._depthTex?.destroy();
    this.gpuSkinning?.destroy();
    this.gpuMorph?.destroy();
    this.morphDeformer?.destroy();

    for (const t of this.gpuTextures) t.destroy();
    this.gpuTextures = [];
    this.matRenders = [];
  }

  private applyMorphDeform(): void {
    if (this.gpuMorphEnabled) return;
    const weights = this.animPlayer?.getMorphWeights();
    if (!weights || !this.morphDeformer) return;
    this.morphDeformer.apply(weights);
  }
}

