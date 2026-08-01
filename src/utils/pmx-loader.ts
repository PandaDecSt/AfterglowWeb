export interface PMXVertex {
  position: Float32Array;
  normal: Float32Array;
  uv: Float32Array;
  additionalVec4: Float32Array[];
  boneType: number;
  boneIndices: Int32Array;
  boneWeights: Float32Array;
  edgeScale: number;
}

export interface PMXMaterial {
  name: string;
  nameEn: string;
  diffuse: Float32Array;
  specular: Float32Array;
  specularPower: number;
  ambient: Float32Array;
  flag: number;
  edgeColor: Float32Array;
  edgeScale: number;
  textureIndex: number;
  sphereTextureIndex: number;
  sphereMode: number;
  toonSharing: number;
  toonTextureIndex: number;
  comment: string;
  faceCount: number;
}

export interface PMXBone {
  name: string;
  nameEn: string;
  parentIndex: number;
  transformLevel: number;
  position: Float32Array;
  flag: number;
  ikTargetIndex: number;
  ikLoopCount: number;
  ikUnitLength: number;
  ikLinks: { linkIndex: number; hasLimit: boolean; limitMin: Float32Array; limitMax: Float32Array }[];
  appendParentIndex: number;
  appendRatio: number;
  appendRotate: boolean;
  appendMove: boolean;
}

export interface PMXMorph {
  name: string;
  nameEn: string;
  type: number;
  panel: number;
  offsets: { vertexIndex: number; position: Float32Array }[];
}

export interface PMXTexture {
  path: string;
}

export interface PMXRigidbody {
  name: string;
  nameEn: string;
  boneIndex: number;
  group: number;
  collisionMask: number;
  shape: number;
  size: Float32Array;
  position: Float32Array;
  rotation: Float32Array;
  mass: number;
  linearDamping: number;
  angularDamping: number;
  restitution: number;
  friction: number;
  type: number;
}

export interface PMXJoint {
  name: string;
  nameEn: string;
  type: number;
  rigidbodyIndexA: number;
  rigidbodyIndexB: number;
  position: Float32Array;
  rotation: Float32Array;
  positionMin: Float32Array;
  positionMax: Float32Array;
  rotationMin: Float32Array;
  rotationMax: Float32Array;
  springPosition: Float32Array;
  springRotation: Float32Array;
}

export interface PMXModel {
  name: string;
  nameEn: string;
  comment: string;
  commentEn: string;
  vertices: PMXVertex[];
  indices: Int32Array;
  materials: PMXMaterial[];
  bones: PMXBone[];
  morphs: PMXMorph[];
  textures: PMXTexture[];
  rigidbodies: PMXRigidbody[];
  joints: PMXJoint[];
}

class PMXReader {
  private view: DataView;
  private offset = 0;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  get remaining(): number {
    return this.view.byteLength - this.offset;
  }

  readUint8(): number {
    return this.view.getUint8(this.offset++);
  }

  readInt8(): number {
    return this.view.getInt8(this.offset++);
  }

  readInt16(): number {
    const v = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return v;
  }

  readUint16(): number {
    const v = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return v;
  }

  readInt32(): number {
    const v = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return v;
  }

  readFloat32(): number {
    const v = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return v;
  }

  readVec3(): Float32Array {
    const v = new Float32Array(3);
    v[0] = this.readFloat32();
    v[1] = this.readFloat32();
    v[2] = this.readFloat32();
    return v;
  }

  readVec4(): Float32Array {
    const v = new Float32Array(4);
    v[0] = this.readFloat32();
    v[1] = this.readFloat32();
    v[2] = this.readFloat32();
    v[3] = this.readFloat32();
    return v;
  }

  readText(encoding: number): string {
    const len = this.readInt32();
    if (len <= 0) return "";
    const bytes = new Uint8Array(this.view.buffer, this.view.byteOffset + this.offset, len);
    this.offset += len;
    if (encoding === 0) {
      return new TextDecoder("utf-16le").decode(bytes);
    }
    return new TextDecoder("utf-8").decode(bytes);
  }

  readVertexIndex(size: number): number {
    switch (size) {
      case 1: return this.readUint8();
      case 2: return this.readUint16();
      case 4: return this.readInt32();
      default: return this.readUint8();
    }
  }

  readNonVertexIndex(size: number): number {
    switch (size) {
      case 1: return this.readInt8();
      case 2: return this.readInt16();
      case 4: return this.readInt32();
      default: return this.readInt8();
    }
  }
}

export function parsePMX(buffer: ArrayBuffer): PMXModel {
  const r = new PMXReader(buffer);

  const magic = String.fromCharCode(r.readUint8(), r.readUint8(), r.readUint8(), r.readUint8());
  if (magic !== "PMX ") {
    throw new Error(`[PMX] Invalid magic: ${magic}`);
  }

  const version = r.readFloat32();
  if (version < 2.0 || version > 2.2) {
    throw new Error(`[PMX] Unsupported version: ${version}`);
  }

  const globalsCount = r.readUint8();
  const globals = new Uint8Array(globalsCount);
  for (let i = 0; i < globalsCount; i++) globals[i] = r.readUint8();

  const encoding = globals[0];
  const additionalVec4Count = globals[1];
  const vertexIndexSize = globals[2];
  const textureIndexSize = globals[3];
  const materialIndexSize = globals[4];
  const boneIndexSize = globals[5];
  const morphIndexSize = globals[6];
  const rigidbodyIndexSize = globals[7];

  const name = r.readText(encoding);
  const nameEn = r.readText(encoding);
  const comment = r.readText(encoding);
  const commentEn = r.readText(encoding);

  // --- Vertices ---
  // PMX boneType encoding (per reze-engine reference):
  //   0 = BDEF1 (1 bone)
  //   1 = BDEF2 (2 bones, 1 weight)
  //   2 = BDEF4 (4 bones, 4 weights stored in file)
  //   3 = SDEF  (2 bones, 1 weight + 3 vec3 SDEF data)
  //   4 = QDEF  (4 bones, 4 weights, quaternion deformation)
  const vertexCount = r.readInt32();
  const vertices: PMXVertex[] = [];

  for (let i = 0; i < vertexCount; i++) {
    const position = r.readVec3();
    const normal = r.readVec3();
    const uv = new Float32Array([r.readFloat32(), r.readFloat32()]);

    const additionalVec4: Float32Array[] = [];
    for (let j = 0; j < additionalVec4Count; j++) {
      additionalVec4.push(r.readVec4());
    }

    const boneType = r.readUint8();
    let boneIndices: Int32Array;
    let boneWeights: Float32Array;

    switch (boneType) {
      case 0: {
        boneIndices = new Int32Array([r.readNonVertexIndex(boneIndexSize)]);
        boneWeights = new Float32Array([1.0]);
        break;
      }
      case 1: {
        const b0 = r.readNonVertexIndex(boneIndexSize);
        const b1 = r.readNonVertexIndex(boneIndexSize);
        const w0 = r.readFloat32();
        boneIndices = new Int32Array([b0, b1]);
        boneWeights = new Float32Array([w0, 1 - w0]);
        break;
      }
      case 3: {
        const b0 = r.readNonVertexIndex(boneIndexSize);
        const b1 = r.readNonVertexIndex(boneIndexSize);
        const w0 = r.readFloat32();
        r.readFloat32(); r.readFloat32(); r.readFloat32();
        r.readFloat32(); r.readFloat32(); r.readFloat32();
        r.readFloat32(); r.readFloat32(); r.readFloat32();
        boneIndices = new Int32Array([b0, b1]);
        boneWeights = new Float32Array([w0, 1 - w0]);
        break;
      }
      case 2:
      case 4: {
        const b0 = r.readNonVertexIndex(boneIndexSize);
        const b1 = r.readNonVertexIndex(boneIndexSize);
        const b2 = r.readNonVertexIndex(boneIndexSize);
        const b3 = r.readNonVertexIndex(boneIndexSize);
        const w0 = r.readFloat32();
        const w1 = r.readFloat32();
        const w2 = r.readFloat32();
        const w3 = r.readFloat32();
        const sum = w0 + w1 + w2 + w3;
        boneIndices = new Int32Array([b0, b1, b2, b3]);
        if (sum > 0) {
          boneWeights = new Float32Array([w0 / sum, w1 / sum, w2 / sum, w3 / sum]);
        } else {
          boneWeights = new Float32Array([1, 0, 0, 0]);
        }
        break;
      }
      default: {
        boneIndices = new Int32Array([0]);
        boneWeights = new Float32Array([1.0]);
      }
    }

    const edgeScale = r.readFloat32();

    vertices.push({
      position, normal, uv, additionalVec4,
      boneType, boneIndices, boneWeights, edgeScale,
    });
  }

  // --- Face indices ---
  const faceCount = r.readInt32();
  const indices = new Int32Array(faceCount);
  for (let i = 0; i < faceCount; i++) {
    indices[i] = r.readVertexIndex(vertexIndexSize);
  }

  // --- Textures ---
  const textureCount = r.readInt32();
  const textures: PMXTexture[] = [];
  for (let i = 0; i < textureCount; i++) {
    textures.push({ path: r.readText(encoding) });
  }

  // --- Materials ---
  // Per reze-engine: comment (1 text field), then faceCount (int32)
  // toonSharing: 1 = shared toon (1 byte 0-9), 0 = individual toon (textureIndexSize bytes)
  const materialCount = r.readInt32();
  const materials: PMXMaterial[] = [];
  for (let i = 0; i < materialCount; i++) {
    const matName = r.readText(encoding);
    const matNameEn = r.readText(encoding);
    const diffuse = r.readVec4();
    const specular = r.readVec3();
    const specularPower = r.readFloat32();
    const ambient = r.readVec3();
    const flag = r.readUint8();
    const edgeColor = r.readVec4();
    const edgeScale = r.readFloat32();
    const textureIndex = r.readNonVertexIndex(textureIndexSize);
    const sphereTextureIndex = r.readNonVertexIndex(textureIndexSize);
    const sphereMode = r.readUint8();
    const toonSharing = r.readUint8();
    const toonTextureIndex = toonSharing === 1
      ? r.readUint8()
      : r.readNonVertexIndex(textureIndexSize);
    const matComment = r.readText(encoding);
    const matFaceCount = r.readInt32();

    materials.push({
      name: matName, nameEn: matNameEn,
      diffuse, specular, specularPower, ambient,
      flag, edgeColor, edgeScale,
      textureIndex, sphereTextureIndex, sphereMode,
      toonSharing, toonTextureIndex,
      comment: matComment,
      faceCount: matFaceCount,
    });
  }

  // --- Bones ---
  // Per reze-engine PMX 2.x bone flags:
  //   0x0001 = tail is bone index (else tail is vec3 offset)
  //   0x0020 = has IK
  //   0x0100 = append rotate (has append parent + ratio)
  //   0x0200 = append move
  //   0x0400 = axis limit (vec3)
  //   0x0800 = local axis (2 vec3: x-axis + z-axis)
  //   0x2000 = external parent (int32)
  const boneCount = r.readInt32();
  const bones: PMXBone[] = [];
  for (let i = 0; i < boneCount; i++) {
    const boneName = r.readText(encoding);
    const boneNameEn = r.readText(encoding);
    const position = r.readVec3();
    const parentIndex = r.readNonVertexIndex(boneIndexSize);
    const transformLevel = r.readInt32();
    const boneFlag = r.readUint16();

    if (boneFlag & 0x0001) {
      r.readNonVertexIndex(boneIndexSize);
    } else {
      r.readVec3();
    }

    let appendParentIndex = -1;
    let appendRatio = 0;
    let appendRotate = false;
    let appendMove = false;
    if (boneFlag & 0x0100 || boneFlag & 0x0200) {
      appendParentIndex = r.readNonVertexIndex(boneIndexSize);
      appendRatio = r.readFloat32();
      appendRotate = (boneFlag & 0x0100) !== 0;
      appendMove = (boneFlag & 0x0200) !== 0;
    }

    if (boneFlag & 0x0400) {
      r.readVec3();
    }

    if (boneFlag & 0x0800) {
      r.readVec3();
      r.readVec3();
    }

    if (boneFlag & 0x2000) {
      r.readInt32();
    }

    let ikTargetIndex = -1;
    let ikLoopCount = 0;
    let ikUnitLength = 0;
    const ikLinks: PMXBone["ikLinks"] = [];

    if (boneFlag & 0x0020) {
      ikTargetIndex = r.readNonVertexIndex(boneIndexSize);
      ikLoopCount = r.readInt32();
      ikUnitLength = r.readFloat32();
      const linkCount = r.readInt32();
      for (let j = 0; j < linkCount; j++) {
        const linkIndex = r.readNonVertexIndex(boneIndexSize);
        const hasLimit = r.readUint8() === 1;
        const limitMin = hasLimit ? r.readVec3() : new Float32Array(3);
        const limitMax = hasLimit ? r.readVec3() : new Float32Array(3);
        ikLinks.push({ linkIndex, hasLimit, limitMin, limitMax });
      }
    }

    bones.push({
      name: boneName, nameEn: boneNameEn,
      parentIndex, transformLevel, position,
      flag: boneFlag,
      ikTargetIndex, ikLoopCount, ikUnitLength, ikLinks,
      appendParentIndex, appendRatio, appendRotate, appendMove,
    });
  }

  // --- Morphs ---
  const morphCount = r.readInt32();
  const morphs: PMXMorph[] = [];
  for (let i = 0; i < morphCount; i++) {
    const morphName = r.readText(encoding);
    const morphNameEn = r.readText(encoding);
    const morphPanel = r.readUint8();
    const morphType = r.readUint8();
    const offsetCount = r.readInt32();

    const offsets: PMXMorph["offsets"] = [];
    if (morphType === 1) {
      for (let j = 0; j < offsetCount; j++) {
        const vertexIndex = r.readVertexIndex(vertexIndexSize);
        const position = r.readVec3();
        offsets.push({ vertexIndex, position });
      }
    } else if (morphType === 0) {
      for (let j = 0; j < offsetCount; j++) {
        r.readNonVertexIndex(morphIndexSize);
        r.readFloat32();
      }
    } else {
      for (let j = 0; j < offsetCount; j++) {
        switch (morphType) {
          case 2: r.readNonVertexIndex(boneIndexSize); r.readFloat32(); r.readFloat32(); r.readFloat32(); r.readFloat32(); r.readFloat32(); r.readFloat32(); r.readFloat32(); break;
          case 3: case 4: case 5: case 6: case 7: r.readVertexIndex(vertexIndexSize); r.readVec4(); break;
          case 8: {
            r.readNonVertexIndex(materialIndexSize); r.readUint8();
            for (let k = 0; k < 28; k++) r.readFloat32();
            break;
          }
          case 9: r.readNonVertexIndex(morphIndexSize); r.readFloat32(); break;
          case 10: r.readNonVertexIndex(rigidbodyIndexSize); r.readUint8(); r.readVec3(); r.readVec3(); break;
          default: break;
        }
      }
    }

    morphs.push({
      name: morphName, nameEn: morphNameEn,
      type: morphType, panel: morphPanel, offsets,
    });
  }

  // --- Display frames (skip) ---
  if (r.remaining > 4) {
    const frameCount = r.readInt32();
    for (let i = 0; i < frameCount; i++) {
      r.readText(encoding);
      r.readText(encoding);
      r.readUint8();
      const elemCount = r.readInt32();
      for (let j = 0; j < elemCount; j++) {
        const elemType = r.readUint8();
        elemType === 0 ? r.readNonVertexIndex(boneIndexSize) : r.readNonVertexIndex(morphIndexSize);
      }
    }
  }

  // --- Rigid bodies ---
  const rigidbodies: PMXRigidbody[] = [];
  if (r.remaining > 4) {
    const rbCount = r.readInt32();
    for (let i = 0; i < rbCount; i++) {
      const rbName = r.readText(encoding);
      const rbNameEn = r.readText(encoding);
      const rbBoneIndex = r.readNonVertexIndex(boneIndexSize);
      const rbGroup = r.readUint8();
      const rbCollisionMask = r.readUint16();
      const rbShape = r.readUint8();
      const rbSize = r.readVec3();
      const rbPosition = r.readVec3();
      const rbRotation = r.readVec3();
      const rbMass = r.readFloat32();
      const rbLinearDamping = r.readFloat32();
      const rbAngularDamping = r.readFloat32();
      const rbRestitution = r.readFloat32();
      const rbFriction = r.readFloat32();
      const rbType = r.readUint8();
      rigidbodies.push({
        name: rbName, nameEn: rbNameEn,
        boneIndex: rbBoneIndex, group: rbGroup, collisionMask: rbCollisionMask,
        shape: rbShape, size: rbSize, position: rbPosition, rotation: rbRotation,
        mass: rbMass, linearDamping: rbLinearDamping, angularDamping: rbAngularDamping,
        restitution: rbRestitution, friction: rbFriction, type: rbType,
      });
    }
  }

  // --- Joints ---
  const joints: PMXJoint[] = [];
  if (r.remaining > 4) {
    const jointCount = r.readInt32();
    for (let i = 0; i < jointCount; i++) {
      const jName = r.readText(encoding);
      const jNameEn = r.readText(encoding);
      const jType = r.readUint8();
      const jRbA = r.readNonVertexIndex(rigidbodyIndexSize);
      const jRbB = r.readNonVertexIndex(rigidbodyIndexSize);
      const jPos = r.readVec3();
      const jRot = r.readVec3();
      const jPosMin = r.readVec3();
      const jPosMax = r.readVec3();
      const jRotMin = r.readVec3();
      const jRotMax = r.readVec3();
      const jSpringPos = r.readVec3();
      const jSpringRot = r.readVec3();
      joints.push({
        name: jName, nameEn: jNameEn, type: jType,
        rigidbodyIndexA: jRbA, rigidbodyIndexB: jRbB,
        position: jPos, rotation: jRot,
        positionMin: jPosMin, positionMax: jPosMax,
        rotationMin: jRotMin, rotationMax: jRotMax,
        springPosition: jSpringPos, springRotation: jSpringRot,
      });
    }
  }

  return { name, nameEn, comment, commentEn, vertices, indices, materials, bones, morphs, textures, rigidbodies, joints };
}

export async function loadPMX(url: string): Promise<PMXModel> {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  return parsePMX(buffer);
}
