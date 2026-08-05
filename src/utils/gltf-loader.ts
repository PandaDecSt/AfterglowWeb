export interface MeshData {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint16Array | Uint32Array;
  materialIndex: number;
  name: string;
}

export interface MaterialData {
  name: string;
  baseColorFactor: [number, number, number, number];
  metallicFactor: number;
  roughnessFactor: number;
  baseColorImage: ImageBitmap | null;
  metallicRoughnessImage: ImageBitmap | null;
  normalImage: ImageBitmap | null;
  occlusionImage: ImageBitmap | null;
  occlusionStrength: number;
}

export interface LoadedModel {
  meshes: MeshData[];
  materials: MaterialData[];
  name: string;
}

interface GLTFAccessor {
  bufferView: number;
  componentType: number;
  count: number;
  type: string;
  byteOffset?: number;
}

interface GLTFBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
}

interface GLTFPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  mode?: number;
  material?: number;
}

interface GLTFMesh {
  name?: string;
  primitives: GLTFPrimitive[];
}

interface GLTFTextureInfo {
  index: number;
  texCoord?: number;
}

interface GLTFMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorFactor?: number[];
    baseColorTexture?: GLTFTextureInfo;
    metallicFactor?: number;
    roughnessFactor?: number;
    metallicRoughnessTexture?: GLTFTextureInfo;
  };
  normalTexture?: GLTFTextureInfo;
  occlusionTexture?: GLTFTextureInfo & { strength?: number };
}

interface GLTFTexture {
  source?: number;
  sampler?: number;
}

interface GLTFImage {
  bufferView?: number;
  mimeType?: string;
  uri?: string;
}

interface GLTFJson {
  accessors: GLTFAccessor[];
  bufferViews: GLTFBufferView[];
  buffers: { uri?: string; byteLength: number }[];
  meshes: GLTFMesh[];
  materials?: GLTFMaterial[];
  textures?: GLTFTexture[];
  images?: GLTFImage[];
  nodes?: {
    name?: string;
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
    children?: number[];
    mesh?: number;
    skin?: number;
  }[];
  scenes?: { name?: string; nodes?: number[] }[];
  scene?: number;
}

const COMPONENT_SIZES: Record<number, number> = {
  5120: 1,
  5121: 1,
  5122: 2,
  5123: 2,
  5125: 4,
  5126: 4,
};

const TYPE_COUNTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT4: 16,
};

// --- Minimal column-major 4x4 matrix helpers (no external dep) -----------
// glTF stores matrices column-major; a local node matrix is T * R * S.
type Mat4 = number[];

function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const o = new Array<number>(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[0 * 4 + r] * b[c * 4 + 0] +
        a[1 * 4 + r] * b[c * 4 + 1] +
        a[2 * 4 + r] * b[c * 4 + 2] +
        a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

function nodeLocalMatrix(node: {
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}): Mat4 {
  if (node.matrix) return node.matrix.slice();
  const t = node.translation ?? [0, 0, 0];
  const q = node.rotation ?? [0, 0, 0, 1];
  const s = node.scale ?? [1, 1, 1];
  const [x, y, z, w] = q;
  const x2 = x + x, y2 = y + y, z2 = z + z, w2 = w + w;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  const r00 = 1 - (yy + zz), r01 = xy - wz, r02 = xz + wy;
  const r10 = xy + wz, r11 = 1 - (xx + zz), r12 = yz - wx;
  const r20 = xz - wy, r21 = yz + wx, r22 = 1 - (xx + yy);
  // column-major: T * R * S  (each rotation column scaled by s)
  return [
    r00 * s[0], r10 * s[0], r20 * s[0], 0,
    r01 * s[1], r11 * s[1], r21 * s[1], 0,
    r02 * s[2], r12 * s[2], r22 * s[2], 0,
    t[0], t[1], t[2], 1,
  ];
}

function transformPoint(m: Mat4, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12],
    m[1] * x + m[5] * y + m[9] * z + m[13],
    m[2] * x + m[6] * y + m[10] * z + m[14],
  ];
}

// Upper 3x3 (no translation) for normals; not the strict inverse-transpose,
// but fine for viewing (character scales are near-uniform).
function transformDir(m: Mat4, x: number, y: number, z: number): [number, number, number] {
  return [
    m[0] * x + m[4] * y + m[8] * z,
    m[1] * x + m[5] * y + m[9] * z,
    m[2] * x + m[6] * y + m[10] * z,
  ];
}

export async function loadGLTF(url: string): Promise<LoadedModel> {
  const response = await fetch(url);

  let json: GLTFJson;
  const binBuffers: ArrayBuffer[] = [];

  if (url.endsWith(".glb")) {
    const glb = await response.arrayBuffer();
    const headerView = new DataView(glb);
    const jsonChunkLength = headerView.getUint32(12, true);
    const jsonBytes = new Uint8Array(glb, 20, jsonChunkLength);
    json = JSON.parse(new TextDecoder().decode(jsonBytes));

    const binChunkOffset = 20 + jsonChunkLength;
    if (binChunkOffset < glb.byteLength) {
      const binChunkLength = headerView.getUint32(binChunkOffset, true);
      binBuffers.push(glb.slice(binChunkOffset + 8, binChunkOffset + 8 + binChunkLength));
    }
  } else {
    json = await response.json();
    const baseDir = url.substring(0, url.lastIndexOf("/") + 1);
    for (const buf of json.buffers) {
      if (buf.uri) {
        const binUrl = buf.uri.startsWith("data:") ? buf.uri : baseDir + buf.uri;
        const binResp = await fetch(binUrl);
        binBuffers.push(await binResp.arrayBuffer());
      }
    }
  }

  function getAccessorData(accessorIndex: number) {
    const accessor = json.accessors[accessorIndex];
    const bufferView = json.bufferViews[accessor.bufferView];
    const buffer = binBuffers[bufferView.buffer];
    const byteOffset = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const componentSize = COMPONENT_SIZES[accessor.componentType];
    const numComponents = TYPE_COUNTS[accessor.type];
    const byteStride = bufferView.byteStride ?? componentSize * numComponents;
    return { data: buffer, componentType: accessor.componentType, count: accessor.count, type: accessor.type, byteOffset, byteStride };
  }

  function extractFloat32(accessorIndex: number, numComponents: number): Float32Array {
    const { data, byteOffset, count, byteStride } = getAccessorData(accessorIndex);
    const result = new Float32Array(count * numComponents);
    const view = new DataView(data);
    for (let i = 0; i < count; i++) {
      const offset = byteOffset + i * byteStride;
      for (let c = 0; c < numComponents; c++) {
        result[i * numComponents + c] = view.getFloat32(offset + c * 4, true);
      }
    }
    return result;
  }

  function extractIndices(accessorIndex: number): Uint16Array | Uint32Array {
    const { data, componentType, byteOffset, count, byteStride } = getAccessorData(accessorIndex);
    const view = new DataView(data);
    if (componentType === 5123) {
      const result = new Uint16Array(count);
      for (let i = 0; i < count; i++) result[i] = view.getUint16(byteOffset + i * byteStride, true);
      return result;
    }
    const result = new Uint32Array(count);
    for (let i = 0; i < count; i++) result[i] = view.getUint32(byteOffset + i * byteStride, true);
    return result;
  }

  async function loadImage(imageIndex: number): Promise<ImageBitmap | null> {
    const images = json.images;
    if (!images || imageIndex >= images.length) return null;
    const img = images[imageIndex];

    try {
      if (img.bufferView !== undefined) {
        const bv = json.bufferViews[img.bufferView];
        const buffer = binBuffers[bv.buffer];
        const bytes = new Uint8Array(buffer, bv.byteOffset ?? 0, bv.byteLength);
        const blob = new Blob([bytes], { type: img.mimeType ?? "image/png" });
        return await createImageBitmap(blob);
      } else if (img.uri) {
        const baseDir = url.substring(0, url.lastIndexOf("/") + 1);
        const imgResp = await fetch(baseDir + img.uri);
        const blob = await imgResp.blob();
        return await createImageBitmap(blob);
      }
    } catch (e) {
      console.warn(`[glTF] Failed to load image ${imageIndex}:`, e);
    }
    return null;
  }

  function getTextureImageIndex(texInfo: GLTFTextureInfo | undefined): number | null {
    if (!texInfo || !json.textures) return null;
    const tex = json.textures[texInfo.index];
    return tex?.source ?? null;
  }

  // Load materials
  const materials: MaterialData[] = [];
  if (json.materials) {
    for (const mat of json.materials) {
      const pbr = mat.pbrMetallicRoughness;
      const baseColorImgIdx = getTextureImageIndex(pbr?.baseColorTexture);
      const mrImgIdx = getTextureImageIndex(pbr?.metallicRoughnessTexture);
      const normalImgIdx = getTextureImageIndex(mat.normalTexture);
      const occImgIdx = getTextureImageIndex(mat.occlusionTexture);

      const [baseColorImage, metallicRoughnessImage, normalImage, occlusionImage] = await Promise.all([
        baseColorImgIdx !== null ? loadImage(baseColorImgIdx) : Promise.resolve(null),
        mrImgIdx !== null ? loadImage(mrImgIdx) : Promise.resolve(null),
        normalImgIdx !== null ? loadImage(normalImgIdx) : Promise.resolve(null),
        occImgIdx !== null ? loadImage(occImgIdx) : Promise.resolve(null),
      ]);

      materials.push({
        name: mat.name ?? "unnamed",
        baseColorFactor: (pbr?.baseColorFactor as [number, number, number, number]) ?? [1, 1, 1, 1],
        metallicFactor: pbr?.metallicFactor ?? 0.0,
        roughnessFactor: pbr?.roughnessFactor ?? 0.7,
        baseColorImage,
        metallicRoughnessImage,
        normalImage,
        occlusionImage,
        occlusionStrength: mat.occlusionTexture?.strength ?? 1.0,
      });
    }
  }

  // Load meshes — walk the node hierarchy and bake each primitive's local
  // vertices into world space via the node's world matrix. This is required
  // for rigged/hierarchical models (e.g. character GLBs) whose primitives are
  // stored in node-local space; without it the body parts collapse to origin.
  // Models without nodes (or the trivial-identity case) fall back to the old
  // direct-json.meshes behaviour, so existing models are unaffected.
  const meshes: MeshData[] = [];

  const nodes: any[] = json.nodes ?? [];
  if (nodes.length > 0) {
    const childrenOf: number[][] = nodes.map(() => []);
    const isChild = new Array<boolean>(nodes.length).fill(false);
    nodes.forEach((n, i) => (n.children ?? []).forEach((c: number) => { childrenOf[i].push(c); isChild[c] = true; }));
    const sceneNodes = json.scenes && json.scenes[json.scene ?? 0] ? json.scenes[json.scene ?? 0].nodes : null;
    const roots = sceneNodes ?? nodes.map((_, i) => i).filter((i) => !isChild[i]);

    const worldMat: Mat4[] = new Array(nodes.length);
    const dfs = (i: number, parentW: Mat4 | null) => {
      const w = parentW ? mat4Multiply(parentW, nodeLocalMatrix(nodes[i])) : nodeLocalMatrix(nodes[i]);
      worldMat[i] = w;
      for (const c of childrenOf[i]) dfs(c, w);
    };
    for (const r of roots) dfs(r, null);

    const bakePrimitive = (mesh: any, prim: any, world: Mat4) => {
      if (prim.mode !== undefined && prim.mode !== 4) return;
      const positions = extractFloat32(prim.attributes["POSITION"], 3);
      const hasNormal = prim.attributes["NORMAL"] !== undefined;
      const normals = hasNormal ? extractFloat32(prim.attributes["NORMAL"], 3) : new Float32Array(positions.length);
      const uvs = prim.attributes["TEXCOORD_0"] !== undefined
        ? extractFloat32(prim.attributes["TEXCOORD_0"], 2)
        : new Float32Array((positions.length / 3) * 2);
      const indices = prim.indices !== undefined
        ? extractIndices(prim.indices)
        : createSequentialIndices(positions.length / 3);

      const bp = new Float32Array(positions.length);
      const bn = new Float32Array(normals.length);
      const vCount = positions.length / 3;
      for (let v = 0; v < vCount; v++) {
        const p = transformPoint(world, positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
        bp[v * 3] = p[0]; bp[v * 3 + 1] = p[1]; bp[v * 3 + 2] = p[2];
        if (hasNormal) {
          const n = transformDir(world, normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2]);
          const len = Math.hypot(n[0], n[1], n[2]) || 1;
          bn[v * 3] = n[0] / len; bn[v * 3 + 1] = n[1] / len; bn[v * 3 + 2] = n[2] / len;
        }
      }
      meshes.push({ positions: bp, normals: bn, uvs, indices, materialIndex: prim.material ?? 0, name: mesh.name ?? `mesh_${meshes.length}` });
    };

    nodes.forEach((n, i) => {
      if (n.mesh == null) return;
      const world = worldMat[i] ?? nodeLocalMatrix(n);
      const gm = json.meshes[n.mesh];
      if (!gm) return;
      for (const prim of gm.primitives) bakePrimitive(gm, prim, world);
    });
  } else {
    // No nodes: original direct-mesh behaviour.
    for (const mesh of json.meshes) {
      for (const prim of mesh.primitives) {
        if (prim.mode !== undefined && prim.mode !== 4) continue;
        const positions = extractFloat32(prim.attributes["POSITION"], 3);
        const normals = prim.attributes["NORMAL"] !== undefined
          ? extractFloat32(prim.attributes["NORMAL"], 3)
          : new Float32Array(positions.length);
        const uvs = prim.attributes["TEXCOORD_0"] !== undefined
          ? extractFloat32(prim.attributes["TEXCOORD_0"], 2)
          : new Float32Array((positions.length / 3) * 2);
        const indices = prim.indices !== undefined
          ? extractIndices(prim.indices)
          : createSequentialIndices(positions.length / 3);
        meshes.push({ positions, normals, uvs, indices, materialIndex: prim.material ?? 0, name: mesh.name ?? `mesh_${meshes.length}` });
      }
    }
  }

  return { meshes, materials, name: url.split("/").pop() ?? "model" };
}

function createSequentialIndices(count: number): Uint32Array {
  const indices = new Uint32Array(count);
  for (let i = 0; i < count; i++) indices[i] = i;
  return indices;
}

export function interleaveMesh(mesh: MeshData): {
  vertices: Float32Array;
  indices: Uint16Array | Uint32Array;
} {
  const vertexCount = mesh.positions.length / 3;
  const vertices = new Float32Array(vertexCount * 8);
  for (let i = 0; i < vertexCount; i++) {
    const base = i * 8;
    vertices[base + 0] = mesh.positions[i * 3 + 0];
    vertices[base + 1] = mesh.positions[i * 3 + 1];
    vertices[base + 2] = mesh.positions[i * 3 + 2];
    vertices[base + 3] = mesh.normals[i * 3 + 0];
    vertices[base + 4] = mesh.normals[i * 3 + 1];
    vertices[base + 5] = mesh.normals[i * 3 + 2];
    vertices[base + 6] = mesh.uvs[i * 2 + 0];
    vertices[base + 7] = mesh.uvs[i * 2 + 1];
  }
  return { vertices, indices: mesh.indices };
}
