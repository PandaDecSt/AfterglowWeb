export interface AABB {
  minX: number; minY: number; minZ: number;
  maxX: number; maxY: number; maxZ: number;
}

export function nonshearTransformAABB(aabb: AABB, m: ArrayLike<number>): AABB {
  const cx = (aabb.minX + aabb.maxX) * 0.5;
  const cy = (aabb.minY + aabb.maxY) * 0.5;
  const cz = (aabb.minZ + aabb.maxZ) * 0.5;
  const hx = (aabb.maxX - aabb.minX) * 0.5;
  const hy = (aabb.maxY - aabb.minY) * 0.5;
  const hz = (aabb.maxZ - aabb.minZ) * 0.5;

  // Column-major mat4: m[col*4+row]
  const m00 = m[0], m01 = m[4], m02 = m[8];
  const m10 = m[1], m11 = m[5], m12 = m[9];
  const m20 = m[2], m21 = m[6], m22 = m[10];
  const tx = m[12], ty = m[13], tz = m[14];

  const newCx = m00 * cx + m01 * cy + m02 * cz + tx;
  const newCy = m10 * cx + m11 * cy + m12 * cz + ty;
  const newCz = m20 * cx + m21 * cy + m22 * cz + tz;

  const newHx = Math.abs(m00) * hx + Math.abs(m01) * hy + Math.abs(m02) * hz;
  const newHy = Math.abs(m10) * hx + Math.abs(m11) * hy + Math.abs(m12) * hz;
  const newHz = Math.abs(m20) * hx + Math.abs(m21) * hy + Math.abs(m22) * hz;

  return {
    minX: newCx - newHx, minY: newCy - newHy, minZ: newCz - newHz,
    maxX: newCx + newHx, maxY: newCy + newHy, maxZ: newCz + newHz,
  };
}

export function frustumCullAABB(planes: Float32Array, aabb: AABB): boolean {
  for (let i = 0; i < 6; i++) {
    const nx = planes[i * 4 + 0];
    const ny = planes[i * 4 + 1];
    const nz = planes[i * 4 + 2];
    const d = planes[i * 4 + 3];

    const px = nx > 0 ? aabb.maxX : aabb.minX;
    const py = ny > 0 ? aabb.maxY : aabb.minY;
    const pz = nz > 0 ? aabb.maxZ : aabb.minZ;

    if (nx * px + ny * py + nz * pz + d < 0) {
      return false;
    }
  }
  return true;
}
