export function createCubeGeometry(): {
  vertices: Float32Array;
  indices: Uint16Array;
} {
  // pos(3) + normal(3) + uv(2) = 8 floats per vertex
  const vertices = new Float32Array([
    // front
    -0.5, -0.5,  0.5,  0, 0, 1,  0, 0,
     0.5, -0.5,  0.5,  0, 0, 1,  1, 0,
     0.5,  0.5,  0.5,  0, 0, 1,  1, 1,
    -0.5,  0.5,  0.5,  0, 0, 1,  0, 1,
    // back
    -0.5, -0.5, -0.5,  0, 0, -1,  1, 0,
     0.5, -0.5, -0.5,  0, 0, -1,  0, 0,
     0.5,  0.5, -0.5,  0, 0, -1,  0, 1,
    -0.5,  0.5, -0.5,  0, 0, -1,  1, 1,
    // top
    -0.5,  0.5, -0.5,  0, 1, 0,  0, 0,
     0.5,  0.5, -0.5,  0, 1, 0,  1, 0,
     0.5,  0.5,  0.5,  0, 1, 0,  1, 1,
    -0.5,  0.5,  0.5,  0, 1, 0,  0, 1,
    // bottom
    -0.5, -0.5, -0.5,  0, -1, 0,  0, 0,
     0.5, -0.5, -0.5,  0, -1, 0,  1, 0,
     0.5, -0.5,  0.5,  0, -1, 0,  1, 1,
    -0.5, -0.5,  0.5,  0, -1, 0,  0, 1,
    // right
     0.5, -0.5, -0.5,  1, 0, 0,  0, 0,
     0.5,  0.5, -0.5,  1, 0, 0,  1, 0,
     0.5,  0.5,  0.5,  1, 0, 0,  1, 1,
     0.5, -0.5,  0.5,  1, 0, 0,  0, 1,
    // left
    -0.5, -0.5, -0.5,  -1, 0, 0,  1, 0,
    -0.5,  0.5, -0.5,  -1, 0, 0,  0, 0,
    -0.5,  0.5,  0.5,  -1, 0, 0,  0, 1,
    -0.5, -0.5,  0.5,  -1, 0, 0,  1, 1,
  ]);

  const indices = new Uint16Array([
    0, 1, 2, 0, 2, 3,
    4, 6, 5, 4, 7, 6,
    8, 10, 9, 8, 11, 10,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 22, 21, 20, 23, 22,
  ]);

  return { vertices, indices };
}

export function createGridGeometry(size = 10, divisions = 10): {
  vertices: Float32Array;
  indices: Uint16Array;
} {
  const step = size / divisions;
  const half = size / 2;
  const vertsPerRow = divisions + 1;
  const vertices = new Float32Array(vertsPerRow * vertsPerRow * 8);

  let offset = 0;
  for (let z = 0; z <= divisions; z++) {
    for (let x = 0; x <= divisions; x++) {
      vertices[offset++] = -half + x * step;
      vertices[offset++] = 0;
      vertices[offset++] = -half + z * step;
      vertices[offset++] = 0;
      vertices[offset++] = 1;
      vertices[offset++] = 0;
      vertices[offset++] = x / divisions;
      vertices[offset++] = z / divisions;
    }
  }

  const indices = new Uint16Array(divisions * divisions * 6);
  let idx = 0;
  for (let z = 0; z < divisions; z++) {
    for (let x = 0; x < divisions; x++) {
      const a = z * vertsPerRow + x;
      const b = a + 1;
      const c = a + vertsPerRow;
      const d = c + 1;
      indices[idx++] = a;
      indices[idx++] = c;
      indices[idx++] = b;
      indices[idx++] = b;
      indices[idx++] = c;
      indices[idx++] = d;
    }
  }

  return { vertices, indices };
}

export function createSphereGeometry(
  radius = 1.0,
  widthSegments = 32,
  heightSegments = 16,
): { vertices: Float32Array; indices: Uint16Array } {
  const vertices: number[] = [];
  const indices: number[] = [];

  for (let y = 0; y <= heightSegments; y++) {
    const v = y / heightSegments;
    const phi = v * Math.PI;
    const sinPhi = Math.sin(phi);
    const cosPhi = Math.cos(phi);

    for (let x = 0; x <= widthSegments; x++) {
      const u = x / widthSegments;
      const theta = u * Math.PI * 2;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      const nx = sinPhi * cosTheta;
      const ny = cosPhi;
      const nz = sinPhi * sinTheta;

      vertices.push(nx * radius, ny * radius, nz * radius);
      vertices.push(nx, ny, nz);
      vertices.push(u, v);
    }
  }

  for (let y = 0; y < heightSegments; y++) {
    for (let x = 0; x < widthSegments; x++) {
      const a = y * (widthSegments + 1) + x;
      const b = a + 1;
      const c = a + widthSegments + 1;
      const d = c + 1;
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }

  return {
    vertices: new Float32Array(vertices),
    indices: new Uint16Array(indices),
  };
}
