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
