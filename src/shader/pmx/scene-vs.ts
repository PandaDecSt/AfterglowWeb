export const SCENE_VS = `
struct Scene {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  cameraPos: vec4<f32>,
  flags: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};
@group(0) @binding(0) var<uniform> scene: Scene;

@group(2) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
};

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

fn safe_normal(n: vec3<f32>) -> vec3<f32> {
  let l2 = dot(n, n);
  if (l2 < 1e-12) { return vec3<f32>(0.0, 1.0, 0.0); }
  return n * inverseSqrt(l2);
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
  var out: VSOut;

  var skinPos: vec4<f32>;
  var skinNrm: vec4<f32>;

  if ((scene.flags & 1u) != 0u) {
    skinPos = vec4<f32>(in.position, 1.0);
    skinNrm = vec4<f32>(in.normal, 0.0);
  } else {
    let weightSum = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    let invW = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
    let w = select(vec4<f32>(1.0, 0.0, 0.0, 0.0), in.weights * invW, weightSum > 0.0001);

    skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    skinNrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let pos4 = vec4<f32>(in.position, 1.0);
    let nrm4 = vec4<f32>(in.normal, 0.0);
    for (var i = 0u; i < 4u; i++) {
      let j = in.joints[i];
      skinPos += skinMatrices[j] * pos4 * w[i];
      skinNrm += skinMatrices[j] * nrm4 * w[i];
    }
  }

  let worldPos = (scene.model * skinPos).xyz;
  out.position = scene.viewProj * vec4<f32>(worldPos, 1.0);
  out.worldNormal = safe_normal((scene.model * skinNrm).xyz);
  out.worldPos = worldPos;
  out.uv = in.uv;
  return out;
}
`;