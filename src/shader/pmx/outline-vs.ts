export const OUTLINE_VS = `
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

struct OutlineMat {
  edgeColor: vec4<f32>,
  edgeSize: f32,
  _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(1) var<uniform> omat: OutlineMat;

@group(2) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
) -> VSOut {
  var out: VSOut;

  var skinPos: vec4<f32>;
  var skinNrm: vec4<f32>;

  if ((scene.flags & 1u) != 0u) {
    skinPos = vec4<f32>(position, 1.0);
    skinNrm = vec4<f32>(normal, 0.0);
  } else {
    skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    skinNrm = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let pos4 = vec4<f32>(position, 1.0);
    let nrm4 = vec4<f32>(normal, 0.0);
    for (var i = 0u; i < 4u; i++) {
      let j = joints[i];
      let w = weights[i];
      skinPos += skinMatrices[j] * pos4 * w;
      skinNrm += skinMatrices[j] * nrm4 * w;
    }
  }

  let worldPos = (scene.model * skinPos).xyz;
  let worldNrm = normalize((scene.model * skinNrm).xyz);
  let clipPos = scene.viewProj * vec4<f32>(worldPos, 1.0);
  let viewNrm = (scene.viewProj * vec4<f32>(worldNrm, 0.0)).xyz;
  let screenNrm = normalize(viewNrm.xy);
  let offset = screenNrm * (omat.edgeSize * 0.003) * clipPos.w;
  out.position = vec4<f32>(clipPos.xy + offset, clipPos.z, clipPos.w);
  out.uv = uv;
  return out;
}
`;