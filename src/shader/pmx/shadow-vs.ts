export const SHADOW_VS = `
struct ShadowScene {
  lightVP: mat4x4<f32>,
  model: mat4x4<f32>,
  flags: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};
@group(0) @binding(0) var<uniform> shadowScene: ShadowScene;

@group(1) @binding(0) var<storage, read> skinMatrices: array<mat4x4<f32>>;

@vertex
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>,
) -> @builtin(position) vec4<f32> {
  var skinPos: vec4<f32>;
  if ((shadowScene.flags & 1u) != 0u) {
    skinPos = vec4<f32>(position, 1.0);
  } else {
    skinPos = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    let pos4 = vec4<f32>(position, 1.0);
    for (var i = 0u; i < 4u; i++) {
      let j = joints[i];
      let w = weights[i];
      skinPos += skinMatrices[j] * pos4 * w;
    }
  }
  let worldPos = (shadowScene.model * skinPos).xyz;
  return shadowScene.lightVP * vec4<f32>(worldPos, 1.0);
}
`;