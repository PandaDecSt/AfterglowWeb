export const OUTLINE_FS = `
struct OutlineMat {
  edgeColor: vec4<f32>,
  edgeSize: f32,
  _p0: f32, _p1: f32, _p2: f32,
};
@group(0) @binding(1) var<uniform> omat: OutlineMat;
@group(0) @binding(2) var diffuseTex: texture_2d<f32>;
@group(0) @binding(5) var texSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

struct OutlineFSOut {
  @location(0) color: vec4<f32>,
  @location(1) mask: vec4<f32>,
};

@fragment
fn fs_main(in: VSOut) -> OutlineFSOut {
  let texA = textureSample(diffuseTex, texSampler, in.uv).a;
  if (texA < 0.05) { discard; }
  var out: OutlineFSOut;
  out.color = vec4<f32>(omat.edgeColor.rgb, omat.edgeColor.a * texA);
  out.mask = vec4<f32>(1.0, omat.edgeColor.a * texA, 0.0, 0.0);
  return out;
}
`;