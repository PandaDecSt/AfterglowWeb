struct Uniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) normal: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) worldPos: vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  out.position = u.viewProj * worldPos;
  out.normal = (u.model * vec4<f32>(normal, 0.0)).xyz;
  out.uv = uv;
  out.worldPos = worldPos.xyz;
  return out;
}

@group(0) @binding(1) var<uniform> params: vec4<f32>;

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let lightDir = normalize(vec3<f32>(1.0, 1.0, 0.5));
  let n = normalize(in.normal);
  let diffuse = max(dot(n, lightDir), 0.0);
  let ambient = 0.15;

  let baseColor = vec3<f32>(
    0.5 + 0.5 * sin(u.time * 0.5 + in.uv.x * 6.28),
    0.5 + 0.5 * sin(u.time * 0.7 + in.uv.y * 6.28),
    0.8
  );

  let color = baseColor * (ambient + diffuse * 0.85);
  return vec4<f32>(color, 1.0);
}
