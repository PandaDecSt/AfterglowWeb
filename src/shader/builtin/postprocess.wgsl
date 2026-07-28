struct PostUniforms {
  resolution: vec2<f32>,
  time: f32,
  bloomEnabled: f32,
};

@group(0) @binding(0) var<uniform> pu: PostUniforms;
@group(0) @binding(1) var sceneTex: texture_2d<f32>;
@group(0) @binding(2) var sceneSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VSOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
  );
  var out: VSOut;
  out.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
  out.uv = positions[vertexIndex] * 0.5 + 0.5;
  return out;
}

fn vignette(uv: vec2<f32>) -> f32 {
  let d = distance(uv, vec2<f32>(0.5));
  return smoothstep(0.7, 0.3, d);
}

fn chromaticAberration(uv: vec2<f32>, strength: f32) -> vec3<f32> {
  let dir = uv - 0.5;
  let dist = length(dir);
  let offset = dir * dist * strength;
  let r = textureSample(sceneTex, sceneSampler, uv + offset).r;
  let g = textureSample(sceneTex, sceneSampler, uv).g;
  let b = textureSample(sceneTex, sceneSampler, uv - offset).b;
  return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  var color = chromaticAberration(in.uv, 0.003);

  color *= vignette(in.uv);

  let scanline = sin(in.uv.y * pu.resolution.y * 1.5 + pu.time * 2.0) * 0.02;
  color -= scanline;

  color = pow(color, vec3<f32>(1.0 / 2.2));

  return vec4<f32>(color, 1.0);
}
