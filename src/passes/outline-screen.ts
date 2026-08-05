// Screen-space outline machine (Blender-style), shared by every demo that
// wants a rim around selected / toon-shaded geometry. One parameterized shader
// draws a rim on background pixels around any fragment whose id == cfg.targetId.
// Two independent layers reuse this same machine:
//   • cel outline  → reads the GBuffer material channel, targetId = ShadingModelID.TOON
//   • selection     → reads the GBuffer normal.w (object id), targetId = selectedId
// Winding- and depth-independent, so it can never "fill" the object the way an
// inverted hull can.

export const OUTLINE_SCREEN_SHADER = `
struct OutlineCfg {
  resolution: vec2<f32>,
  targetId: f32,
  radius: f32,
  color: vec3<f32>,
  enabled: f32,
};

@group(0) @binding(0) var idTex: texture_2d<f32>;
@group(0) @binding(1) var idSamp: sampler;
@group(0) @binding(2) var<uniform> cfg: OutlineCfg;

struct VSOut {
  @builtin(position) position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0)
  );
  var out: VSOut;
  out.position = vec4<f32>(p[vi], 0.0, 1.0);
  return out;
}

fn idAt(uv: vec2<f32>) -> f32 {
  return textureSampleLevel(idTex, idSamp, uv, 0.0).a;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  if (cfg.enabled < 0.5) { return vec4<f32>(0.0, 0.0, 0.0, 0.0); }
  let uv = fragCoord.xy / cfg.resolution;
  let texel = 1.0 / cfg.resolution;
  let selfId = idAt(uv);
  let maxR = i32(cfg.radius);
  for (var r = 1; r <= maxR; r = r + 1) {
    let o = f32(r) * texel;
    var offs = array<vec2<f32>, 8>(
      vec2<f32>(o.x, 0.0), vec2<f32>(-o.x, 0.0),
      vec2<f32>(0.0, o.y), vec2<f32>(0.0, -o.y),
      vec2<f32>(o.x, o.y), vec2<f32>(-o.x, o.y),
      vec2<f32>(o.x, -o.y), vec2<f32>(-o.x, -o.y)
    );
    for (var i = 0; i < 8; i = i + 1) {
      let nId = idAt(uv + offs[i]);
      // self is background, neighbor belongs to the target → rim on this pixel
      if (selfId < 0.5 && abs(nId - cfg.targetId) < 0.5) {
        return vec4<f32>(cfg.color, 1.0);
      }
    }
  }
  return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
`;
