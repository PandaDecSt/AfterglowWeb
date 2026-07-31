export const BRDF_LUT_SIZE = 64
const BAKE_SAMPLE_COUNT = 32

export const BRDF_LUT_BAKE_WGSL = /* wgsl */ `
const LUT_SIZE: f32 = ${BRDF_LUT_SIZE}.0;
const SAMPLE_COUNT: u32 = ${BAKE_SAMPLE_COUNT}u;
const M_2PI: f32 = 6.283185307179586;

@group(0) @binding(0) var ltcSrc: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
  let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
  let y = f32(vid & 2u) * 2.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

fn orthonormal_basis(N: vec3f) -> mat2x3f {
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.99999);
  let T = normalize(cross(up, N));
  let B = cross(N, T);
  return mat2x3f(T, B);
}

fn sample_ggx_vndf(rand: vec3f, alpha: f32, Vt: vec3f) -> vec3f {
  let Vh = normalize(vec3f(alpha * Vt.xy, Vt.z));
  let tb = orthonormal_basis(Vh);
  let Th = tb[0];
  let Bh = tb[1];
  let r = sqrt(rand.x);
  let x = r * rand.y;
  var y = r * rand.z;
  let s = 0.5 * (1.0 + Vh.z);
  y = (1.0 - s) * sqrt(1.0 - x * x) + s * y;
  let z = sqrt(saturate(1.0 - x * x - y * y));
  let Hh = x * Th + y * Bh + z * Vh;
  return normalize(vec3f(alpha * Hh.xy, saturate(Hh.z)));
}

fn G1_Smith_GGX_opti(NX: f32, a2: f32) -> f32 {
  return NX + sqrt(NX * (NX - NX * a2) + a2);
}

fn F_eta(eta: f32, cos_theta: f32) -> f32 {
  let c = abs(cos_theta);
  var g = eta * eta - 1.0 + c * c;
  if (g > 0.0) {
    g = sqrt(g);
    let A = (g - c) / (g + c);
    let B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * A * A * (1.0 + B * B);
  }
  return 1.0;
}

fn f0_from_ior(eta: f32) -> f32 {
  let A = (eta - 1.0) / (eta + 1.0);
  return A * A;
}

fn F_color_blend_zero(eta: f32, fresnel: f32) -> f32 {
  let f0 = f0_from_ior(eta);
  return saturate((fresnel - f0) / (1.0 - f0));
}

@fragment fn fs(@builtin(position) frag: vec4f) -> @location(0) vec4f {
  let y_uv = floor(frag.y) / (LUT_SIZE - 1.0);
  let x_uv = floor(frag.x) / (LUT_SIZE - 1.0);

  let NV = clamp(1.0 - y_uv * y_uv, 1e-4, 0.9999);
  let a = max(x_uv, 1e-4);
  let a2 = clamp(a * a, 1e-4, 0.9999);

  let V = vec3f(sqrt(1.0 - NV * NV), 0.0, NV);

  let eta = (2.0 / (1.0 - sqrt(0.08 * 1.0))) - 1.0;

  var brdf_accum = 0.0;
  var fresnel_accum = 0.0;
  let sc_f = f32(SAMPLE_COUNT);
  for (var j: u32 = 0u; j < SAMPLE_COUNT; j = j + 1u) {
    for (var i: u32 = 0u; i < SAMPLE_COUNT; i = i + 1u) {
      let ix = (f32(i) + 0.5) / sc_f;
      let iy = (f32(j) + 0.5) / sc_f;
      let Xi = vec3f(ix, cos(iy * M_2PI), sin(iy * M_2PI));

      let H = sample_ggx_vndf(Xi, a, V);
      let L = -reflect(V, H);
      let NL = L.z;
      if (NL > 0.0) {
        let NH = max(H.z, 0.0);
        let VH = max(dot(V, H), 0.0);

        let G1v = G1_Smith_GGX_opti(NV, a2);
        let G1l = G1_Smith_GGX_opti(NL, a2);
        let G_smith = 4.0 * NV * NL / (G1v * G1l);

        let brdf = (G_smith * VH) / (NH * NV);

        let fresnel = F_eta(eta, VH);
        let Fc = F_color_blend_zero(eta, fresnel);

        brdf_accum = brdf_accum + (1.0 - Fc) * brdf;
        fresnel_accum = fresnel_accum + Fc * brdf;
      }
    }
  }
  let n2 = sc_f * sc_f;
  let dfg = vec2f(brdf_accum / n2, fresnel_accum / n2);
  let ltc = textureLoad(ltcSrc, vec2i(i32(frag.x), i32(frag.y)), 0).rg;
  return vec4f(dfg, ltc);
}
`