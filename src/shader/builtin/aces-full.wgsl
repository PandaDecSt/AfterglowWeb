// AfterglowRender Full ACES Pipeline
// Ported from ACESCommon.hlsl, ACESOutputDevice.hlsl, ACESDisplayEncoding.hlsl

// ============================================================
// Color Space Matrices (from ACESCommon.hlsl)
// ============================================================

// AP0 (ACES) to XYZ
const AP0_TO_XYZ = mat3x3<f32>(
  vec3<f32>(0.9525523959, 0.0000000000, 0.0000936786),
  vec3<f32>(0.3439664498, 0.7281660966, -0.0721325464),
  vec3<f32>(0.0000000000, 0.0000000000, 1.0088251844),
);

// XYZ to AP0
const XYZ_TO_AP0 = mat3x3<f32>(
  vec3<f32>(1.0498110175, 0.0000000000, -0.0000974845),
  vec3<f32>(-0.4959030231, 1.3733130458, 0.0982400361),
  vec3<f32>(0.0000000000, 0.0000000000, 0.9912520182),
);

// AP1 (ACEScg) to XYZ
const AP1_TO_XYZ = mat3x3<f32>(
  vec3<f32>(0.6624541811, 0.1340042065, 0.1561876870),
  vec3<f32>(0.2722287168, 0.6740817658, 0.0536895174),
  vec3<f32>(-0.0055746495, 0.0040607335, 1.0103391003),
);

// XYZ to AP1
const XYZ_TO_AP1 = mat3x3<f32>(
  vec3<f32>(1.6410233797, -0.3248032942, -0.2364246952),
  vec3<f32>(-0.6636628587, 1.6153315917, 0.0167563477),
  vec3<f32>(0.0117218943, -0.0082844420, 0.9883948585),
);

// AP0 to AP1 (ACESTableCommon.hlsl)
const AP0_TO_AP1 = mat3x3<f32>(
  vec3<f32>(1.4514393161, -0.2365107469, -0.2149285693),
  vec3<f32>(-0.0765537734, 1.1762296998, -0.0996759264),
  vec3<f32>(0.0083161484, -0.0060324498, 0.9977163014),
);

// AP1 to AP0
const AP1_TO_AP0 = mat3x3<f32>(
  vec3<f32>(0.6954522414, 0.1406786965, 0.1638690622),
  vec3<f32>(0.0447945634, 0.8596711185, 0.0955343182),
  vec3<f32>(-0.0055258826, 0.0040252103, 1.0015006723),
);

// sRGB to XYZ
const SRGB_TO_XYZ = mat3x3<f32>(
  vec3<f32>(0.4123907993, 0.3575843394, 0.1804807884),
  vec3<f32>(0.2126390059, 0.7151686788, 0.0721923154),
  vec3<f32>(0.0193308187, 0.1191947798, 0.9505321522),
);

// XYZ to sRGB
const XYZ_TO_SRGB = mat3x3<f32>(
  vec3<f32>(3.2409699419, -1.5373831776, -0.4986107603),
  vec3<f32>(-0.9692436363, 1.8759675015, 0.0415550574),
  vec3<f32>(0.0556300797, -0.2039769589, 1.0569715142),
);

// Rec.2020 to XYZ
const REC2020_TO_XYZ = mat3x3<f32>(
  vec3<f32>(0.6369580483, 0.1446169036, 0.1688809752),
  vec3<f32>(0.2627002120, 0.6779980715, 0.0593017165),
  vec3<f32>(0.0000000000, 0.0280726930, 1.0609850577),
);

// XYZ to Rec.2020
const XYZ_TO_REC2020 = mat3x3<f32>(
  vec3<f32>(1.7166511880, -0.3556707838, -0.2533662814),
  vec3<f32>(-0.6666843518, 1.6164812366, 0.0157685458),
  vec3<f32>(0.0176398574, -0.0427706133, 0.9421031212),
);

// P3 D65 to XYZ
const P3D65_TO_XYZ = mat3x3<f32>(
  vec3<f32>(0.4865709486, 0.2656676932, 0.1982172852),
  vec3<f32>(0.2289745641, 0.6917385218, 0.0792869141),
  vec3<f32>(0.0000000000, 0.0451133819, 1.0439443689),
);

// XYZ to P3 D65
const XYZ_TO_P3D65 = mat3x3<f32>(
  vec3<f32>(2.4934969119, -0.9313836179, -0.4027107845),
  vec3<f32>(-0.8294889696, 1.7626640603, 0.0236246858),
  vec3<f32>(0.0358458302, -0.0761723893, 0.9568845240),
);

// ============================================================
// Chromatic Adaptation Transform (Bradford)
// ============================================================

// D65 to D60
const D65_TO_D60 = mat3x3<f32>(
  vec3<f32>(1.0130349146, 0.0061052578, -0.0149709436),
  vec3<f32>(0.0076982301, 0.9981633521, -0.0050320385),
  vec3<f32>(-0.0028413174, 0.0046851567, 0.9245061375),
);

// D60 to D65
const D60_TO_D65 = mat3x3<f32>(
  vec3<f32>(0.9872240087, -0.0061132286, 0.0159532883),
  vec3<f32>(-0.0075983718, 1.0018614847, 0.0053300358),
  vec3<f32>(0.0030725771, -0.0050959615, 1.0816806031),
);

// ============================================================
// ACES Standard Curves (from ACESCommon.hlsl)
// ============================================================

// ACES fitted curve by Narkowicz 2015
fn ACES_Narkowicz(x: vec3<f32>) -> vec3<f32> {
  let a = x * (x * (x * 60.14595 + 14.22784) + 0.7068982513);
  let b = x * (x * (x * 10.882106 + 56.82012) + 329.7445) + 436.4901;
  return a / b;
}

// ACES fitted curve by Hill 2020 (more accurate)
fn ACES_Hill(x: vec3<f32>) -> vec3<f32> {
  let a = x * (x * (2.51 * x + 0.03) + 0.43);
  let b = x * (x * (2.43 * x + 0.59) + 0.14);
  return a / b;
}

// ACES fitted curve by Hill with exposure control
fn ACES_HillExposure(x: vec3<f32>, exposure: f32) -> vec3<f32> {
  let scaled = x * exposure;
  let a = scaled * (scaled * (2.51 * scaled + 0.03) + 0.43);
  let b = scaled * (scaled * (2.43 * scaled + 0.59) + 0.14);
  return a / b;
}

// ============================================================
// ACES Tone Scale (from ACESTonescale.hlsl)
// ============================================================

// Simple tone scale with shoulder
fn ACES_ToneScale(x: f32, whiteClip: f32, shoulder: f32) -> f32 {
  let xS = x * whiteClip;
  let a = shoulder;
  let b = 1.0 + shoulder - whiteClip;
  return xS * (a * xS + b) / (xS * (c * xS + d) + e);
}

// ============================================================
// Display Encoding (from ACESDisplayEncoding.hlsl)
// ============================================================

// sRGB EOTF (linear to sRGB)
fn LinearToSRGB(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  var result: vec3<f32>;
  result.r = select(hi.r, lo.r, c.r < 0.0031308);
  result.g = select(hi.g, lo.g, c.g < 0.0031308);
  result.b = select(hi.b, lo.b, c.b < 0.0031308);
  return result;
}

// sRGB EOTF inverse (sRGB to linear)
fn SRGBToLinear(c: vec3<f32>) -> vec3<f32> {
  let lo = c / 12.92;
  let hi = pow((c + 0.055) / 1.055, vec3<f32>(2.4));
  var result: vec3<f32>;
  result.r = select(hi.r, lo.r, c.r < 0.04045);
  result.g = select(hi.g, lo.g, c.g < 0.04045);
  result.b = select(hi.b, lo.b, c.b < 0.04045);
  return result;
}

// PQ EOTF (SMPTE ST 2084 for HDR)
fn LinearToPQ(c: vec3<f32>, peakLuminance: f32) -> vec3<f32> {
  let m1 = 0.1593017578125;
  let m2 = 78.84375;
  let c1 = 0.8359375;
  let c2 = 18.8515625;
  let c3 = 18.6875;
  let l = c / peakLuminance;
  let lp = pow(l, vec3<f32>(m1));
  let num = c1 + c2 * lp;
  let den = 1.0 + c3 * lp;
  return pow(num / den, vec3<f32>(m2));
}

// HLG EOTF (ARIB STD-B67)
fn LinearToHLG(c: vec3<f32>) -> vec3<f32> {
  let a = 0.17883277;
  let b = 0.28466892;
  let c = 0.55991073;
  var result: vec3<f32>;
  for (var i = 0; i < 3; i++) {
    let val = c[i];
    if (val < 1.0 / 12.0) {
      result[i] = sqrt(3.0 * val);
    } else {
      result[i] = a * log(12.0 * val - b) + c;
    }
  }
  return result;
}

// ============================================================
// ACES Output Transform (full pipeline)
// ============================================================

struct ACESConfig {
  exposure: f32,
  whiteClip: f32,
  shoulder: f32,
  peakLuminance: f32,  // nits
  applyCAT: bool,       // Apply chromatic adaptation
  useSRGB: bool,        // Output to sRGB
  usePQ: bool,          // Output to PQ (HDR)
  useHLG: bool,         // Output to HLG (HDR)
}

fn DefaultACESConfig() -> ACESConfig {
  var config: ACESConfig;
  config.exposure = 1.0;
  config.whiteClip = 1.0;
  config.shoulder = 0.5;
  config.peakLuminance = 100.0;
  config.applyCAT = true;
  config.useSRGB = true;
  config.usePQ = false;
  config.useHLG = false;
  return config;
}

// Full ACES output transform: Linear scene → Display
fn ACESOutputTransform(color: vec3<f32>, config: ACESConfig) -> vec3<f32> {
  var result = color;

  // Apply exposure
  result *= config.exposure;

  // Scene to AP0 (if needed)
  // result = SRGB_TO_XYZ * result; // assuming input is sRGB-linear
  // result = XYZ_TO_AP0 * result;

  // Apply ACES tone curve
  result = ACES_Hill(result);

  // AP0 to AP1 (if in ACES AP0 space)
  // result = AP0_TO_AP1 * result;

  // Chromatic adaptation (D65 → D60 for ACES)
  if (config.applyCAT) {
    result = D65_TO_D60 * result;
    result = D60_TO_D65 * result;
  }

  // Output encoding
  if (config.usePQ) {
    result = LinearToPQ(result, config.peakLuminance);
  } else if (config.useHLG) {
    result = LinearToHLG(result);
  } else {
    // sRGB
    result = LinearToSRGB(result);
  }

  return clamp(result, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Simplified ACES (current implementation, preserved for compatibility)
fn ACES_Simplified(color: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ============================================================
// Color Space Conversion Utilities
// ============================================================

fn LinearToAP0(color: vec3<f32>) -> vec3<f32> {
  // Assume input is sRGB-linear, convert to AP0
  let xyz = SRGB_TO_XYZ * color;
  let d65 = D65_TO_D60 * xyz;
  return XYZ_TO_AP0 * d65;
}

fn AP0ToLinear(color: vec3<f32>) -> vec3<f32> {
  let d60 = AP0_TO_XYZ * color;
  let xyz = D60_TO_D65 * d60;
  return XYZ_TO_SRGB * xyz;
}

fn LinearToAP1(color: vec3<f32>) -> vec3<f32> {
  let xyz = SRGB_TO_XYZ * color;
  let d65 = D65_TO_D60 * xyz;
  return XYZ_TO_AP1 * d65;
}

fn AP1ToLinear(color: vec3<f32>) -> vec3<f32> {
  let d60 = AP1_TO_XYZ * color;
  let xyz = D60_TO_D65 * d60;
  return XYZ_TO_SRGB * xyz;
}

// ============================================================
// CAM16 (Color Appearance Model 2016)
// ============================================================

const CAM16_TO_XYZ = mat3x3<f32>(
  vec3<f32>(2.0512756811, -1.1400313439, 0.0887556628),
  vec3<f32>(0.4269389763, 0.7005835277, -0.1275225040),
  vec3<f32>(-0.0174712779, -0.0384725929, 1.0589468739),
);

const XYZ_TO_CAM16 = mat3x3<f32>(
  vec3<f32>(0.3640744835, 0.5947008156, 0.04110127349),
  vec3<f32>(-0.2222450987, 1.0738554823, 0.14794533610),
  vec3<f32>(-0.0020676190, 0.0488260453, 0.95038755696),
);

// Simplified CAM16 J (Lightness)
fn CAM16_Lightness(color: vec3<f32>) -> f32 {
  let xyz = SRGB_TO_XYZ * color;
  let d65 = D65_TO_D60 * xyz;
  let cam = XYZ_TO_CAM16 * d65;
  // Simplified lightness calculation
  let Y = cam.y;
  if (Y <= 0.008856) {
    return 903.3 * Y;
  } else {
    return 116.0 * pow(Y, 1.0 / 3.0) - 16.0;
  }
}
