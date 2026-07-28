// Endfield Toon Textured Shader
// Supports: Color Ramp, SDF Face Shadow, Material Mask, Hair Property/Ramp textures

struct ToonUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  objectUp: vec4<f32>,
  objectRight: vec4<f32>,
  objectForward: vec4<f32>,
  time: f32,
  outlineWidth: f32,
  materialID: f32,
  viewAttenuation: f32,
  rimLightWidth: f32,
  rimLightIntensity: f32,
  subsurfaceMFP: f32,
  rampOffset: f32,
};

@group(0) @binding(0) var<uniform> u: ToonUniforms;

// Textures
@group(1) @binding(0) var albedoTex: texture_2d<f32>;
@group(1) @binding(1) var albedoSampler: sampler;
@group(1) @binding(2) var rampTex: texture_2d<f32>;
@group(1) @binding(3) var rampSampler: sampler;
@group(1) @binding(4) var maskTex: texture_2d<f32>;
@group(1) @binding(5) var maskSampler: sampler;
@group(1) @binding(6) var sdfTex: texture_2d<f32>;
@group(1) @binding(7) var sdfSampler: sampler;
@group(1) @binding(8) var propertyTex: texture_2d<f32>;
@group(1) @binding(9) var propertySampler: sampler;
@group(1) @binding(10) var hairRampTex: texture_2d<f32>;
@group(1) @binding(11) var hairRampSampler: sampler;
@group(1) @binding(12) var normalTex: texture_2d<f32>;
@group(1) @binding(13) var normalSampler: sampler;

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) vertColor: vec3<f32>,
  @location(4) worldTangent: vec3<f32>,
  @location(5) worldBitangent: vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) color: vec3<f32>,
  @location(4) tangent: vec3<f32>,
  @location(5) bitangent: vec3<f32>,
) -> VSOut {
  var out: VSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  out.position = u.viewProj * worldPos;
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  out.vertColor = color;
  out.worldTangent = normalize((u.invTransModel * vec4<f32>(tangent, 0.0)).xyz);
  out.worldBitangent = normalize((u.invTransModel * vec4<f32>(bitangent, 0.0)).xyz);
  return out;
}

// ============================================================
// Utility Functions
// ============================================================

fn Snorm(value: f32) -> f32 {
  return value * 2.0 - 1.0;
}

fn Unorm(value: f32) -> f32 {
  return value * 0.5 + 0.5;
}

fn Desaturation(color: vec3<f32>, amount: f32) -> vec3<f32> {
  let grey = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  return mix(color, vec3<f32>(grey), amount);
}

fn HueShift(color: vec3<f32>, amount: f32) -> vec3<f32> {
  let angle = amount * 3.14159265;
  let s = sin(angle);
  let c = cos(angle);
  let weights = vec3<f32>(0.2126, 0.7152, 0.0722);
  let grey = dot(color, weights);
  let result = vec3<f32>(
    dot(color, vec3<f32>(0.787 * c + 0.213, -0.213 * c + 0.213 * s - 0.143 * s, 0.143 * c - 0.213 * s - 0.787 * s)),
    dot(color, vec3<f32>(-0.213 * c + 0.143 * s + 0.787 * s, 0.787 * c + 0.213, -0.143 * c + 0.213 * s + 0.143 * s)),
    dot(color, vec3<f32>(0.143 * c + 0.213 * s + 0.143 * s, -0.213 * c + 0.787 * s - 0.143 * s, 0.787 * c + 0.213))
  );
  return mix(color, result, amount);
}

fn ProjectDirection(dir: vec3<f32>, up: vec3<f32>) -> vec3<f32> {
  return normalize(dir - up * dot(dir, up));
}

fn ReconstructNormal(encoded: vec2<f32>) -> vec3<f32> {
  let z = sqrt(1.0 - encoded.x * encoded.x - encoded.y * encoded.y);
  return vec3<f32>(encoded.x, encoded.y, z);
}

fn DecodeNormal(encoded: vec2<f32>) -> vec2<f32> {
  return encoded * 2.0 - 1.0;
}

// ============================================================
// Rim Lighting (from EndfieldCommon.hlsl)
// ============================================================

fn EndfieldRimLighting(nov: f32, vol: f32, nol: f32, smoothness: f32, occlusion: f32, width: f32) -> f32 {
  let rimStepMin = clamp(0.9 - width, 0.0, 0.99);
  let rimStepMax = clamp(1.0 - width, 0.01, 1.0);
  var rimLighting = smoothness * smoothstep(rimStepMin, rimStepMax, 1.0 - nov);
  rimLighting *= max(vol, 0.0) * max(nol + 0.5, 0.0) * 2.0;
  rimLighting *= u.lightDir.w;
  return rimLighting;
}

// ============================================================
// Face Shading (with SDF shadow)
// ============================================================

fn shadeFace(N: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let baseColor = textureSample(albedoTex, albedoSampler, uv).rgb;
  let mask = textureSample(maskTex, maskSampler, uv);

  let vertNov = dot(N, V);
  let vol = dot(-V, u.lightDir.xyz);
  let vertNol = dot(N, u.lightDir.xyz);

  // SDF Face Shadow
  let projectedLightDir = ProjectDirection(u.lightDir.xyz, u.objectUp.xyz);
  let rol = dot(projectedLightDir, u.objectRight.xyz);

  var sdfCoord = uv;
  if (rol > 0.0) {
    sdfCoord.x = 1.0 - sdfCoord.x;
  }
  let sdf = textureSample(sdfTex, sdfSampler, sdfCoord);

  let fol = dot(u.lightDir.xyz, u.objectForward.xyz);
  let normalizedlightAngle = atan2(rol, -fol) * 0.318309886;
  let faceLightAngle = Unorm(((sdf.x + sdf.y) * 0.5) - abs(normalizedlightAngle));
  var faceRadiance = smoothstep(0.5, 0.6, faceLightAngle);
  faceRadiance = mix(faceRadiance, max(dot(N, u.lightDir.xyz), 0.0), mask.y);

  // View attenuation
  let viewFactor = lerp(1.0, max(vertNov, 0.0), u.viewAttenuation * mask.x);
  faceRadiance *= viewFactor;
  let fadedSubsurfaceMFP = u.subsurfaceMFP * (0.5 + viewFactor * 0.5);

  // Color Ramp
  let rampCoord = vec2<f32>(clamp((1.0 - u.rampOffset) - faceRadiance * (0.5 - (u.rampOffset * 0.5)), 0.1, 0.9), 0.5);
  let rampColor = textureSample(rampTex, rampSampler, rampCoord);

  var finalColor = baseColor * (faceRadiance + (rampColor.rgb * rampColor.a) * min(fadedSubsurfaceMFP, baseColor));

  // Nose color
  finalColor = mix(finalColor * 0.5, finalColor, baseColor.a);

  // Rim light
  let rimLighting = EndfieldRimLighting(vertNov, vol, vertNol, 1.0, mask.w, u.rimLightWidth);
  let rimLightMask = smoothstep(0.0, 0.01, max(Snorm(uv.x) * rol, 0.0));
  finalColor += rimLighting * rimLightMask * u.rimLightIntensity;

  finalColor *= u.lightDir.xyz * u.lightDir.w * 0.5;

  return finalColor;
}

// ============================================================
// Body Shading (with color ramp)
// ============================================================

fn shadeBody(N: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let baseColor = textureSample(albedoTex, albedoSampler, uv).rgb;

  let vertNov = dot(N, V);
  let vol = dot(-V, u.lightDir.xyz);
  let vertNol = dot(N, u.lightDir.xyz);

  let radiance = max(dot(N, u.lightDir.xyz), 0.0);

  // View attenuation
  let viewFactor = lerp(1.0, max(vertNov, 0.0), u.viewAttenuation);
  let fadedRadiance = radiance * viewFactor;
  let fadedSubsurfaceMFP = u.subsurfaceMFP * (0.5 + viewFactor * 0.5);

  // Color Ramp
  let rampCoord = vec2<f32>(clamp((1.0 - u.rampOffset) - fadedRadiance * (0.5 - (u.rampOffset * 0.5)), 0.1, 0.9), 0.5);
  let rampColor = textureSample(rampTex, rampSampler, rampCoord);

  var finalColor = baseColor * (fadedRadiance + (rampColor.rgb * rampColor.a) * min(fadedSubsurfaceMFP, baseColor));

  // Rim light
  let rimLighting = EndfieldRimLighting(vertNov, vol, vertNol, 1.0, 1.0, u.rimLightWidth);
  finalColor += rimLighting * u.rimLightIntensity;

  finalColor *= u.lightDir.xyz * u.lightDir.w * 0.5;

  return finalColor;
}

// ============================================================
// Hair Shading (anisotropic + property texture)
// ============================================================

fn shadeHair(N: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let baseColor = textureSample(albedoTex, albedoSampler, uv).rgb;
  let property = textureSample(propertyTex, propertySampler, uv);

  // Reconstruct normal from property texture
  let normal = normalize(N);
  let tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), normal));
  let bitangent = cross(normal, tangent);

  // Gram-Schmidt orthogonalization
  let orthoTangent = normalize(tangent - dot(tangent, normal) * normal);
  let orthoBitangent = cross(property.x >= 0.5 ? normal : normal, orthoTangent);

  // Rim light
  let rimLighting = EndfieldRimLighting(vertNov, vol, vertNol, property.w, property.z, u.rimLightWidth);

  // Hair color ramp
  let hairRampCoord = saturate(vec2<f32>(abs(dot(normal, u.lightDir.xyz)), 0.5));
  let hairRamp = textureSample(hairRampTex, hairRampSampler, hairRampCoord);

  // Simplified anisotropic shading
  let H = normalize(V + u.lightDir.xyz);
  let TdotH = dot(orthoTangent, H);
  let aniso = pow(sqrt(max(1.0 - TdotH * TdotH, 0.0)), 8.0);

  var finalColor = baseColor * property.z * 0.7;
  finalColor += hairRamp.rgb * property.w * aniso;
  finalColor += rimLighting;

  return finalColor;
}

// ============================================================
// Iris/Eye Shading
// ============================================================

fn shadeIris(N: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  let baseColor = textureSample(albedoTex, albedoSampler, uv).rgb;

  let H = normalize(V + u.lightDir.xyz);
  let spec = pow(max(dot(N, H), 0.0), 64.0);
  var finalColor = baseColor;
  finalColor += step(0.5, spec) * 0.9;

  // Eye shadow (top gradient)
  let shadow = 1.0 - smoothstep(0.7, 1.0, uv.y) * 0.4;
  finalColor *= shadow;

  return finalColor;
}

// ============================================================
// Eye Shadow Pass
// ============================================================

fn shadeEyeShadow(N: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  // Simple shadow color from uniform
  return vec3<f32>(0.3, 0.2, 0.25);
}

// ============================================================
// Hair Shadow Pass
// ============================================================

fn shadeHairShadow(N: vec3<f32>, V: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
  // Hair shadow color
  return vec3<f32>(0.15, 0.1, 0.12);
}

// ============================================================
// Main Fragment
// ============================================================

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(u.cameraPosition.xyz - in.worldPos);

  var color: vec3<f32>;
  let matID = i32(u.materialID);

  if (matID == 0) {
    // Face with SDF shadow
    color = shadeFace(N, V, in.uv);
  } else if (matID == 1) {
    // Hair with anisotropic shading
    color = shadeHair(N, V, in.uv);
  } else if (matID == 2) {
    // Body/Clothes with color ramp
    color = shadeBody(N, V, in.uv);
  } else if (matID == 3) {
    // Eye/Iris
    color = shadeIris(N, V, in.uv);
  } else if (matID == 4) {
    // Eye shadow pass
    color = shadeEyeShadow(N, V, in.uv);
  } else {
    // Hair shadow pass
    color = shadeHairShadow(N, V, in.uv);
  }

  return vec4<f32>(color, 1.0);
}
