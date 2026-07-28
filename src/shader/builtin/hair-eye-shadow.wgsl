// AfterglowRender Hair/Eye Shadow Shader
// Ported from EndfieldHairShadow_VS/FS.hlsl, EndfieldEyeShadow_FS.hlsl, HairCommon.hlsl

// ============================================================
// Hair Shadow Pass
// ============================================================

struct HairShadowUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  shadowColor: vec4<f32>,
  shadowIntensity: f32,
  depthOffset: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> u: HairShadowUniforms;

// Hair shadow textures
@group(1) @binding(0) var albedoTex: texture_2d<f32>;
@group(1) @binding(1) var albedoSampler: sampler;
@group(1) @binding(2) var propertyTex: texture_2d<f32>;
@group(1) @binding(3) var propertySampler: sampler;

struct HairShadowVSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> HairShadowVSOut {
  var out: HairShadowVSOut;
  let worldPos = u.model * vec4<f32>(pos, 1.0);
  // Apply depth offset for shadow
  let lightDir = normalize(u.lightDir.xyz);
  let offsetPos = worldPos.xyz + lightDir * u.depthOffset;
  out.position = u.viewProj * vec4<f32>(offsetPos, 1.0);
  out.worldNormal = normalize((u.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  return out;
}

@fragment
fn fs_main(in: HairShadowVSOut) -> @location(0) vec4<f32> {
  // Simple shadow color output
  return vec4<f32>(u.shadowColor.rgb, u.shadowColor.a * u.shadowIntensity);
}

// ============================================================
// Hair Shading (Anisotropic Kajiya-Kay)
// ============================================================

struct HairShadingUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  time: f32,
  specularPower: f32,
  anisotropy: f32,
  rimWidth: f32,
  rimIntensity: f32,
  strandCount: f32,
  pad0: f32,
};

@group(0) @binding(0) var<uniform> hu: HairShadingUniforms;

@group(1) @binding(0) var hairAlbedoTex: texture_2d<f32>;
@group(1) @binding(1) var hairAlbedoSampler: sampler;
@group(1) @binding(2) var hairPropertyTex: texture_2d<f32>;
@group(1) @binding(3) var hairPropertySampler: sampler;
@group(1) @binding(4) var hairNormalTex: texture_2d<f32>;
@group(1) @binding(5) var hairNormalSampler: sampler;
@group(1) @binding(6) var hairRampTex: texture_2d<f32>;
@group(1) @binding(7) var hairRampSampler: sampler;

struct HairShadingVSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) worldTangent: vec3<f32>,
  @location(4) worldBitangent: vec3<f32>,
};

@vertex
fn vs_hair(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) tangent: vec3<f32>,
  @location(4) bitangent: vec3<f32>,
) -> HairShadingVSOut {
  var out: HairShadingVSOut;
  let worldPos = hu.model * vec4<f32>(pos, 1.0);
  out.position = hu.viewProj * worldPos;
  out.worldNormal = normalize((hu.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  out.worldTangent = normalize((hu.invTransModel * vec4<f32>(tangent, 0.0)).xyz);
  out.worldBitangent = normalize((hu.invTransModel * vec4<f32>(bitangent, 0.0)).xyz);
  return out;
}

// Kajiya-Kay anisotropic specular model
fn KajiyaKaySpecular(
  tangent: vec3<f32>,
  normal: vec3<f32>,
  viewDir: vec3<f32>,
  lightDir: vec3<f32>,
  power: f32
) -> f32 {
  let T = normalize(tangent);
  let V = normalize(viewDir);
  let L = normalize(lightDir);

  // Tangent-based specular
  let TdotV = dot(T, V);
  let TdotL = dot(T, L);
  let VxL = cross(V, L);

  let sinH = length(VxL);
  let cosH = dot(V, L);

  // Kajiya-Kay model
  let spec = pow(sqrt(1.0 - TdotV * TdotV) * sqrt(1.0 - TdotL * TdotL) - TdotV * TdotL, power);
  return spec;
}

// Hair diffuse (wrapped diffuse)
fn HairDiffuse(normal: vec3<f32>, lightDir: vec3<f32>) -> f32 {
  let NdotL = dot(normal, lightDir);
  return max(NdotL * 0.5 + 0.5, 0.0);
}

// Hair rim light
fn HairRimLight(normal: vec3<f32>, viewDir: vec3<f32>, width: f32) -> f32 {
  let rim = 1.0 - max(dot(normal, viewDir), 0.0);
  return smoothstep(1.0 - width, 1.0, rim);
}

@fragment
fn fs_hair(in: HairShadingVSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let T = normalize(in.worldTangent);
  let B = normalize(in.worldBitangent);
  let V = normalize(hu.cameraPosition.xyz - in.worldPos);
  let L = normalize(-hu.lightDir.xyz);

  // Sample textures
  let baseColor = textureSample(hairAlbedoTex, hairAlbedoSampler, in.uv).rgb;
  let property = textureSample(hairPropertyTex, hairPropertySampler, in.uv);

  // Reconstruct hair normal from property texture
  let hairNormal = normalize(mix(N, B, property.x));

  // Gram-Schmidt orthogonalization for tangent
  let orthoT = normalize(T - dot(T, hairNormal) * hairNormal);
  let orthoB = cross(hairNormal, orthoT);

  // Diffuse
  let diffuse = HairDiffuse(hairNormal, L) * property.z;

  // Anisotropic specular
  let H = normalize(V + L);
  let spec = KajiyaKaySpecular(orthoT, hairNormal, V, L, hu.specularPower);

  // Hair ramp texture for specular color
  let rampCoord = vec2<f32>(abs(dot(hairNormal, L)), 0.5);
  let hairRamp = textureSample(hairRampTex, hairRampSampler, rampCoord);

  // Rim light
  let rim = HairRimLight(N, V, hu.rimWidth) * hu.rimIntensity;

  // Strand pattern (simulated)
  let strands = sin(in.uv.x * hu.strandCount + in.uv.y * 20.0) * 0.5 + 0.5;

  // Final color
  var color = baseColor * diffuse;
  color += hairRamp.rgb * property.w * spec;
  color += baseColor * rim;
  color *= 0.9 + strands * 0.1;

  return vec4<f32>(color, 1.0);
}

// ============================================================
// Eye Shadow Pass
// ============================================================

struct EyeShadowUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  shadowColor: vec4<f32>,
  shadowIntensity: f32,
  depthOffset: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> eu: EyeShadowUniforms;

struct EyeShadowVSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_eye_shadow(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> EyeShadowVSOut {
  var out: EyeShadowVSOut;
  let worldPos = eu.model * vec4<f32>(pos, 1.0);
  let lightDir = normalize(eu.lightDir.xyz);
  let offsetPos = worldPos.xyz + lightDir * eu.depthOffset;
  out.position = eu.viewProj * vec4<f32>(offsetPos, 1.0);
  out.worldNormal = normalize((eu.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  return out;
}

@fragment
fn fs_eye_shadow(in: EyeShadowVSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(eu.shadowColor.rgb, eu.shadowColor.a * eu.shadowIntensity);
}

// ============================================================
// Eye Shading (with SDF shadow)
// ============================================================

struct EyeShadingUniforms {
  viewProj: mat4x4<f32>,
  model: mat4x4<f32>,
  invTransModel: mat4x4<f32>,
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  objectUp: vec4<f32>,
  objectRight: vec4<f32>,
  objectForward: vec4<f32>,
  time: f32,
  viewAttenuation: f32,
  pad0: f32,
  pad1: f32,
};

@group(0) @binding(0) var<uniform> eyeu: EyeShadingUniforms;

@group(1) @binding(0) var eyeAlbedoTex: texture_2d<f32>;
@group(1) @binding(1) var eyeAlbedoSampler: sampler;
@group(1) @binding(2) var eyeMaskTex: texture_2d<f32>;
@group(1) @binding(3) var eyeMaskSampler: sampler;
@group(1) @binding(4) var eyeSdfTex: texture_2d<f32>;
@group(1) @binding(5) var eyeSdfSampler: sampler;
@group(1) @binding(6) var eyeRampTex: texture_2d<f32>;
@group(1) @binding(7) var eyeRampSampler: sampler;
@group(1) @binding(8) var eyeHighlightTex: texture_2d<f32>;
@group(1) @binding(9) var eyeHighlightSampler: sampler;

struct EyeShadingVSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_eye(
  @location(0) pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> EyeShadingVSOut {
  var out: EyeShadingVSOut;
  let worldPos = eyeu.model * vec4<f32>(pos, 1.0);
  out.position = eyeu.viewProj * worldPos;
  out.worldNormal = normalize((eyeu.invTransModel * vec4<f32>(normal, 0.0)).xyz);
  out.worldPos = worldPos.xyz;
  out.uv = uv;
  return out;
}

@fragment
fn fs_eye(in: EyeShadingVSOut) -> @location(0) vec4<f32> {
  let N = normalize(in.worldNormal);
  let V = normalize(eyeu.cameraPosition.xyz - in.worldPos);

  let baseColor = textureSample(eyeAlbedoTex, eyeAlbedoSampler, in.uv).rgb;
  let mask = textureSample(eyeMaskTex, eyeMaskSampler, in.uv);

  let vertNov = dot(N, V);
  let vol = dot(-V, eyeu.lightDir.xyz);
  let vertNol = dot(N, eyeu.lightDir.xyz);

  // SDF Face Shadow (for eye area)
  let projectedLightDir = normalize(eyeu.lightDir.xyz - eyeu.objectUp.xyz * dot(eyeu.lightDir.xyz, eyeu.objectUp.xyz));
  let rol = dot(projectedLightDir, eyeu.objectRight.xyz);

  var sdfCoord = in.uv;
  if (rol > 0.0) {
    sdfCoord.x = 1.0 - sdfCoord.x;
  }
  let sdf = textureSample(eyeSdfTex, eyeSdfSampler, sdfCoord);

  let fol = dot(eyeu.lightDir.xyz, eyeu.objectForward.xyz);
  let normalizedlightAngle = atan2(rol, -fol) * 0.318309886;
  let faceLightAngle = (sdf.x + sdf.y) * 0.5 - abs(normalizedlightAngle);
  var faceRadiance = smoothstep(0.0, 0.1, faceLightAngle);
  faceRadiance = mix(faceRadiance, max(dot(N, eyeu.lightDir.xyz), 0.0), mask.y);

  // View attenuation
  let viewFactor = mix(1.0, max(vertNov, 0.0), eyeu.viewAttenuation * mask.x);
  faceRadiance *= viewFactor;

  // Color Ramp
  let rampCoord = vec2<f32>(clamp(faceRadiance, 0.1, 0.9), 0.5);
  let rampColor = textureSample(eyeRampTex, eyeRampSampler, rampCoord);

  var finalColor = baseColor * (faceRadiance + rampColor.rgb * 0.3);

  // Nose color
  finalColor = mix(finalColor * 0.5, finalColor, baseColor.a);

  // Lip highlight
  let highlightMask = textureSample(eyeHighlightTex, eyeHighlightSampler, vec2<f32>(in.uv.x - rol * 0.025, in.uv.y));
  let lipHighlight = highlightMask.r * faceRadiance;
  finalColor += lipHighlight;

  finalColor *= eyeu.lightDir.xyz * eyeu.lightDir.w * 0.5;

  return vec4<f32>(finalColor, 1.0);
}
