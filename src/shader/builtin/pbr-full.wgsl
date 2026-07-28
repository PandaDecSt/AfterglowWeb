// AfterglowRender PBR Module - Full WGSL Implementation
// Ported from BxDF.hlsl, ShadingModels.hlsl, ShadingEnergyConservation.hlsl

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 1.0e-10;

// ============================================================
// BxDF Context
// ============================================================

struct BxDFContext {
  nov: f32,   // dot(normal, view)
  nol: f32,   // dot(normal, light)
  vol: f32,   // dot(view, light)
  noh: f32,   // dot(normal, half)
  voh: f32,   // dot(view, halfVec)
  xov: f32,   // dot(tangent, view)
  xol: f32,   // dot(tangent, light)
  xoh: f32,   // dot(tangent, half)
  yov: f32,   // dot(bitangent, view)
  yol: f32,   // dot(bitangent, light)
  yoh: f32,   // dot(bitangent, half)
};

fn initBxDFContext_nol_nov(context: ptr<function, BxDFContext>, normal: vec3<f32>, view: vec3<f32>, light: vec3<f32>) {
  (*context).nol = dot(normal, light);
  (*context).nov = dot(normal, view);
  (*context).vol = clamp(dot(view, light), -1.0, 1.0);
  let invLenH = inverseSqrt(2.0 + 2.0 * (*context).vol);
  (*context).noh = clamp((*context).nol + (*context).nov) * invLenH, 0.0, 1.0);
  (*context).voh = clamp(invLenH + invLenH * (*context).vol, 0.0, 1.0);

  (*context).xov = 0.0;
  (*context).xol = 0.0;
  (*context).xoh = 0.0;
  (*context).yov = 0.0;
  (*context).yol = 0.0;
  (*context).yoh = 0.0;
}

fn initBxDFContext_tangent(
  context: ptr<function, BxDFContext>,
  normal: vec3<f32>, tangent: vec3<f32>, bitangent: vec3<f32>,
  view: vec3<f32>, light: vec3<f32>
) {
  (*context).nol = dot(normal, light);
  (*context).nov = dot(normal, view);
  (*context).vol = clamp(dot(view, light), -1.0, 1.0);
  let invLenH = inverseSqrt(2.0 + 2.0 * (*context).vol);
  (*context).noh = clamp(((*context).nol + (*context).nov) * invLenH, 0.0, 1.0);
  (*context).voh = clamp(invLenH + invLenH * (*context).vol, 0.0, 1.0);

  (*context).xov = dot(tangent, view);
  (*context).xol = dot(tangent, light);
  (*context).xoh = ((*context).xov + (*context).xol) * invLenH;
  (*context).yov = dot(bitangent, view);
  (*context).yol = dot(bitangent, light);
  (*context).yoh = ((*context).yov + (*context).yol) * invLenH;

  // Re-normalize to prevent unexpected cusps from GGXAnisotropic
  let hTangent = normalize(vec3<f32>((*context).xoh, (*context).yoh, (*context).noh));
  (*context).xoh = hTangent.x;
  (*context).yoh = hTangent.y;
  (*context).noh = hTangent.z;
}

// ============================================================
// Diffuse Models
// ============================================================

fn diffuseLambert(diffuseColor: vec3<f32>) -> vec3<f32> {
  return diffuseColor * INV_PI;
}

// [Burley 2012, "Physically-Based Shading at Disney"]
fn diffuseBurley(diffuseColor: vec3<f32>, roughness: f32, nov: f32, nol: f32, voh: f32) -> vec3<f32> {
  let fd90 = 0.5 + 2.0 * voh * voh * roughness;
  let fdv = 1.0 + (fd90 - 1.0) * pow(1.0 - nov, 5.0);
  let fdl = 1.0 + (fd90 - 1.0) * pow(1.0 - nol, 5.0);
  return diffuseColor * (INV_PI * fdv * fdl);
}

// ============================================================
// Distribution Functions
// ============================================================

// [Walter et al. 2007]
fn distributionGGX(a2: f32, noh: f32) -> f32 {
  let d = (noh * a2 - noh) * noh + 1.0;
  return a2 / (PI * d * d + EPSILON);
}

// [Burley 2012]
fn distributionAnisotropicGGX(ax: f32, ay: f32, noh: f32, xoh: f32, yoh: f32) -> f32 {
  let a2 = ax * ay;
  let v = vec3<f32>(ay * xoh, ax * yoh, a2 * noh);
  let s = max(dot(v, v), EPSILON);
  return INV_PI * a2 * pow(a2 / s, 2.0);
}

// ============================================================
// Visibility (Geometric Shadowing)
// ============================================================

fn visibilitySchlick(a2: f32, nov: f32, nol: f32) -> f32 {
  let k = sqrt(a2) * 0.5;
  let schlickV = nov * (1.0 - k) + k;
  let schlickL = nol * (1.0 - k) + k;
  return 0.25 / (schlickV * schlickL);
}

fn visibilitySmithJointApprox(a2: f32, nov: f32, nol: f32) -> f32 {
  let a = sqrt(a2);
  let smithV = nol * (nov * (1.0 - a) + a);
  let smithL = nov * (nol * (1.0 - a) + a);
  return 0.5 / (smithV + smithL + EPSILON);
}

fn visibilitySmithJoint(a2: f32, nov: f32, nol: f32) -> f32 {
  let smithV = nol * sqrt(nov * (nov - nov * a2) + a2);
  let smithL = nov * sqrt(nol * (nol - nol * a2) + a2);
  return 0.5 / (smithV + smithL + EPSILON);
}

fn visibilityAnisotropicSmithJoint(
  ax: f32, ay: f32,
  nov: f32, nol: f32,
  xov: f32, xol: f32,
  yov: f32, yol: f32
) -> f32 {
  let smithV = nol * length(vec3<f32>(ax * xov, ay * yov, nov));
  let smithL = nov * length(vec3<f32>(ax * xol, ay * yol, nol));
  return 0.5 / (smithV + smithL + EPSILON);
}

// ============================================================
// Fresnel
// ============================================================

fn fresnelSchlick(f0: vec3<f32>, voh: f32) -> vec3<f32> {
  let fc = pow(1.0 - voh, 5.0);
  return saturate(50.0 * f0.g) * fc + (1.0 - fc) * f0;
}

fn fresnelSchlickF90(f0: vec3<f32>, f90: vec3<f32>, voh: f32) -> vec3<f32> {
  let fc = pow(1.0 - voh, 5.0);
  return f90 * fc + (1.0 - fc) * f0;
}

// ============================================================
// Anisotropic Roughness
// ============================================================

fn anisotropicSquaredRoughness(alpha: f32, anisotropy: f32) -> vec2<f32> {
  let ax = max(alpha * (1.0 + anisotropy), 0.001);
  let ay = max(alpha * (1.0 - anisotropy), 0.001);
  return vec2<f32>(ax, ay);
}

fn anisotropicRoughness(roughness: f32, anisotropy: f32) -> vec2<f32> {
  let r = saturate(roughness);
  let a = clamp(anisotropy, -1.0, 1.0);
  let ax = max(r * sqrt(1.0 + a), 0.001);
  let ay = max(r * sqrt(1.0 - a), 0.001);
  return vec2<f32>(ax, ay);
}

// ============================================================
// EnvBRDF Approximation
// ============================================================

// [Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II"]
fn envBRDFApproxLazarov(roughness: f32, nov: f32) -> vec2<f32> {
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
  let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
  let r = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}

fn envBRDFApprox(specularColor: vec3<f32>, roughness: f32, nov: f32) -> vec3<f32> {
  let ab = envBRDFApproxLazarov(roughness, nov);
  let f90 = saturate(50.0 * specularColor.g);
  return specularColor * ab.x + f90 * ab.y;
}

fn envBRDFApproxF90(f0: vec3<f32>, f90: vec3<f32>, roughness: f32, nov: f32) -> vec3<f32> {
  let ab = envBRDFApproxLazarov(roughness, nov);
  return f0 * ab.x + f90 * ab.y;
}

// ============================================================
// Energy Conservation
// ============================================================

struct BxDFEnergyTerms {
  w: vec3<f32>,  // Overall weight for energy conservation
  e: vec3<f32>,  // Directional albedo for energy preservation
};

fn ggxEnergyLookup(roughness: f32, nov: f32) -> vec2<f32> {
  let energy = 1.0 - saturate(pow(roughness, nov / roughness) * ((roughness * nov + 0.0266916) / (0.466495 + nov)));
  let energyFresnel = pow(1.0 - nov, 5.0) * pow(2.36651 * pow(nov, max(4.7703 * roughness, 1.0e-12)) + 0.0387332, roughness);
  return vec2<f32>(energy, energyFresnel);
}

fn f0RGBToMicroOcclusion(f0: vec3<f32>) -> f32 {
  return saturate(50.0 * f0.g);
}

fn computeGGXSpecularEnergyTerms(roughness: f32, nov: f32, f0: vec3<f32>) -> BxDFEnergyTerms {
  let energy = ggxEnergyLookup(roughness, nov);
  let f90 = vec3<f32>(f0RGBToMicroOcclusion(f0));
  var result: BxDFEnergyTerms;
  result.w = vec3<f32>(1.0) + f0 * ((1.0 - energy.x) / energy.x);
  result.e = result.w * (energy.x * f0 + energy.y * (f90 - f0));
  return result;
}

// Returns energy absorbed by upper layer (for specular -> diffuse attenuation)
fn computeEnergyPreservation(energyTerms: BxDFEnergyTerms) -> vec3<f32> {
  return vec3<f32>(1.0) - energyTerms.e;
}

// Returns energy conservation weight for multiple scattering
fn computeEnergyConservation(energyTerms: BxDFEnergyTerms) -> vec3<f32> {
  return energyTerms.w;
}

// ============================================================
// Subsurface Scattering Utilities
// ============================================================

fn transmittanceToExtinction(transmittanceColor: vec3<f32>, thicknessMeters: f32) -> vec3<f32> {
  let minTransmittance = vec3<f32>(1.0e-12);
  let minMFP = 1.0e-12;
  return -log(clamp(transmittanceColor, minTransmittance, vec3<f32>(1.0))) / max(minMFP, thicknessMeters);
}

fn extinctionToTransmittance(extinction: vec3<f32>, thicknessMeters: f32) -> vec3<f32> {
  return exp(-extinction * thicknessMeters);
}

// ============================================================
// Specular GGX (Isotropic)
// ============================================================

fn specularGGX_isotropic(ctx: BxDFContext, roughness: f32, specularColor: vec3<f32>) -> vec3<f32> {
  let a2 = pow(roughness, 4.0);
  let distribution = distributionGGX(a2, ctx.noh);
  let visibility = visibilitySmithJointApprox(a2, ctx.nov, ctx.nol);
  let fresnel = fresnelSchlick(specularColor, ctx.voh);
  return distribution * visibility * fresnel;
}

// ============================================================
// Specular GGX (Anisotropic)
// ============================================================

fn specularGGX_anisotropic(
  ctx: BxDFContext,
  roughness: f32, anisotropy: f32,
  specularColor: vec3<f32>
) -> vec3<f32> {
  let alpha = roughness * roughness;
  let anisotropicAlpha = anisotropicSquaredRoughness(alpha, anisotropy);

  let distribution = distributionAnisotropicGGX(
    anisotropicAlpha.x, anisotropicAlpha.y,
    ctx.noh, ctx.xoh, ctx.yoh
  );

  let visibility = visibilityAnisotropicSmithJoint(
    anisotropicAlpha.x, anisotropicAlpha.y,
    ctx.nov, ctx.nol,
    ctx.xov, ctx.xol,
    ctx.yov, ctx.yol
  );

  let fresnel = fresnelSchlick(specularColor, ctx.voh);
  return distribution * visibility * fresnel;
}

// ============================================================
// Shading Context & Result
// ============================================================

struct ShadingContext {
  baseColor: vec3<f32>,
  metallic: f32,
  specular: f32,
  roughness: f32,
  ambientOcclusion: f32,
  anisotropy: f32,
  normal: vec3<f32>,
  tangent: vec3<f32>,
  bitangent: vec3<f32>,
  view: vec3<f32>,
  // Subsurface
  subsurfaceColor: vec3<f32>,
  opacity: f32,
};

struct LightingResult {
  diffuse: vec3<f32>,
  specular: vec3<f32>,
  transmission: vec3<f32>,
};

struct LightContext {
  color: vec3<f32>,
  intensity: f32,
};

// ============================================================
// Default Lit BxDF (Isotropic)
// ============================================================

fn defaultLitBxDF(shadingContext: ShadingContext, lightDirection: vec3<f32>, specularColor: vec3<f32>) -> LightingResult {
  var result: LightingResult;
  result.transmission = vec3<f32>(0.0);

  var ctx: BxDFContext;
  initBxDFContext_nol_nov(&ctx, shadingContext.normal, shadingContext.view, lightDirection);

  ctx.nol = clamp(ctx.nol, 0.0, 1.0);
  ctx.nov = clamp(abs(ctx.nov) + 1.0e-6, 0.0, 1.0);

  result.diffuse = diffuseBurley(
    shadingContext.baseColor, shadingContext.roughness,
    ctx.nov, ctx.nol, ctx.voh
  ) * ctx.nol;

  result.specular = specularGGX_isotropic(ctx, shadingContext.roughness, specularColor) * ctx.nol;

  // Energy conservation
  let energyTerms = computeGGXSpecularEnergyTerms(shadingContext.roughness, ctx.nov, specularColor);
  result.diffuse *= computeEnergyPreservation(energyTerms);
  result.specular *= computeEnergyConservation(energyTerms);

  return result;
}

// ============================================================
// Default Lit BxDF (Anisotropic)
// ============================================================

fn defaultLitBxDFAnisotropic(shadingContext: ShadingContext, lightDirection: vec3<f32>, specularColor: vec3<f32>) -> LightingResult {
  var result: LightingResult;
  result.transmission = vec3<f32>(0.0);

  var ctx: BxDFContext;
  initBxDFContext_tangent(
    &ctx,
    shadingContext.normal, shadingContext.tangent, shadingContext.bitangent,
    shadingContext.view, lightDirection
  );

  ctx.nol = clamp(ctx.nol, 0.0, 1.0);
  ctx.nov = clamp(abs(ctx.nov) + 1.0e-6, 0.0, 1.0);

  result.diffuse = diffuseBurley(
    shadingContext.baseColor, shadingContext.roughness,
    ctx.nov, ctx.nol, ctx.voh
  ) * ctx.nol;

  result.specular = specularGGX_anisotropic(
    ctx, shadingContext.roughness, shadingContext.anisotropy, specularColor
  ) * ctx.nol;

  // Energy conservation
  let energyTerms = computeGGXSpecularEnergyTerms(shadingContext.roughness, ctx.nov, specularColor);
  result.diffuse *= computeEnergyPreservation(energyTerms);
  result.specular *= computeEnergyConservation(energyTerms);

  return result;
}

// ============================================================
// Subsurface BxDF (Isotropic)
// ============================================================

fn subsurfaceBxDF(shadingContext: ShadingContext, lightDirection: vec3<f32>, specularColor: vec3<f32>) -> LightingResult {
  var lighting = defaultLitBxDF(shadingContext, lightDirection, specularColor);

  // In-scatter (see-through effect)
  let inScatter = pow(saturate(dot(lightDirection, -shadingContext.view)), 12.0) * mix(3.0, 0.1, shadingContext.opacity);

  // Warped diffuse (wrap lighting)
  let warppedDiffuse = pow(saturate(dot(shadingContext.normal, lightDirection) * (1.0 / 1.5) + (0.5 / 1.5)), 1.5) * (2.5 / 1.5);
  let normalContribution = mix(1.0, warppedDiffuse, shadingContext.opacity);
  let backScatter = shadingContext.ambientOcclusion * normalContribution / (PI * 2.0);

  // Transmission
  let extinctionCoefficients = transmittanceToExtinction(shadingContext.subsurfaceColor, 0.515);
  let rawTransmittedColor = extinctionToTransmittance(extinctionCoefficients, 1.0);

  lighting.transmission = mix(vec3<f32>(backScatter), vec3<f32>(1.0), inScatter) *
    mix(rawTransmittedColor, shadingContext.subsurfaceColor, vec3<f32>(0.0));

  return lighting;
}

// ============================================================
// Subsurface BxDF (Anisotropic)
// ============================================================

fn subsurfaceBxDFAnisotropic(shadingContext: ShadingContext, lightDirection: vec3<f32>, specularColor: vec3<f32>) -> LightingResult {
  var lighting = defaultLitBxDFAnisotropic(shadingContext, lightDirection, specularColor);

  let inScatter = pow(saturate(dot(lightDirection, -shadingContext.view)), 12.0) * mix(3.0, 0.1, shadingContext.opacity);
  let warppedDiffuse = pow(saturate(dot(shadingContext.normal, lightDirection) * (1.0 / 1.5) + (0.5 / 1.5)), 1.5) * (2.5 / 1.5);
  let normalContribution = mix(1.0, warppedDiffuse, shadingContext.opacity);
  let backScatter = shadingContext.ambientOcclusion * normalContribution / (PI * 2.0);

  let extinctionCoefficients = transmittanceToExtinction(shadingContext.subsurfaceColor, 0.515);
  let rawTransmittedColor = extinctionToTransmittance(extinctionCoefficients, 1.0);

  lighting.transmission = mix(vec3<f32>(backScatter), vec3<f32>(1.0), inScatter) *
    mix(rawTransmittedColor, shadingContext.subsurfaceColor, vec3<f32>(0.0));

  return lighting;
}

// ============================================================
// Image-Based Lighting (simplified)
// ============================================================

fn envLighting(
  reflectionVector: vec3<f32>,
  baseColor: vec3<f32>,
  specularColor: vec3<f32>,
  nov: f32,
  roughness: f32,
  metallic: f32
) -> LightingResult {
  var result: LightingResult;

  // Simplified environment lighting (no texture, procedural)
  let envColor = mix(vec3<f32>(0.1, 0.15, 0.2), vec3<f32>(0.3, 0.35, 0.4), reflectionVector.y * 0.5 + 0.5);

  result.specular = envColor * envBRDFApprox(specularColor, roughness, nov);
  result.diffuse = max(1.0 - metallic, 0.0) * baseColor * 0.08;
  result.transmission = vec3<f32>(0.0);

  return result;
}

// ============================================================
// Full Shading Functions (Isotropic)
// ============================================================

fn defaultShading(shadingContext: ShadingContext) -> LightingResult {
  var lighting: LightingResult;
  lighting.diffuse = vec3<f32>(0.0);
  lighting.specular = vec3<f32>(0.0);
  lighting.transmission = vec3<f32>(0.0);

  let specularColor = mix(vec3<f32>(0.04), shadingContext.baseColor, shadingContext.metallic);

  // Directional light
  let lightDir = normalize(vec3<f32>(-0.4, -1.0, -0.3));
  let lightColor = vec3<f32>(3.0);
  let lightIntensity = 1.0;

  let directionalLighting = defaultLitBxDF(shadingContext, lightDir, specularColor);
  let lightScale = lightColor * lightIntensity;

  lighting.diffuse += directionalLighting.diffuse * lightScale;
  lighting.specular += directionalLighting.specular * lightScale;
  lighting.transmission += directionalLighting.transmission * lightScale;

  // Environment lighting
  let reflectionVector = reflect(-shadingContext.view, shadingContext.normal);
  let envLight = envLighting(
    reflectionVector, shadingContext.baseColor, specularColor,
    dot(shadingContext.normal, shadingContext.view),
    shadingContext.roughness, shadingContext.metallic
  );

  lighting.diffuse += envLight.diffuse * shadingContext.ambientOcclusion;
  lighting.specular += envLight.specular * shadingContext.ambientOcclusion;

  return lighting;
}

// ============================================================
// Full Shading Functions (Anisotropic)
// ============================================================

fn defaultShadingAnisotropic(shadingContext: ShadingContext) -> LightingResult {
  var lighting: LightingResult;
  lighting.diffuse = vec3<f32>(0.0);
  lighting.specular = vec3<f32>(0.0);
  lighting.transmission = vec3<f32>(0.0);

  let specularColor = mix(vec3<f32>(0.04), shadingContext.baseColor, shadingContext.metallic);

  let lightDir = normalize(vec3<f32>(-0.4, -1.0, -0.3));
  let lightColor = vec3<f32>(3.0);
  let lightIntensity = 1.0;

  let directionalLighting = defaultLitBxDFAnisotropic(shadingContext, lightDir, specularColor);
  let lightScale = lightColor * lightIntensity;

  lighting.diffuse += directionalLighting.diffuse * lightScale;
  lighting.specular += directionalLighting.specular * lightScale;
  lighting.transmission += directionalLighting.transmission * lightScale;

  let reflectionVector = reflect(-shadingContext.view, shadingContext.normal);
  let envLight = envLighting(
    reflectionVector, shadingContext.baseColor, specularColor,
    dot(shadingContext.normal, shadingContext.view),
    shadingContext.roughness, shadingContext.metallic
  );

  lighting.diffuse += envLight.diffuse * shadingContext.ambientOcclusion;
  lighting.specular += envLight.specular * shadingContext.ambientOcclusion;

  return lighting;
}

// ============================================================
// Full Shading Functions (Subsurface Isotropic)
// ============================================================

fn subsurfaceShading(shadingContext: ShadingContext) -> LightingResult {
  var lighting: LightingResult;
  lighting.diffuse = vec3<f32>(0.0);
  lighting.specular = vec3<f32>(0.0);
  lighting.transmission = vec3<f32>(0.0);

  let specularColor = mix(vec3<f32>(0.04), shadingContext.baseColor, shadingContext.metallic);

  let lightDir = normalize(vec3<f32>(-0.4, -1.0, -0.3));
  let lightColor = vec3<f32>(3.0);
  let lightIntensity = 1.0;

  let directionalLighting = subsurfaceBxDF(shadingContext, lightDir, specularColor);
  let lightScale = lightColor * lightIntensity;

  lighting.diffuse += directionalLighting.diffuse * lightScale;
  lighting.specular += directionalLighting.specular * lightScale;
  lighting.transmission += directionalLighting.transmission * lightScale;

  let reflectionVector = reflect(-shadingContext.view, shadingContext.normal);
  let envLight = envLighting(
    reflectionVector, shadingContext.baseColor, specularColor,
    dot(shadingContext.normal, shadingContext.view),
    shadingContext.roughness, shadingContext.metallic
  );

  lighting.diffuse += envLight.diffuse * shadingContext.ambientOcclusion;
  lighting.specular += envLight.specular * shadingContext.ambientOcclusion;

  return lighting;
}

// ============================================================
// Full Shading Functions (Subsurface Anisotropic)
// ============================================================

fn subsurfaceShadingAnisotropic(shadingContext: ShadingContext) -> LightingResult {
  var lighting: LightingResult;
  lighting.diffuse = vec3<f32>(0.0);
  lighting.specular = vec3<f32>(0.0);
  lighting.transmission = vec3<f32>(0.0);

  let specularColor = mix(vec3<f32>(0.04), shadingContext.baseColor, shadingContext.metallic);

  let lightDir = normalize(vec3<f32>(-0.4, -1.0, -0.3));
  let lightColor = vec3<f32>(3.0);
  let lightIntensity = 1.0;

  let directionalLighting = subsurfaceBxDFAnisotropic(shadingContext, lightDir, specularColor);
  let lightScale = lightColor * lightIntensity;

  lighting.diffuse += directionalLighting.diffuse * lightScale;
  lighting.specular += directionalLighting.specular * lightScale;
  lighting.transmission += directionalLighting.transmission * lightScale;

  let reflectionVector = reflect(-shadingContext.view, shadingContext.normal);
  let envLight = envLighting(
    reflectionVector, shadingContext.baseColor, specularColor,
    dot(shadingContext.normal, shadingContext.view),
    shadingContext.roughness, shadingContext.metallic
  );

  lighting.diffuse += envLight.diffuse * shadingContext.ambientOcclusion;
  lighting.specular += envLight.specular * shadingContext.ambientOcclusion;

  return lighting;
}

// ============================================================
// Utility Functions
// ============================================================

fn computeF0(specular: f32, baseColor: vec3<f32>, metallic: f32) -> vec3<f32> {
  return mix(vec3<f32>(specular), baseColor, metallic);
}

fn luminance(v: vec3<f32>) -> f32 {
  return dot(v, vec3<f32>(0.2126, 0.7152, 0.0722));
}
