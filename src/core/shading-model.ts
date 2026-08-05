// Shading Model IDs — the heart of the "all-powerful renderer" architecture.
//
// Every opaque pixel in the GBuffer carries its shading model id in the
// `.a` (w) channel of the material texture (`material.a`). The deferred
// lighting pass reads that id and dispatches to the correct BxDF — exactly
// like Unreal Engine's `ShadingModelID` GBuffer channel. This is what lets a
// photorealistic PBR character and a toon (cel-shaded) MMD character share
// ONE scene, ONE lighting pass, ONE set of shadows.
//
// The ids below are the canonical set. Adding a new look (e.g. hair, eye,
// skin) is a matter of: (1) reserve a new id here, (2) write the GBuffer FS
// that stamps it, (3) add a branch in the lighting pass. No new pipeline.

export const ShadingModel = {
  /** Physically based (GGX + IBL). Used for Stellar-Blade-style realism. */
  STANDARD: 0,
  /** Cel / anime shading: quantized diffuse bands + crisp specular. */
  TOON: 1,
  /** Reserved (Step 2): subsurface-scattering skin. */
  SKIN: 2,
  /** Reserved (Step 2): anisotropic dual-spec hair. */
  HAIR: 3,
  /** Reserved (Step 2): cornea/iris eye. */
  EYE: 4,
} as const;

export type ShadingModelId = (typeof ShadingModel)[keyof typeof ShadingModel];

/** Human-readable names, handy for debug overlays / GUI. */
export const SHADING_MODEL_NAMES: Record<number, string> = {
  [ShadingModel.STANDARD]: "Standard PBR",
  [ShadingModel.TOON]: "Toon / Cel",
  [ShadingModel.SKIN]: "Skin (SSS)",
  [ShadingModel.HAIR]: "Hair (Aniso)",
  [ShadingModel.EYE]: "Eye",
};
