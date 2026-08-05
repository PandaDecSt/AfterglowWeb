// Shading-model registry — the single catalog of every look the renderer can
// produce. This is the cornerstone of the "add a new appearance = add one
// registry entry" rule: a new id + a GBuffer-stamping FS + a lighting BxDF,
// and nothing else changes.
//
// Every model listed here is dispatched by DeferredLightingPass via the
// ShadingModelID stamped into GBuffer material.a. Demos never branch on a look
// themselves; they only classify a material and let a preset pick the id.

import { ShadingModel } from "./shading-model";

export interface ShadingModelDef {
  id: number;
  /** Stable string key, handy for debug overlays / serialization. */
  key: string;
  name: string;
  description: string;
  /**
   * How the GBuffer's material.rg channels are interpreted by this model's
   * BxDF. The deferred lighting pass reads material.r / material.g with a
   * different meaning per model, so the writer needs this table to pack them.
   */
  packMeaning: [string, string];
  /** Default (material.r, material.g) when the demo has nothing better. */
  defaultPack: [number, number];
  /** "unlit" models ignore lights entirely (debug views, outline fill). */
  unlit?: boolean;
}

// Toon sub-variants used by the preset system (the Endfield-style look that
// used to live inside glb-viewer's forward fragment shader).
export const TOON_BODY = 5;
export const TOON_FACE = 6;
export const TOON_HAIR = 7;
export const TOON_EYE = 8;
export const TOON_EYELASH = 9;
// Debug / utility models.
export const NORMAL_DEBUG = 10;
export const UNLIT = 11;

export const SHADING_MODELS: ShadingModelDef[] = [
  {
    id: ShadingModel.STANDARD, key: "STANDARD", name: "Standard PBR",
    description: "GGX + IBL", packMeaning: ["metallic", "roughness"], defaultPack: [0.0, 0.5],
  },
  {
    id: ShadingModel.TOON, key: "TOON", name: "Toon / Cel",
    description: "Quantized diffuse + crisp specular", packMeaning: ["unused", "bandCount"], defaultPack: [0.0, 4.0],
  },
  {
    id: ShadingModel.SKIN, key: "SKIN", name: "Skin (SSS)",
    description: "Wrap diffuse + subsurface transmission", packMeaning: ["sssStrength", "roughness"], defaultPack: [0.7, 0.45],
  },
  {
    id: ShadingModel.HAIR, key: "HAIR", name: "Hair (Aniso)",
    description: "Kajiya-Kay dual spec", packMeaning: ["roughness", "aniso"], defaultPack: [0.35, 1.0],
  },
  {
    id: ShadingModel.EYE, key: "EYE", name: "Eye",
    description: "Cornea highlight + iris", packMeaning: ["cornea", "irisDark"], defaultPack: [0.6, 0.5],
  },
  {
    id: TOON_BODY, key: "TOON_BODY", name: "Toon Body (Endfield)",
    description: "Endfield-style toon ramp + rim", packMeaning: ["rimWidth", "unused"], defaultPack: [0.12, 0.0],
  },
  {
    id: TOON_FACE, key: "TOON_FACE", name: "Toon Face (SDF)",
    description: "Hard face shadow terminator", packMeaning: ["rimWidth", "unused"], defaultPack: [0.08, 0.0],
  },
  {
    id: TOON_HAIR, key: "TOON_HAIR", name: "Toon Hair (Aniso)",
    description: "Anisotropic toon hair streak", packMeaning: ["rimWidth", "unused"], defaultPack: [0.10, 0.0],
  },
  {
    id: TOON_EYE, key: "TOON_EYE", name: "Toon Eye/Iris",
    description: "Sharp toon eye specular", packMeaning: ["unused", "unused"], defaultPack: [0.0, 0.0],
  },
  {
    id: TOON_EYELASH, key: "TOON_EYELASH", name: "Toon Eyelash",
    description: "Flat toon eyelash", packMeaning: ["unused", "unused"], defaultPack: [0.0, 0.0],
  },
  {
    id: NORMAL_DEBUG, key: "NORMAL_DEBUG", name: "Normal View (Debug)",
    description: "World-space normal as color", packMeaning: ["unused", "unused"], defaultPack: [0.0, 0.0], unlit: true,
  },
  {
    id: UNLIT, key: "UNLIT", name: "Unlit / Flat",
    description: "Albedo straight through (outline fill)", packMeaning: ["unused", "unused"], defaultPack: [0.0, 0.0], unlit: true,
  },
];

const BY_ID = new Map<number, ShadingModelDef>(SHADING_MODELS.map(m => [m.id, m]));

export function getShadingModel(id: number): ShadingModelDef | undefined {
  return BY_ID.get(id);
}

export function shadingModelName(id: number): string {
  return BY_ID.get(id)?.name ?? `Unknown(${id})`;
}

/**
 * Pack the GBuffer material.rg pair for a given shading model.
 *
 * PBR-ish models want the glTF metallic/roughness the asset shipped with;
 * stylized models want their own authored constants. Keeping this decision on
 * the CPU means the GBuffer fragment shader stays a dumb writer with no
 * per-model branching (and therefore no uniform-control-flow hazards).
 */
export function packForModel(id: number, metallic: number, roughness: number): [number, number] {
  const def = BY_ID.get(id);
  switch (id) {
    case ShadingModel.STANDARD:
      return [metallic, Math.max(roughness, 0.04)];
    case ShadingModel.SKIN:
      return [def?.defaultPack[0] ?? 0.7, Math.max(roughness, 0.2)];
    case ShadingModel.HAIR:
      return [Math.max(roughness, 0.1), def?.defaultPack[1] ?? 1.0];
    default:
      return def ? [...def.defaultPack] : [metallic, roughness];
  }
}

/** name → id map, in the shape lil-gui wants for a dropdown. */
export function shadingModelOptions(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of SHADING_MODELS) out[m.name] = m.id;
  return out;
}
