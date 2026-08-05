// Render-preset data layer — the "non-programmer entry" to the renderer.
//
// This module is deliberately demo-independent. It knows how to (a) classify a
// material by name into a coarse category (body / face / hair / eye / ...) and
// (b) map that category to a ShadingModelID via a preset. The demo only feeds
// material names in and reads a mapping out; it never hard-codes shading logic.
//
// A preset IS the rendering scheme. Swapping "Endfield Default" for
// "Realistic PBR" re-skins the whole character without touching a line of
// shader code, which is the entire point of the kernel/shell split.

import { ShadingModel } from "../core/shading-model";
import {
  TOON_BODY, TOON_FACE, TOON_HAIR, TOON_EYE, TOON_EYELASH, NORMAL_DEBUG,
  shadingModelName,
} from "../core/shading-registry";

// === Material Classification ===
export type MaterialCategory = "body" | "face" | "hair" | "eye" | "eyelash" | "other";

export function classifyMaterial(name: string): MaterialCategory {
  const n = name.toLowerCase();
  if (/face|脸|面部/.test(n)) return "face";
  if (/hair|头发|发/.test(n)) return "hair";
  if (/eyelash|睫毛/.test(n)) return "eyelash";
  if (/eye|iris|眼|瞳孔|schlera/.test(n)) return "eye";
  if (/body|dress|tights|nude|neck|hat|衣|身|裙|帽|颈/.test(n)) return "body";
  return "other";
}

// === Preset System ===
/** Bump when the meaning of `mapping` values changes (see migratePreset). */
export const PRESET_VERSION = 2;

export interface RenderPreset {
  name: string;
  /** category → ShadingModelID (see core/shading-registry). */
  mapping: Record<MaterialCategory, number>;
  /** Absent / 1 = legacy forward-mode ids, needs migration. */
  version?: number;
}

export const PRESETS_KEY = "afterglow-glb-presets";

// v1 presets stored the glb-viewer forward shader's `mode` int, whose numbering
// collides with the new ShadingModelIDs (old 5 = eyelash, new 5 = toon body).
// Translate rather than silently mis-shade someone's saved scheme.
const V1_MODE_TO_SHADING_MODEL: Record<number, number> = {
  0: ShadingModel.STANDARD,
  1: TOON_BODY,
  2: TOON_FACE,
  3: TOON_HAIR,
  4: TOON_EYE,
  5: TOON_EYELASH,
  6: NORMAL_DEBUG,
};

export function migratePreset(preset: RenderPreset): RenderPreset {
  if ((preset.version ?? 1) >= PRESET_VERSION) return preset;
  const mapping = {} as Record<MaterialCategory, number>;
  for (const [cat, mode] of Object.entries(preset.mapping)) {
    mapping[cat as MaterialCategory] = V1_MODE_TO_SHADING_MODEL[mode] ?? ShadingModel.STANDARD;
  }
  return { name: preset.name, mapping, version: PRESET_VERSION };
}

export function loadPresets(): RenderPreset[] {
  try {
    const raw = localStorage.getItem(PRESETS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return mergeBuiltinPresets(parsed.map(migratePreset));
    }
  } catch { /* ignore */ }
  return builtinPresets();
}

export function savePresets(presets: RenderPreset[]) {
  localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
}

export function defaultPreset(): RenderPreset {
  return {
    name: "Endfield Default",
    version: PRESET_VERSION,
    mapping: {
      body: TOON_BODY, face: TOON_FACE, hair: TOON_HAIR,
      eye: TOON_EYE, eyelash: TOON_EYELASH, other: TOON_BODY,
    },
  };
}

// Built-in rendering schemes. These are the "north star" in miniature: the same
// mesh, the same lights, the same deferred pass — only the category→model table
// changes, and the character goes from anime to photoreal.
export function builtinPresets(): RenderPreset[] {
  const P = PRESET_VERSION;
  return [
    defaultPreset(),
    // 写实基础（剑星风格的 PBR 底子）：全身走标准 PBR，无卡通色阶。
    {
      name: "Realistic PBR", version: P,
      mapping: {
        body: ShadingModel.STANDARD, face: ShadingModel.STANDARD, hair: ShadingModel.STANDARD,
        eye: ShadingModel.STANDARD, eyelash: ShadingModel.STANDARD, other: ShadingModel.STANDARD,
      },
    },
    // 写实角色专业档：皮肤走次表面散射、头发走各向异性、眼睛走角膜高光。
    {
      name: "Realistic Character (SSS)", version: P,
      mapping: {
        body: ShadingModel.SKIN, face: ShadingModel.SKIN, hair: ShadingModel.HAIR,
        eye: ShadingModel.EYE, eyelash: ShadingModel.STANDARD, other: ShadingModel.STANDARD,
      },
    },
    // 混合档：写实皮肤 + 风格化眼发，验证同一场景内两种审美共存。
    {
      name: "Realistic Skin · Stylized Hair", version: P,
      mapping: {
        body: ShadingModel.SKIN, face: ShadingModel.SKIN, hair: TOON_HAIR,
        eye: TOON_EYE, eyelash: TOON_EYELASH, other: ShadingModel.STANDARD,
      },
    },
    // 通用色阶卡通（非 Endfield 风）：走带 bandCount 的经典 cel。
    {
      name: "Classic Cel", version: P,
      mapping: {
        body: ShadingModel.TOON, face: ShadingModel.TOON, hair: ShadingModel.TOON,
        eye: ShadingModel.TOON, eyelash: ShadingModel.TOON, other: ShadingModel.TOON,
      },
    },
    // 法线调试：全部显示世界空间法线。
    {
      name: "Normal View (Debug)", version: P,
      mapping: {
        body: NORMAL_DEBUG, face: NORMAL_DEBUG, hair: NORMAL_DEBUG,
        eye: NORMAL_DEBUG, eyelash: NORMAL_DEBUG, other: NORMAL_DEBUG,
      },
    },
  ];
}

export function mergeBuiltinPresets(stored: RenderPreset[]): RenderPreset[] {
  const names = new Set(stored.map(p => p.name));
  for (const b of builtinPresets()) {
    if (!names.has(b.name)) stored.push(b);
  }
  return stored;
}

/** Display label for a mapping value, e.g. for an inspector row. */
export function presetModelLabel(id: number): string {
  return shadingModelName(id);
}
