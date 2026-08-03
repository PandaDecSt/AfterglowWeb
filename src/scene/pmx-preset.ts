export type RenderClass = "auto" | "eye" | "hair";

export interface PresetConfig {
  metallic: number;
  roughness: number;
  emissionStrength: number;
  nprMix: number;
  rimColor: [number, number, number];
  rimStrength: number;
  rimPower: number;
  alphaMode?: number;
  renderClass: RenderClass;
}

export const PRESETS: Record<string, PresetConfig> = {
  default:      { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.0, rimColor: [1, 1, 1], rimStrength: 0.0, rimPower: 3.0, renderClass: "auto" },
  body:         { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.5, rimColor: [1, 0.85, 0.7], rimStrength: 0.3, rimPower: 3.0, renderClass: "auto" },
  face:         { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.5, rimColor: [1, 0.9, 0.8], rimStrength: 0.2, rimPower: 4.0, renderClass: "auto" },
  hair:         { metallic: 0.0, roughness: 0.3, emissionStrength: 0.0, nprMix: 0.2, rimColor: [1, 1, 1], rimStrength: 0.4, rimPower: 2.5, renderClass: "hair" },
  eye:          { metallic: 0.0, roughness: 0.1, emissionStrength: 1.5, nprMix: 0.0, rimColor: [1, 1, 1], rimStrength: 0.0, rimPower: 3.0, renderClass: "eye" },
  eyelash:      { metallic: 0.0, roughness: 0.5, emissionStrength: 0.0, nprMix: 0.3, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 3.0, renderClass: "eye" },
  metal:        { metallic: 1.0, roughness: 0.3, emissionStrength: 0.0, nprMix: 0.3, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 5.0, renderClass: "auto" },
  stockings:    { metallic: 0.0, roughness: 0.8, emissionStrength: 0.0, nprMix: 0.0, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 3.0, alphaMode: 1, renderClass: "auto" },
  cloth_smooth: { metallic: 0.0, roughness: 0.6, emissionStrength: 0.0, nprMix: 0.1, rimColor: [1, 1, 1], rimStrength: 0.15, rimPower: 3.0, renderClass: "auto" },
  cloth_rough:  { metallic: 0.0, roughness: 0.82, emissionStrength: 0.0, nprMix: 0.1, rimColor: [1, 1, 1], rimStrength: 0.1, rimPower: 3.5, renderClass: "auto" },
};

export function detectPreset(name: string, isTransparent: boolean): PresetConfig {
  const n = name.toLowerCase();
  if (n.includes("顔") || n.includes("面") || n.includes("face")) return PRESETS.face;
  if (n.includes("睫") || n.includes("まつげ") || n.includes("まつ毛") || n.includes("eyelash")) return PRESETS.eyelash;
  if (n.includes("髪") || n.includes("毛") || n.includes("hair")) return PRESETS.hair;
  if (n.includes("目") || n.includes("眼") || n.includes("eye") || n.includes("瞳")) return PRESETS.eye;
  if (n.includes("金属") || n.includes("metal") || n.includes("メタル")) return PRESETS.metal;
  if (n.includes("ストッキング") || n.includes("靴下") || n.includes("stocking") || n.includes("ニーソ")) return PRESETS.stockings;
  if (n.includes("服") || n.includes("衣") || n.includes("cloth") || n.includes("シャツ") || n.includes("スカート")) return PRESETS.cloth_smooth;
  if (n.includes("肌") || n.includes("体") || n.includes("body") || n.includes("skin")) return PRESETS.body;
  if (isTransparent) return PRESETS.stockings;
  return PRESETS.default;
}