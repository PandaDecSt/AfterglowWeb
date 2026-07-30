import type { ShaderGraph } from "./schema";

export const PRINCIPLED_GRAPH: ShaderGraph = {
  version: 1,
  name: "Principled BSDF",
  nodes: [
    { id: "tex", type: "texture", params: {} },
    { id: "base_const", type: "constant/color", params: { r: 0.8, g: 0.8, b: 0.8 } },
    { id: "mix_base", type: "mix/multiply", params: {} },
    { id: "pbr", type: "principled", params: { metallic: 0.0, roughness: 0.5, specular: 0.5 } },
    { id: "output", type: "output", params: {} },
  ],
  links: [
    { fromNode: "tex", fromSocket: "color", toNode: "mix_base", toSocket: "a" },
    { fromNode: "base_const", fromSocket: "color", toNode: "mix_base", toSocket: "b" },
    { fromNode: "mix_base", fromSocket: "color", toNode: "pbr", toSocket: "base_color" },
    { fromNode: "pbr", fromSocket: "color", toNode: "output", toSocket: "color" },
  ],
  output: { node: "output", socket: "color" },
};

export const TOON_GRAPH: ShaderGraph = {
  version: 1,
  name: "Toon Ramp",
  nodes: [
    { id: "tex", type: "texture", params: {} },
    { id: "base_const", type: "constant/color", params: { r: 0.8, g: 0.8, b: 0.8 } },
    { id: "mix_base", type: "mix/multiply", params: {} },
    { id: "toon", type: "toon/ramp", params: { ramp_threshold: 0.5, ramp_smooth: 0.05 } },
    { id: "output", type: "output", params: {} },
  ],
  links: [
    { fromNode: "tex", fromSocket: "color", toNode: "mix_base", toSocket: "a" },
    { fromNode: "base_const", fromSocket: "color", toNode: "mix_base", toSocket: "b" },
    { fromNode: "mix_base", fromSocket: "color", toNode: "toon", toSocket: "base_color" },
    { fromNode: "toon", fromSocket: "color", toNode: "output", toSocket: "color" },
  ],
  output: { node: "output", socket: "color" },
};

export const EMISSIVE_PULSE_GRAPH: ShaderGraph = {
  version: 1,
  name: "Emissive Pulse",
  nodes: [
    { id: "tex", type: "texture", params: {} },
    { id: "pbr", type: "principled", params: { metallic: 0.0, roughness: 0.5 } },
    { id: "emissive", type: "emissive/pulse", params: { speed: 2.0, intensity: 3.0 } },
    { id: "blend", type: "mix/blend", params: {} },
    { id: "output", type: "output", params: {} },
  ],
  links: [
    { fromNode: "tex", fromSocket: "color", toNode: "pbr", toSocket: "base_color" },
    { fromNode: "tex", fromSocket: "color", toNode: "emissive", toSocket: "color" },
    { fromNode: "pbr", fromSocket: "color", toNode: "blend", toSocket: "a" },
    { fromNode: "emissive", fromSocket: "emissive", toNode: "blend", toSocket: "b" },
    { fromNode: "blend", fromSocket: "color", toNode: "output", toSocket: "color" },
  ],
  output: { node: "output", socket: "color" },
  params: [
    { name: "pulse_speed", type: "float", default: 2.0, min: 0.0, max: 10.0, step: 0.1 },
    { name: "pulse_intensity", type: "float", default: 3.0, min: 0.0, max: 10.0, step: 0.1 },
  ],
};

export const METALLIC_GRAPH: ShaderGraph = {
  version: 1,
  name: "Metallic",
  nodes: [
    { id: "tex", type: "texture", params: {} },
    { id: "pbr", type: "principled", params: { metallic: 1.0, roughness: 0.2, specular: 1.0 } },
    { id: "output", type: "output", params: {} },
  ],
  links: [
    { fromNode: "tex", fromSocket: "color", toNode: "pbr", toSocket: "base_color" },
    { fromNode: "pbr", fromSocket: "color", toNode: "output", toSocket: "color" },
  ],
  output: { node: "output", socket: "color" },
};

export const PRESETS: Record<string, ShaderGraph> = {
  "principled": PRINCIPLED_GRAPH,
  "toon": TOON_GRAPH,
  "emissive-pulse": EMISSIVE_PULSE_GRAPH,
  "metallic": METALLIC_GRAPH,
};