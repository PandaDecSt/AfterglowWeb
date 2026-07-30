import type { NodeDecl, SocketType } from "./schema";

const registry = new Map<string, NodeDecl>();

export function registerNode(decl: NodeDecl): void {
  registry.set(decl.type, decl);
}

export function getNode(type: string): NodeDecl | undefined {
  return registry.get(type);
}

export function getAllNodeTypes(): string[] {
  return [...registry.keys()];
}

function wgslType(t: SocketType): string {
  switch (t) {
    case "float": return "f32";
    case "vec2": return "vec2<f32>";
    case "vec3": return "vec3<f32>";
    case "vec4": return "vec4<f32>";
    case "color": return "vec3<f32>";
    case "normal": return "vec3<f32>";
    case "bool": return "bool";
    default: return "f32";
  }
}

export { wgslType };

registerNode({
  type: "output",
  inputs: [
    { name: "color", type: "color" },
    { name: "alpha", type: "float", default: 1.0 },
  ],
  outputs: [],
  emit: (inputs) => `return vec4<f32>(${inputs.color}, ${inputs.alpha});`,
});

registerNode({
  type: "principled",
  inputs: [
    { name: "base_color", type: "color", default: 0.8 },
    { name: "metallic", type: "float", default: 0.0 },
    { name: "roughness", type: "float", default: 0.5 },
    { name: "specular", type: "float", default: 0.5 },
    { name: "emissive", type: "color", default: 0.0 },
    { name: "emissive_strength", type: "float", default: 1.0 },
    { name: "alpha", type: "float", default: 1.0 },
    { name: "normal", type: "normal" },
    { name: "ao", type: "float", default: 1.0 },
  ],
  outputs: [
    { name: "color", type: "color" },
  ],
  emit: (inputs, node) => {
    const id = node.id.replace(/[^a-zA-Z0-9]/g, "_");
    return [
      `var ${id}_F0 = mix(vec3<f32>(0.04), ${inputs.base_color}, ${inputs.metallic});`,
      `let ${id}_H = normalize(u.cameraPosition.xyz - in.worldPos + u.lightDir.xyz);`,
      `let ${id}_NdotV = max(dot(${inputs.normal}, normalize(u.cameraPosition.xyz - in.worldPos)), 0.0);`,
      `let ${id}_NdotL = max(dot(${inputs.normal}, normalize(u.lightDir.xyz)), 0.0);`,
      `let ${id}_NdotH = max(dot(${inputs.normal}, ${id}_H), 0.0);`,
      `let ${id}_VdotH = max(dot(normalize(u.cameraPosition.xyz - in.worldPos), ${id}_H), 0.0);`,
      `let ${id}_a = ${inputs.roughness} * ${inputs.roughness};`,
      `let ${id}_a2 = ${id}_a * ${id}_a;`,
      `let ${id}_NDF = ${id}_a2 / (PI * pow(${id}_NdotH * ${id}_NdotH * (${id}_a2 - 1.0) + 1.0, 2.0));`,
      `let ${id}_k = pow(${id}_a + 1.0, 2.0) / 8.0;`,
      `let ${id}_G1 = ${id}_NdotV / (${id}_NdotV * (1.0 - ${id}_k) + ${id}_k);`,
      `let ${id}_G2 = ${id}_NdotL / (${id}_NdotL * (1.0 - ${id}_k) + ${id}_k);`,
      `let ${id}_G = ${id}_G1 * ${id}_G2;`,
      `let ${id}_F = ${id}_F0 + (1.0 - ${id}_F0) * pow(1.0 - ${id}_VdotH, 5.0);`,
      `let ${id}_spec = (${id}_NDF * ${id}_G * ${id}_F) / (4.0 * ${id}_NdotV * ${id}_NdotL + 0.0001);`,
      `let ${id}_kD = (vec3<f32>(1.0) - ${id}_F) * (1.0 - ${inputs.metallic});`,
      `let ${id}_Lo = (${id}_kD * ${inputs.base_color} / PI + ${id}_spec) * ${id}_NdotL * u.lightColor.rgb;`,
      `let ${id}_ambient = vec3<f32>(0.03) * ${inputs.base_color} * ${inputs.ao};`,
      `let ${id}_emissive = ${inputs.emissive} * ${inputs.emissive_strength};`,
      `let ${id}_color = ${id}_ambient + ${id}_Lo + ${id}_emissive;`,
    ].join("\n  ") + `\nlet ${id}_out = ${id}_color;`;
  },
});

registerNode({
  type: "texture",
  inputs: [
    { name: "uv", type: "vec2" },
  ],
  outputs: [
    { name: "color", type: "color" },
    { name: "alpha", type: "float" },
  ],
  emit: (inputs, node) => {
    const id = node.id.replace(/[^a-zA-Z0-9]/g, "_");
    const uv = inputs.uv || "in.uv";
    return [
      `let ${id}_sample = textureSample(materialTexture, materialSampler, ${uv});`,
      `let ${id}_color = ${id}_sample.rgb;`,
      `let ${id}_alpha = ${id}_sample.a;`,
    ].join("\n  ");
  },
});

registerNode({
  type: "normal_map",
  inputs: [
    { name: "tangent_normal", type: "vec3" },
  ],
  outputs: [
    { name: "normal", type: "normal" },
  ],
  emit: (inputs, node) => {
    const id = node.id.replace(/[^a-zA-Z0-9]/g, "_");
    return [
      `let ${id}_N = normalize(in.worldNormal);`,
      `let ${id}_T = normalize(in.worldTangent);`,
      `let ${id}_B = cross(${id}_N, ${id}_T);`,
      `let ${id}_map = ${inputs.tangent_normal} * 2.0 - 1.0;`,
      `let ${id}_worldNormal = normalize(${id}_T * ${id}_map.x + ${id}_B * ${id}_map.y + ${id}_N * ${id}_map.z);`,
    ].join("\n  ");
  },
});

registerNode({
  type: "mix/blend",
  inputs: [
    { name: "a", type: "color" },
    { name: "b", type: "color" },
    { name: "factor", type: "float", default: 0.5 },
  ],
  outputs: [
    { name: "color", type: "color" },
  ],
  emit: (inputs) => `mix(${inputs.a}, ${inputs.b}, clamp(${inputs.factor}, 0.0, 1.0))`,
});

registerNode({
  type: "mix/multiply",
  inputs: [
    { name: "a", type: "color" },
    { name: "b", type: "color" },
  ],
  outputs: [
    { name: "color", type: "color" },
  ],
  emit: (inputs) => `${inputs.a} * ${inputs.b}`,
});

registerNode({
  type: "math/add",
  inputs: [
    { name: "a", type: "float" },
    { name: "b", type: "float" },
  ],
  outputs: [
    { name: "value", type: "float" },
  ],
  emit: (inputs) => `(${inputs.a} + ${inputs.b})`,
});

registerNode({
  type: "math/multiply",
  inputs: [
    { name: "a", type: "float" },
    { name: "b", type: "float" },
  ],
  outputs: [
    { name: "value", type: "float" },
  ],
  emit: (inputs) => `(${inputs.a} * ${inputs.b})`,
});

registerNode({
  type: "math/clamp",
  inputs: [
    { name: "value", type: "float" },
    { name: "min", type: "float", default: 0.0 },
    { name: "max", type: "float", default: 1.0 },
  ],
  outputs: [
    { name: "value", type: "float" },
  ],
  emit: (inputs) => `clamp(${inputs.value}, ${inputs.min}, ${inputs.max})`,
});

registerNode({
  type: "math/sin",
  inputs: [
    { name: "value", type: "float" },
  ],
  outputs: [
    { name: "value", type: "float" },
  ],
  emit: (inputs) => `sin(${inputs.value})`,
});

registerNode({
  type: "math/time",
  inputs: [],
  outputs: [
    { name: "value", type: "float" },
  ],
  emit: () => `u.time`,
});

registerNode({
  type: "input/uv",
  inputs: [],
  outputs: [
    { name: "uv", type: "vec2" },
  ],
  emit: () => `in.uv`,
});

registerNode({
  type: "input/normal",
  inputs: [],
  outputs: [
    { name: "normal", type: "normal" },
  ],
  emit: () => `normalize(in.worldNormal)`,
});

registerNode({
  type: "constant/float",
  inputs: [],
  outputs: [
    { name: "value", type: "float" },
  ],
  emit: (_inputs, node) => String(node.params.value ?? 1.0),
});

registerNode({
  type: "constant/color",
  inputs: [],
  outputs: [
    { name: "color", type: "color" },
  ],
  emit: (_inputs, node) => {
    const r = node.params.r ?? 1.0;
    const g = node.params.g ?? 1.0;
    const b = node.params.b ?? 1.0;
    return `vec3<f32>(${r}, ${g}, ${b})`;
  },
});

registerNode({
  type: "toon/ramp",
  inputs: [
    { name: "base_color", type: "color" },
    { name: "ramp_threshold", type: "float", default: 0.5 },
    { name: "ramp_smooth", type: "float", default: 0.1 },
  ],
  outputs: [
    { name: "color", type: "color" },
  ],
  emit: (inputs, node) => {
    const id = node.id.replace(/[^a-zA-Z0-9]/g, "_");
    return [
      `let ${id}_NdotL = max(dot(normalize(in.worldNormal), normalize(u.lightDir.xyz)), 0.0);`,
      `let ${id}_ramp = smoothstep(${inputs.ramp_threshold} - ${inputs.ramp_smooth}, ${inputs.ramp_threshold} + ${inputs.ramp_smooth}, ${id}_NdotL);`,
      `let ${id}_color = ${inputs.base_color} * (0.3 + 0.7 * ${id}_ramp);`,
    ].join("\n  ");
  },
});

registerNode({
  type: "emissive/pulse",
  inputs: [
    { name: "color", type: "color" },
    { name: "speed", type: "float", default: 1.0 },
    { name: "intensity", type: "float", default: 2.0 },
  ],
  outputs: [
    { name: "emissive", type: "color" },
  ],
  emit: (inputs) => {
    return `${inputs.color} * ${inputs.intensity} * (0.5 + 0.5 * sin(u.time * ${inputs.speed}))`;
  },
});