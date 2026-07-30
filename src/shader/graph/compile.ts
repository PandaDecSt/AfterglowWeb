import type {
  ShaderGraph,
  GraphNode,
  GraphLink,
  CompiledGraph,
  ParamSlot,
  SocketType,
} from "./schema";
import { getNode, wgslType } from "./registry";

export function compileGraph(graph: ShaderGraph): CompiledGraph {
  validateGraph(graph);
  const reachable = pruneUnreachable(graph);
  const sorted = topologicalSort(graph, reachable);
  const { wgslBody, paramSlots, usedTextures } = emitWGSL(graph, sorted);
  const wgsl = assembleModule(wgslBody, usedTextures);
  return { wgsl, paramSlots, usedTextures };
}

function validateGraph(graph: ShaderGraph): void {
  if (graph.version !== 1) {
    throw new Error(`[ShaderGraph] Unsupported version: ${graph.version}`);
  }

  const nodeIds = new Set<string>();
  for (const node of graph.nodes) {
    if (nodeIds.has(node.id)) {
      throw new Error(`[ShaderGraph] Duplicate node id: "${node.id}"`);
    }
    nodeIds.add(node.id);
    if (!getNode(node.type)) {
      throw new Error(`[ShaderGraph] Unknown node type: "${node.type}" (node: ${node.id})`);
    }
  }

  for (const link of graph.links) {
    if (!nodeIds.has(link.fromNode)) {
      throw new Error(`[ShaderGraph] Link references unknown node: "${link.fromNode}"`);
    }
    if (!nodeIds.has(link.toNode)) {
      throw new Error(`[ShaderGraph] Link references unknown node: "${link.toNode}"`);
    }
  }

  if (!nodeIds.has(graph.output.node)) {
    throw new Error(`[ShaderGraph] Output node "${graph.output.node}" not found`);
  }
}

function pruneUnreachable(graph: ShaderGraph): Set<string> {
  const reachable = new Set<string>();
  const stack = [graph.output.node];

  while (stack.length > 0) {
    const nodeId = stack.pop()!;
    if (reachable.has(nodeId)) continue;
    reachable.add(nodeId);

    for (const link of graph.links) {
      if (link.toNode === nodeId && !reachable.has(link.fromNode)) {
        stack.push(link.fromNode);
      }
    }
  }

  return reachable;
}

function topologicalSort(graph: ShaderGraph, reachable: Set<string>): GraphNode[] {
  const nodes = graph.nodes.filter((n) => reachable.has(n.id));
  const nodeIdSet = new Set(nodes.map((n) => n.id));

  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();

  for (const node of nodes) {
    inDegree.set(node.id, 0);
    adj.set(node.id, []);
  }

  for (const link of graph.links) {
    if (!nodeIdSet.has(link.fromNode) || !nodeIdSet.has(link.toNode)) continue;
    adj.get(link.fromNode)!.push(link.toNode);
    inDegree.set(link.toNode, (inDegree.get(link.toNode) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id);
  }
  queue.sort();

  const sorted: string[] = [];
  while (queue.length > 0) {
    const id = queue.shift()!;
    sorted.push(id);
    const targets = adj.get(id)!.slice().sort();
    for (const t of targets) {
      const newDeg = (inDegree.get(t) ?? 1) - 1;
      inDegree.set(t, newDeg);
      if (newDeg === 0) queue.push(t);
    }
    queue.sort();
  }

  if (sorted.length !== nodes.length) {
    throw new Error("[ShaderGraph] Cycle detected in graph");
  }

  return sorted.map((id) => nodes.find((n) => n.id === id)!);
}

interface EmitResult {
  wgslBody: string;
  paramSlots: ParamSlot[];
  usedTextures: string[];
}

function emitWGSL(graph: ShaderGraph, sorted: GraphNode[]): EmitResult {
  const paramSlots: ParamSlot[] = [];
  const usedTextures: string[] = [];
  const lines: string[] = [];
  const outputValues = new Map<string, Map<string, string>>();

  function getOutput(nodeId: string, socket: string): string {
    return outputValues.get(nodeId)?.get(socket) ?? `vec3<f32>(0.0)`;
  }

  function resolveInput(nodeId: string, socketName: string, socketType: SocketType): string {
    for (const link of graph.links) {
      if (link.toNode === nodeId && link.toSocket === socketName) {
        return getOutput(link.fromNode, link.fromSocket);
      }
    }

    const node = graph.nodes.find((n) => n.id === nodeId)!;
    const decl = getNode(node.type)!;
    const inputDecl = decl.inputs.find((i) => i.name === socketName);
    if (inputDecl?.default !== undefined) {
      if (typeof inputDecl.default === "number") {
        if (socketType === "color" || socketType === "normal") {
          const v = inputDecl.default;
          return `vec3<f32>(${v}, ${v}, ${v})`;
        }
        return String(inputDecl.default);
      }
      return String(inputDecl.default);
    }

    if (socketType === "color" || socketType === "normal") return "vec3<f32>(0.0)";
    if (socketType === "float") return "0.0";
    if (socketType === "vec2") return "vec2<f32>(0.0)";
    if (socketType === "vec4") return "vec4<f32>(0.0)";
    return "0.0";
  }

  for (const node of sorted) {
    const decl = getNode(node.type)!;
    const id = node.id.replace(/[^a-zA-Z0-9]/g, "_");

    if (node.type === "texture") {
      usedTextures.push(node.id);
    }

    const inputValues: Record<string, string> = {};
    for (const inputDecl of decl.inputs) {
      inputValues[inputDecl.name] = resolveInput(node.id, inputDecl.name, inputDecl.type);
    }

    const emitted = decl.emit(inputValues, node);

    if (decl.outputs.length === 0) {
      lines.push(`  ${emitted}`);
    } else if (decl.outputs.length === 1) {
      const outName = decl.outputs[0].name;
      const outType = wgslType(decl.outputs[0].type);

      if (emitted.includes("\n")) {
        lines.push(`  ${emitted}`);
        const lastLet = emitted.match(/let (\w+_out)\s*=/);
        if (lastLet) {
          if (!outputValues.has(node.id)) outputValues.set(node.id, new Map());
          outputValues.get(node.id)!.set(outName, lastLet[1]);
        } else {
          if (!outputValues.has(node.id)) outputValues.set(node.id, new Map());
          outputValues.get(node.id)!.set(outName, `${id}_${outName}`);
        }
      } else {
        const varName = `${id}_${outName}`;
        lines.push(`  let ${varName}: ${outType} = ${emitted};`);
        if (!outputValues.has(node.id)) outputValues.set(node.id, new Map());
        outputValues.get(node.id)!.set(outName, varName);
      }
    } else {
      lines.push(`  ${emitted}`);
      if (!outputValues.has(node.id)) outputValues.set(node.id, new Map());
      for (const outDecl of decl.outputs) {
        outputValues.get(node.id)!.set(outDecl.name, `${id}_${outDecl.name}`);
      }
    }
  }

  const finalValue = getOutput(graph.output.node, graph.output.socket);
  lines.push(`  return vec4<f32>(${finalValue}, 1.0);`);

  return { wgslBody: lines.join("\n"), paramSlots, usedTextures };
}

function assembleModule(fsBody: string, usedTextures: string[]): string {
  const hasTexture = usedTextures.length > 0;

  const uniformBlock = `
struct GraphUniforms {
  cameraPosition: vec4<f32>,
  lightDir: vec4<f32>,
  lightColor: vec4<f32>,
  time: f32,
  pad0: f32,
  pad1: f32,
  pad2: f32,
};

@group(0) @binding(0) var<uniform> u: GraphUniforms;`;

  const textureBindings = hasTexture
    ? `
@group(0) @binding(1) var materialTexture: texture_2d<f32>;
@group(0) @binding(2) var materialSampler: sampler;`
    : "";

  const vs = `
struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VSIn) -> VSOut {
  var out: VSOut;
  out.position = u.cameraPosition; // placeholder - overridden by pipeline setup
  out.worldNormal = in.normal;
  out.worldPos = in.position;
  out.uv = in.uv;
  return out;
}`;

  const fs = `
struct FSIn {
  @location(0) worldNormal: vec3<f32>,
  @location(1) worldPos: vec3<f32>,
  @location(2) uv: vec2<f32>,
};

const PI: f32 = 3.14159265359;

@fragment
fn fs_main(in: FSIn) -> @location(0) vec4<f32> {
${fsBody}
}`;

  return uniformBlock + textureBindings + vs + fs;
}