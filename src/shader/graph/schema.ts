export type SocketType =
  | "float" | "vec2" | "vec3" | "vec4"
  | "color" | "normal"
  | "texture" | "sampler"
  | "bool";

export interface SocketDecl {
  name: string;
  type: SocketType;
  default?: number | boolean | string;
}

export interface NodeDecl {
  type: string;
  inputs: SocketDecl[];
  outputs: SocketDecl[];
  emit: (inputs: Record<string, string>, node: GraphNode) => string;
}

export interface GraphNode {
  id: string;
  type: string;
  params: Record<string, number | boolean | string>;
}

export interface GraphLink {
  fromNode: string;
  fromSocket: string;
  toNode: string;
  toSocket: string;
}

export interface ExposedParam {
  name: string;
  type: SocketType;
  default: number | boolean | string;
  min?: number;
  max?: number;
  step?: number;
}

export interface ShaderGraph {
  version: number;
  name: string;
  nodes: GraphNode[];
  links: GraphLink[];
  output: {
    node: string;
    socket: string;
  };
  params?: ExposedParam[];
}

export interface CompiledGraph {
  wgsl: string;
  paramSlots: ParamSlot[];
  usedTextures: string[];
}

export interface ParamSlot {
  name: string;
  type: SocketType;
  offset: number;
  size: number;
}