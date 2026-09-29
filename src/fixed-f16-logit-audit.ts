import { type FixedAttentionNode } from "./fixed-f16-attention-program.js";
import { type FixedF16Projection } from "./fixed-f16-projection.js";
import { addDyadic, f16BitsToDyadic, f32BitsToDyadic, multiplyDyadic, roundDyadicToF16IfElse, roundDyadicToF32IfElse } from "./fixed-f16-projection.js";
import { f16Bits, f16Number, f32Bits, f32Number, type FixedMlpProgram } from "./fixed-f16-layer-ops.js";
import { softmaxTwoF16IfElse } from "./fixed-f16-two-way-softmax.js";
import { type FixedTwoTokenModel } from "./fixed-f16-two-token-model.js";

type AuditNode =
  | Exclude<FixedAttentionNode, { op: "input" }>
  | { op: "embedding"; token: number; dimension: number }
  | { op: "rms-scale"; inputs: number[] }
  | { op: "rms-output"; input: number; scale: number; weightBits: number }
  | { op: "linear4"; inputs: number[]; weights: number[] }
  | { op: "silu-f16"; input: number };

/** One executable scalar root; all learned values are embedded and all edges point backward. */
export interface FixedLogitAudit {
  kind: "fixed-logit-audit";
  token: number;
  dimension: number;
  embeddingBase64: string;
  vocabSize: number;
  nodes: AuditNode[];
  root: number;
}

class Builder {
  readonly nodes: AuditNode[] = [];
  private readonly memo = new Map<string, number>();
  add(node: AuditNode): number {
    const key = JSON.stringify(node);
    const old = this.memo.get(key);
    if (old !== undefined) return old;
    const id = this.nodes.length;
    this.nodes.push(node);
    this.memo.set(key, id);
    return id;
  }
  binary(op: "add-f16" | "mul-f16", left: number, right: number): number { return this.add({ op, left, right }); }
  norm(input: number[], weight: number[]): number[] {
    return [0, 1].flatMap((token) => {
      const vector = input.slice(token * 16, token * 16 + 16);
      const scale = this.add({ op: "rms-scale", inputs: vector });
      return vector.map((value, dimension) => this.add({ op: "rms-output", input: value, scale, weightBits: weight[dimension]! }));
    });
  }
  linear(program: FixedF16Projection, input: number[]): number[] {
    return program.rows.map((row) => {
      const weights = Array<number>(program.inputSize).fill(0);
      row.terms.forEach((term) => { weights[term.input] = term.weightBits; });
      return this.add({ op: "linear4", inputs: input, weights });
    });
  }
  mlp(program: FixedMlpProgram, input: number[]): number[] {
    const gate = this.linear(program.gate, input);
    const up = this.linear(program.up, input);
    const hidden = gate.map((value, index) => this.binary("mul-f16", this.add({ op: "silu-f16", input: value }), up[index]!));
    return this.linear(program.down, hidden);
  }
  attention(program: FixedTwoTokenModel["layers"][number]["attention"], input: number[]): number[] {
    const map: number[] = [];
    for (const node of program.nodes) {
      if (node.op === "input") { map.push(input[node.index]!); continue; }
      if ("left" in node) map.push(this.add({ op: node.op, left: map[node.left]!, right: map[node.right]! }));
      else if ("input" in node) map.push(this.add({ op: node.op, input: map[node.input]! }));
      else map.push(this.add(node));
    }
    return program.outputs.map((id) => map[id]!);
  }
}

export function compileFixedLogitAudit(model: FixedTwoTokenModel, token: number, dimension: number): FixedLogitAudit {
  if (token !== 0 && token !== 1 || !Number.isInteger(dimension) || dimension < 0 || dimension >= model.vocabSize) {
    throw new Error("Coordenada de logit inválida.");
  }
  const b = new Builder();
  let hidden = [0, 1].flatMap((position) => Array.from({ length: 16 }, (_, dim) => b.add({ op: "embedding", token: position, dimension: dim })));
  for (const layer of model.layers) {
    const normalized = b.norm(hidden, layer.inputNorm);
    const attention = b.attention(layer.attention, normalized);
    const residual = hidden.map((value, index) => b.binary("add-f16", value, attention[index]!));
    const mlpInput = b.norm(residual, layer.postAttentionNorm);
    const mlp = [0, 1].flatMap((position) => b.mlp(layer.mlp, mlpInput.slice(position * 16, position * 16 + 16)));
    hidden = residual.map((value, index) => b.binary("add-f16", value, mlp[index]!));
  }
  hidden = b.norm(hidden, model.finalNorm);
  const head = Buffer.from(model.headBase64, "base64");
  const weights = Array.from({ length: 16 }, (_, index) => head.readUInt16LE(dimension * 32 + index * 2));
  const root = b.add({ op: "linear4", inputs: hidden.slice(token * 16, token * 16 + 16), weights });
  return { kind: "fixed-logit-audit", token, dimension, embeddingBase64: model.embeddingBase64,
    vocabSize: model.vocabSize, nodes: b.nodes, root };
}

export function evaluateFixedLogitAudit(audit: FixedLogitAudit, tokenIds: readonly number[]): number {
  if (tokenIds.length !== 2 || tokenIds.some((id) => !Number.isInteger(id) || id < 0 || id >= audit.vocabSize)) throw new Error("IDs inválidos.");
  const embedding = Buffer.from(audit.embeddingBase64, "base64");
  const values: number[] = [];
  const probabilityCache = new Map<string, [number, number]>();
  for (const node of audit.nodes) {
    const get = (id: number) => { const value = values[id]; if (value === undefined) throw new Error("DAG fora de ordem."); return value; };
    switch (node.op) {
      case "embedding": values.push(embedding.readUInt16LE(tokenIds[node.token]! * 32 + node.dimension * 2)); break;
      case "constant": values.push(node.bits); break;
      case "neg-f16": values.push(get(node.input) ^ 0x8000); break;
      case "f32-from-f16": values.push(roundDyadicToF32IfElse(f16BitsToDyadic(get(node.input)))); break;
      case "f16-from-f32": values.push(roundDyadicToF16IfElse(f32BitsToDyadic(get(node.input)))); break;
      case "mul-f16": values.push(roundDyadicToF16IfElse(multiplyDyadic(f16BitsToDyadic(get(node.left)), f16BitsToDyadic(get(node.right))))); break;
      case "add-f16": values.push(roundDyadicToF16IfElse(addDyadic(f16BitsToDyadic(get(node.left)), f16BitsToDyadic(get(node.right))))); break;
      case "mul-f32": values.push(roundDyadicToF32IfElse(multiplyDyadic(f16BitsToDyadic(get(node.left)), f16BitsToDyadic(get(node.right))))); break;
      case "add-f32": values.push(roundDyadicToF32IfElse(addDyadic(f32BitsToDyadic(get(node.left)), f32BitsToDyadic(get(node.right))))); break;
      case "softmax-0": case "softmax-1": {
        const key = `${node.left}:${node.right}`;
        let pair = probabilityCache.get(key);
        if (!pair) { pair = softmaxTwoF16IfElse(get(node.left), get(node.right)); probabilityCache.set(key, pair); }
        values.push(pair[node.op === "softmax-0" ? 0 : 1]);
        break;
      }
      case "rms-scale": {
        let variance = Math.fround(0);
        for (const id of node.inputs) { const x = f16Number(get(id)); variance = Math.fround(variance + Math.fround(x * x)); }
        variance = Math.fround(variance / node.inputs.length);
        values.push(f32Bits(Math.fround(1 / Math.sqrt(Math.fround(variance + Math.fround(1e-6))))));
        break;
      }
      case "rms-output": values.push(f16Bits(f16Number(f16Bits(Math.fround(f16Number(get(node.input)) * f32Number(get(node.scale))))) * f16Number(node.weightBits))); break;
      case "silu-f16": { const x = f16Number(get(node.input)); values.push(f16Bits(x / (1 + Math.exp(-x)))); break; }
      case "linear4": {
        const lanes = [0, 0, 0, 0];
        for (let index = 0; index < node.inputs.length; index++) {
          const lane = index & 3;
          lanes[lane] = Math.fround(lanes[lane]! + Math.fround(f16Number(get(node.inputs[index]!)) * f16Number(node.weights[index]!)));
        }
        values.push(f16Bits(Math.fround(Math.fround(lanes[0]! + lanes[1]!) + Math.fround(lanes[2]! + lanes[3]!))));
        break;
      }
    }
  }
  return values[audit.root]!;
}
