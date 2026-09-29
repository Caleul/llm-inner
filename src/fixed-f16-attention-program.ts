import { type SafetensorsCatalogReader } from "./safetensors.js";
import {
  addDyadic, compileFixedF16Projection, f16BitsToDyadic, f32BitsToDyadic,
  multiplyDyadic, roundDyadicToF16IfElse, roundDyadicToF32IfElse,
} from "./fixed-f16-projection.js";
import { softmaxTwoF16IfElse } from "./fixed-f16-two-way-softmax.js";

export type FixedAttentionNode =
  | { op: "input"; index: number }
  | { op: "constant"; bits: number }
  | { op: "mul-f16" | "add-f16" | "mul-f32" | "add-f32" | "softmax-0" | "softmax-1"; left: number; right: number }
  | { op: "f32-from-f16" | "f16-from-f32" | "neg-f16"; input: number };

/** Outputs are 32 scalar roots in token-major order. All node dependencies point backward. */
export interface FixedTwoTokenAttentionProgram {
  kind: "fixed-two-token-attention-program";
  inputSize: 32;
  nodes: FixedAttentionNode[];
  outputs: number[];
}

class Builder {
  readonly nodes: FixedAttentionNode[] = [];
  private readonly ids = new Map<string, number>();
  node(value: FixedAttentionNode): number {
    const key = JSON.stringify(value);
    const found = this.ids.get(key);
    if (found !== undefined) return found;
    const id = this.nodes.length;
    this.nodes.push(value);
    this.ids.set(key, id);
    return id;
  }
  binary(op: Extract<FixedAttentionNode, { left: number }>["op"], left: number, right: number): number {
    return this.node({ op, left, right });
  }
  unary(op: Extract<FixedAttentionNode, { input: number }>["op"], input: number): number {
    return this.node({ op, input });
  }
  constant(bits: number): number { return this.node({ op: "constant", bits }); }
  dot(left: number[], right: number[]): number {
    if (left.length !== right.length) throw new Error("Dimensões do produto escalar incompatíveis.");
    let accumulator = this.unary("f32-from-f16", this.constant(0));
    for (let i = 0; i < left.length; i++) {
      const product = this.binary("mul-f32", left[i]!, right[i]!);
      accumulator = this.binary("add-f32", accumulator, product);
    }
    return this.unary("f16-from-f32", accumulator);
  }
}

/** Compile literal checkpoint weights into a shared scalar DAG; no projected tensor is an input. */
export async function compileFixedTwoTokenAttention(
  reader: SafetensorsCatalogReader, layer: number, cos: readonly (readonly number[])[], sin: readonly (readonly number[])[],
): Promise<FixedTwoTokenAttentionProgram> {
  if (cos.length !== 2 || sin.length !== 2 || cos.some((row) => row.length !== 4) || sin.some((row) => row.length !== 4)) {
    throw new Error("RoPE fixa requer dois tokens e head_dim=4.");
  }
  const b = new Builder();
  const input = Array.from({ length: 2 }, (_, token) => Array.from({ length: 16 }, (_, dimension) =>
    b.node({ op: "input", index: token * 16 + dimension })));
  const project = async (name: string, vectors: number[][]): Promise<number[][]> => {
    const projection = await compileFixedF16Projection(reader, `model.layers.${layer}.self_attn.${name}.weight`);
    if (projection.inputSize !== 16 || projection.outputSize !== 16) throw new Error(`${name}: shape incompatível.`);
    return vectors.map((vector) => projection.rows.map((row) => {
      const lanes = Array.from({ length: 4 }, () => b.unary("f32-from-f16", b.constant(0)));
      for (const term of row.terms) {
        const product = b.binary("mul-f32", vector[term.input]!, b.constant(term.weightBits));
        const lane = term.input & 3;
        lanes[lane] = b.binary("add-f32", lanes[lane]!, product);
      }
      const first = b.binary("add-f32", lanes[0]!, lanes[1]!);
      const second = b.binary("add-f32", lanes[2]!, lanes[3]!);
      return b.unary("f16-from-f32", b.binary("add-f32", first, second));
    }));
  };
  const q = await project("q_proj", input), k = await project("k_proj", input), v = await project("v_proj", input);
  const rotate = (vectors: number[][]) => Array.from({ length: 4 }, (_, head) => Array.from({ length: 2 }, (_, token) =>
    Array.from({ length: 4 }, (_, dimension) => {
      const value = vectors[token]![head * 4 + dimension]!;
      const partner = vectors[token]![head * 4 + (dimension + 2) % 4]!;
      const rotated = dimension < 2 ? b.unary("neg-f16", partner) : partner;
      const first = b.binary("mul-f16", value, b.constant(cos[token]![dimension]!));
      const second = b.binary("mul-f16", rotated, b.constant(sin[token]![dimension]!));
      return b.binary("add-f16", first, second);
    })));
  const qr = rotate(q), kr = rotate(k);
  const context = Array.from({ length: 2 }, () => Array<number>(16));
  for (let head = 0; head < 4; head++) for (let token = 0; token < 2; token++) {
    const scores = [0, 1].map((keyToken) => {
      const score = b.dot(qr[head]![token]!, kr[head]![keyToken]!);
      const scaled = b.binary("mul-f16", score, b.constant(0x3800));
      return b.binary("add-f16", scaled, b.constant(keyToken > token ? 0xfbff : 0));
    });
    const probabilities = [b.binary("softmax-0", scores[0]!, scores[1]!), b.binary("softmax-1", scores[0]!, scores[1]!)];
    for (let dimension = 0; dimension < 4; dimension++) {
      context[token]![head * 4 + dimension] = b.dot(probabilities,
        [v[0]![head * 4 + dimension]!, v[1]![head * 4 + dimension]!]);
    }
  }
  const output = await project("o_proj", context);
  return { kind: "fixed-two-token-attention-program", inputSize: 32, nodes: b.nodes, outputs: output.flat() };
}

/** Every output is evaluated from the 32 input bits and serialized literal nodes. */
export function evaluateFixedTwoTokenAttention(program: FixedTwoTokenAttentionProgram, input: readonly number[]): number[] {
  if (input.length !== 32 || program.inputSize !== 32 || program.outputs.length !== 32) throw new Error("Entrada ou saída da atenção fixa inválida.");
  const values: number[] = [];
  const probabilities = new Map<string, [number, number]>();
  for (const node of program.nodes) {
    const get = (id: number) => { const value = values[id]; if (value === undefined) throw new Error("DAG de atenção fora de ordem."); return value; };
    switch (node.op) {
      case "input": values.push(input[node.index]!); break;
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
        let pair = probabilities.get(key);
        if (!pair) { pair = softmaxTwoF16IfElse(get(node.left), get(node.right)); probabilities.set(key, pair); }
        values.push(pair[node.op === "softmax-0" ? 0 : 1]);
        break;
      }
    }
  }
  return program.outputs.map((id) => getOutput(values, id));
}

function getOutput(values: readonly number[], id: number): number {
  const value = values[id];
  if (value === undefined) throw new Error("Raiz de saída ausente.");
  return value;
}
