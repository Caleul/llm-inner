import type { Gemma4RealExpressionGraph, Gemma4RealExpressionNode } from "./gemma4-real-expression.js";
import { validateGemma4RealExpressionGraph } from "./gemma4-real-expression.js";

export type Gemma4RealEvaluatedValue = number | boolean;

export interface Gemma4RealExpressionEvaluation {
  semantics: "fast-f64-no-intermediate-f32-bf16-casts";
  root: string;
  realApproximation: number;
  finalF32: number;
  finalF32BitsHex: string;
  finalBF16: number;
  finalBF16BitsHex: string;
}

export function evaluateGemma4RealExpressionFastF64(
  graph: Gemma4RealExpressionGraph,
  root: string,
  inputs: Readonly<Record<string, number | boolean>>,
): Gemma4RealExpressionEvaluation {
  validateGemma4RealExpressionGraph(graph);
  const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
  const memo = new Map<string, Gemma4RealEvaluatedValue>();
  const evaluate = (id: string): Gemma4RealEvaluatedValue => {
    const known = memo.get(id);
    if (known !== undefined) return known;
    const node = nodes.get(id);
    if (!node) throw new Error(`Avaliador real Gemma 4 não encontrou ${id}.`);
    const value = evaluateNode(node, evaluate, inputs);
    memo.set(id, value);
    return value;
  };
  const result = evaluate(root);
  if (typeof result !== "number" || !Number.isFinite(result)) throw new Error("Raiz real Gemma 4 não produziu número finito.");
  const finalF32 = Math.fround(result);
  const f32Bits = numberToF32Bits(finalF32);
  const bf16Bits = roundF32BitsToBf16(f32Bits);
  return {
    semantics: "fast-f64-no-intermediate-f32-bf16-casts",
    root,
    realApproximation: result,
    finalF32,
    finalF32BitsHex: `0x${f32Bits.toString(16).padStart(8, "0")}`,
    finalBF16: bf16BitsToNumber(bf16Bits),
    finalBF16BitsHex: `0x${bf16Bits.toString(16).padStart(4, "0")}`,
  };
}

export function gemma4RealInputKey(name: string, coordinates: readonly string[]): string {
  return coordinates.length === 0 ? name : `${name}[${coordinates.join(",")}]`;
}

function evaluateNode(
  node: Gemma4RealExpressionNode,
  evaluate: (id: string) => Gemma4RealEvaluatedValue,
  inputs: Readonly<Record<string, number | boolean>>,
): Gemma4RealEvaluatedValue {
  const number = (id: string): number => {
    const value = evaluate(id);
    if (typeof value !== "number") throw new Error(`${node.id}: argumento real esperado, recebeu booleano.`);
    return value;
  };
  const boolean = (id: string): boolean => {
    const value = evaluate(id);
    if (typeof value !== "boolean") throw new Error(`${node.id}: argumento booleano esperado, recebeu real.`);
    return value;
  };
  switch (node.kind) {
    case "rational": return Number(BigInt(node.value.numerator)) / Number(BigInt(node.value.denominator));
    case "boolean": return node.value;
    case "negative-infinity": return Number.NEGATIVE_INFINITY;
    case "input": {
      const key = gemma4RealInputKey(node.name, node.coordinates);
      const value = inputs[key];
      if (value === undefined) throw new Error(`Input real Gemma 4 ausente: ${key}.`);
      return value;
    }
    case "add": return node.arguments.reduce((sum, argument) => sum + number(argument), 0);
    case "multiply": return node.arguments.reduce((product, argument) => product * number(argument), 1);
    case "minimum": return Math.min(...node.arguments.map(number));
    case "maximum": return Math.max(...node.arguments.map(number));
    case "divide": return number(node.numerator) / number(node.denominator);
    case "modulo": {
      const left = number(node.left), right = number(node.right);
      return ((left % right) + right) % right;
    }
    case "integer-power": return number(node.base) ** node.exponent;
    case "unary-function": {
      const argument = number(node.argument);
      switch (node.function) {
        case "abs": return Math.abs(argument);
        case "exp": return Math.exp(argument);
        case "log1p": return Math.log1p(argument);
        case "sin": return Math.sin(argument);
        case "cos": return Math.cos(argument);
        case "tan": return Math.tan(argument);
        case "tanh": return Math.tanh(argument);
        case "sqrt": return Math.sqrt(argument);
        case "floor": return Math.floor(argument);
      }
    }
    case "compare": {
      const left = number(node.left), right = number(node.right);
      switch (node.comparison) {
        case "equal": return left === right;
        case "not-equal": return left !== right;
        case "less": return left < right;
        case "less-equal": return left <= right;
        case "greater": return left > right;
        case "greater-equal": return left >= right;
      }
    }
    case "select": return boolean(node.condition) ? evaluate(node.whenTrue) : evaluate(node.whenFalse);
  }
}

function numberToF32Bits(value: number): number {
  const bytes = Buffer.allocUnsafe(4);
  bytes.writeFloatLE(value);
  return bytes.readUInt32LE();
}

function roundF32BitsToBf16(bits: number): number {
  const exponent = bits & 0x7f800000;
  const fraction = bits & 0x007fffff;
  if (exponent === 0x7f800000 && fraction !== 0) return ((bits >>> 16) | 0x0040) & 0xffff;
  return ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) & 0xffff;
}

function bf16BitsToNumber(bits: number): number {
  const bytes = Buffer.allocUnsafe(4);
  bytes.writeUInt32LE((bits << 16) >>> 0);
  return bytes.readFloatLE();
}
