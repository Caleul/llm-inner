import type {
  Gemma4ParametricExactRealProgram,
  Gemma4ParametricOperationFunction,
  Gemma4ParametricOutputFunction,
} from "./gemma4-parametric-global-real-program.js";
import type { Gemma4ParametricRealNode } from "./gemma4-parametric-real-expression.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { decodeIeeeBF16ToF32, decodeIeeeF16ToF32 } from "./utils.js";

type Scalar = number | boolean;

export interface Gemma4ParametricTensorProvider {
  element(tensor: string, coordinates: readonly number[], valueType: "real" | "integer" | "boolean"): Scalar | Promise<Scalar>;
  axis(tensor: string, axis: number): number | Promise<number>;
}

export interface Gemma4ParametricLearnedProvider {
  element(tensor: string, storageDtype: "BF16" | "F16" | "F32", coordinates: readonly number[], decoderId: string): number | Promise<number>;
}

export interface Gemma4ParametricEvaluationContext {
  inputs: Gemma4ParametricTensorProvider;
  learned: Gemma4ParametricLearnedProvider;
  parameters: Readonly<Record<string, number>>;
}

/** Bounded, integrity-verified learned-rational provider backed only by the literal artifact. */
export function gemma4ArtifactLearnedProvider(
  artifact: OpenGemma4CompositeLiteralArtifact,
  options: { pageBytes?: number; maxCachedPages?: number } = {},
): Gemma4ParametricLearnedProvider {
  const pageBytes = options.pageBytes ?? 1024 * 1024, maxCachedPages = options.maxCachedPages ?? 256;
  const pages = new Map<string, Promise<Buffer>>();
  return {
    element: async (tensor, storageDtype, coordinates) => {
      const constant = artifact.constants.get(tensor);
      if (!constant || constant.storageDtype !== storageDtype || coordinates.length !== constant.storageShape.length) {
        throw new Error(`${tensor}: constante racional aprendida incompatível.`);
      }
      let linear = 0;
      for (let axis = 0; axis < coordinates.length; axis += 1) {
        const coordinate = coordinates[axis]!, extent = constant.storageShape[axis]!;
        if (!Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= extent) throw new Error(`${tensor}: coordenada aprendida fora do domínio.`);
        linear = linear * extent + coordinate;
      }
      const bytesPerElement = storageDtype === "F32" ? 4 : 2;
      const byteOffset = linear * bytesPerElement;
      const pageOffset = Math.floor(byteOffset / pageBytes) * pageBytes;
      const key = `${tensor}:${pageOffset}`;
      let page = pages.get(key);
      if (!page) {
        const length = Math.min(pageBytes, constant.payloadBytes - pageOffset);
        page = artifact.readTensorBytesRange(constant, pageOffset, length);
        pages.set(key, page);
        if (pages.size > maxCachedPages) pages.delete(pages.keys().next().value!);
      }
      const bytes = await page, within = byteOffset - pageOffset;
      return storageDtype === "F32" ? bytes.readFloatLE(within)
        : storageDtype === "F16" ? decodeIeeeF16ToF32(bytes.readUInt16LE(within))
        : decodeIeeeBF16ToF32(bytes.readUInt16LE(within));
    },
  };
}

/** Executes the simplified real semantics and rounds only at the declared public output. */
export async function evaluateGemma4ParametricOutput(
  program: Gemma4ParametricExactRealProgram,
  family: string,
  dimension: number,
  context: Gemma4ParametricEvaluationContext,
): Promise<number> {
  const output = program.outputFunctions.find((candidate) => candidate.name === family && candidate.fixedDimension === dimension);
  if (!output) throw new Error(`${family}[${dimension}]: função real de saída ausente.`);
  const evaluator = new ParametricEvaluator(program, context);
  const parameters = new Map<string, number>();
  output.parameters.forEach((axis, index) => {
    const coordinate = program.expressionGraph.nodes.find((node) => node.id === output.coordinate[index]);
    if (!coordinate || coordinate.kind !== "integer-parameter") throw new Error(`${family}: parâmetro de coordenada ${axis} inválido.`);
    const value = context.parameters[axis];
    if (value === undefined) throw new Error(`${family}: parâmetro ${axis} ausente.`);
    parameters.set(coordinate.name, value);
  });
  const value = numeric(await evaluator.evaluate(output.root, parameters));
  return quantizeFinal(value, output.finalQuantization);
}

class ParametricEvaluator {
  readonly #nodes: ReadonlyMap<string, Gemma4ParametricRealNode>;
  readonly #functions: ReadonlyMap<string, Gemma4ParametricOperationFunction>;
  readonly #memo = new Map<string, Promise<Scalar>>();

  constructor(
    program: Gemma4ParametricExactRealProgram,
    readonly context: Gemma4ParametricEvaluationContext,
  ) {
    this.#nodes = new Map(program.expressionGraph.nodes.map((node) => [node.id, node]));
    this.#functions = new Map(program.operationFunctions.map((entry) => [entry.functionId, entry]));
  }

  evaluate(id: string, parameters: ReadonlyMap<string, number> = new Map(Object.entries(this.context.parameters)), bounds: ReadonlyMap<string, number> = new Map()): Promise<Scalar> {
    const key = `${id}|p:${stableEnvironment(parameters)}|b:${stableEnvironment(bounds)}`;
    let result = this.#memo.get(key);
    if (!result) {
      result = this.#evaluateNode(required(this.#nodes, id, "nó"), parameters, bounds);
      this.#memo.set(key, result);
    }
    return result;
  }

  async #evaluateNode(node: Gemma4ParametricRealNode, parameters: ReadonlyMap<string, number>, bounds: ReadonlyMap<string, number>): Promise<Scalar> {
    const value = (id: string) => this.evaluate(id, parameters, bounds);
    const integer = async (id: string) => exactInteger(await value(id), id);
    switch (node.kind) {
      case "integer-constant": return node.value;
      case "integer-parameter": return required(parameters, node.name, "parâmetro");
      case "integer-bound-index": return required(bounds, node.name, "índice ligado");
      case "integer-add": return await integer(node.left) + await integer(node.right);
      case "integer-subtract": return await integer(node.left) - await integer(node.right);
      case "integer-multiply": return await integer(node.left) * await integer(node.right);
      case "integer-floor-divide": return Math.floor(await integer(node.left) / await integer(node.right));
      case "integer-modulo": { const left = await integer(node.left), right = await integer(node.right); return ((left % right) + right) % right; }
      case "tensor-axis": return this.context.inputs.axis(node.tensor, node.axis);
      case "stable-true-prefix-rank": {
        const batch = await integer(node.batch), sequence = await integer(node.sequence);
        if (Number(await this.context.inputs.element(node.tensor, [batch, sequence], "integer")) !== node.equals) return -1;
        const rows = Number(await this.context.inputs.axis(node.tensor, 0)), columns = Number(await this.context.inputs.axis(node.tensor, 1));
        let rank = 0;
        for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
          if (row === batch && column === sequence) return rank;
          if (Number(await this.context.inputs.element(node.tensor, [row, column], "integer")) === node.equals) rank += 1;
        }
        throw new Error(`${node.tensor}: coordenada fora do domínio para stable prefix rank.`);
      }
      case "rational": return Number(BigInt(node.value.numerator)) / Number(BigInt(node.value.denominator));
      case "boolean": return node.value;
      case "negative-infinity": return Number.NEGATIVE_INFINITY;
      case "positive-infinity": return Number.POSITIVE_INFINITY;
      case "input-element": return this.context.inputs.element(node.tensor, await Promise.all(node.coordinates.map(integer)), node.valueType);
      case "learned-rational-element": return this.context.learned.element(node.tensor, node.storageDtype, await Promise.all(node.coordinates.map(integer)), node.decoderId);
      case "function-call": {
        const function_ = required(this.#functions, node.functionId, "função");
        const arguments_ = await Promise.all(node.arguments.map(integer));
        if (arguments_.length !== function_.parameters.length) throw new Error(`${node.functionId}: aridade divergente.`);
        const calledParameters = new Map(function_.parameters.map((parameter, index) => [required(this.#nodes, parameter.node, "parâmetro").kind === "integer-parameter"
          ? (required(this.#nodes, parameter.node, "parâmetro") as { kind: "integer-parameter"; name: string }).name
          : parameter.name, arguments_[index]!]));
        return this.evaluate(function_.root, calledParameters, new Map());
      }
      case "add": return (await Promise.all(node.arguments.map(value))).reduce<number>((sum, entry) => sum + numeric(entry), 0);
      case "multiply": return (await Promise.all(node.arguments.map(value))).reduce<number>((product, entry) => product * numeric(entry), 1);
      case "minimum": return Math.min(...(await Promise.all(node.arguments.map(value))).map(numeric));
      case "maximum": return Math.max(...(await Promise.all(node.arguments.map(value))).map(numeric));
      case "divide": return numeric(await value(node.numerator)) / numeric(await value(node.denominator));
      case "integer-power": return numeric(await value(node.base)) ** node.exponent;
      case "power": return numeric(await value(node.base)) ** numeric(await value(node.exponent));
      case "unary-function": return unary(node.function, numeric(await value(node.argument)));
      case "compare": return compare(node.comparison, numeric(await value(node.left)), numeric(await value(node.right)));
      case "select": return truthy(await value(node.condition)) ? value(node.whenTrue) : value(node.whenFalse);
      case "finite-sum": case "finite-maximum": {
        const start = await integer(node.startInclusive), end = await integer(node.endExclusive);
        let result = node.kind === "finite-sum" ? 0 : Number.NEGATIVE_INFINITY;
        for (let index = start; index < end; index += 1) {
          const iterationBounds = new Map(bounds).set(node.index, index);
          if (node.predicate && !truthy(await this.evaluate(node.predicate, parameters, iterationBounds))) continue;
          const entry = numeric(await this.evaluate(node.body, parameters, iterationBounds));
          result = node.kind === "finite-sum" ? result + entry : Math.max(result, entry);
        }
        return result;
      }
    }
  }
}

function required<K, V>(map: ReadonlyMap<K, V>, key: K, kind: string): V {
  const value = map.get(key);
  if (value === undefined) throw new Error(`${kind} paramétrico ausente: ${String(key)}.`);
  return value;
}

function stableEnvironment(environment: ReadonlyMap<string, number>): string {
  return [...environment].sort(([left], [right]) => left.localeCompare(right, "en")).map(([name, value]) => `${name}=${value}`).join(",");
}

function numeric(value: Scalar): number { if (typeof value !== "number") throw new Error("Valor real esperado; boolean recebido."); return value; }
function truthy(value: Scalar): boolean { return typeof value === "boolean" ? value : value !== 0; }
function exactInteger(value: Scalar, id: string): number { const number = numeric(value); if (!Number.isSafeInteger(number)) throw new Error(`${id}: inteiro seguro esperado, recebeu ${number}.`); return number; }

function unary(function_: "abs" | "exp" | "sin" | "cos" | "tan" | "tanh" | "log1p" | "sqrt" | "floor", value: number): number {
  switch (function_) {
    case "abs": return Math.abs(value);
    case "exp": return Math.exp(value); case "sin": return Math.sin(value); case "cos": return Math.cos(value); case "tan": return Math.tan(value);
    case "tanh": return Math.tanh(value); case "log1p": return Math.log1p(value); case "sqrt": return Math.sqrt(value); case "floor": return Math.floor(value);
  }
}

function compare(comparison: "equal" | "not-equal" | "less" | "less-equal" | "greater" | "greater-equal", left: number, right: number): boolean {
  switch (comparison) {
    case "equal": return left === right; case "not-equal": return left !== right; case "less": return left < right;
    case "less-equal": return left <= right; case "greater": return left > right; case "greater-equal": return left >= right;
  }
}

function quantizeFinal(value: number, quantization: Gemma4ParametricOutputFunction["finalQuantization"]): number {
  if (quantization === "none") return value;
  const f32 = Math.fround(value);
  if (quantization === "F32-round-to-nearest-ties-to-even") return f32;
  const view = new DataView(new ArrayBuffer(4)); view.setFloat32(0, f32, true);
  let bits = view.getUint32(0, true), upper = bits >>> 16, lower = bits & 0xffff;
  if (lower > 0x8000 || (lower === 0x8000 && (upper & 1) === 1)) upper = (upper + 1) & 0xffff;
  bits = upper << 16; view.setUint32(0, bits, true); return view.getFloat32(0, true);
}
