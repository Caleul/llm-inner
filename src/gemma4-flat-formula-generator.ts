import type { Gemma4ParametricExactRealProgram, Gemma4ParametricOutputFunction } from "./gemma4-parametric-global-real-program.js";
import type { Gemma4ParametricRealNode } from "./gemma4-parametric-real-expression.js";
import { estimateGemma4FlatExpansion } from "./gemma4-real-ssa-export.js";

export interface Gemma4FlatFormulaGeneratorOptions {
  /** Hard output bound. No partial or misleading JSON is returned. */
  maxCharacters: bigint;
  maxUnrolledReductionTerms: number;
  /** Fixes public axes such as batch and sequence; they cannot remain free. */
  outputParameters?: Readonly<Record<string, number>>;
  inputVariable(tensor: string, coordinates: readonly string[], valueType: "real" | "integer" | "boolean"): string;
  learnedLiteral(tensor: string, storageDtype: "BF16" | "F16" | "F32", coordinates: readonly number[], decoderId: string): string | Promise<string>;
}

export interface Gemma4FlatFormulaResult {
  formulas: Readonly<Record<string, string>>;
  freeVariables: string[];
  characters: number;
}

/**
 * Physically substitutes every reachable operation function and fixed finite
 * reduction. The returned strings contain no node ids or operation names.
 */
export async function generateGemma4FlatFormulaObject(
  program: Gemma4ParametricExactRealProgram,
  outputs: readonly Gemma4ParametricOutputFunction[],
  options: Gemma4FlatFormulaGeneratorOptions,
): Promise<Gemma4FlatFormulaResult> {
  if (outputs.length === 0) throw new Error("Nenhuma saída foi solicitada para a fórmula plana.");
  for (const output of outputs) for (const parameter of output.parameters) {
    const value = options.outputParameters?.[parameter];
    if (value === undefined || !Number.isSafeInteger(value) || value < 0) throw new Error(`${output.name}[${output.fixedDimension}]: parâmetro público ${parameter} precisa ser fixado.`);
  }
  let conservativeCharacters = 0n;
  for (const output of outputs) {
    conservativeCharacters += BigInt(estimateGemma4FlatExpansion(program, output).conservativeMinimumBytes);
    if (conservativeCharacters > options.maxCharacters) throw new Gemma4FlatFormulaLimitError(conservativeCharacters, options.maxCharacters, outputs.length);
  }
  const renderer = new FlatRenderer(program, options);
  const formulas: Record<string, string> = {};
  let characters = 0;
  for (const output of outputs) {
    const substitutions = new Map<string, string>();
    output.parameters.forEach((parameter, index) => substitutions.set(output.coordinate[index]!, String(options.outputParameters![parameter])));
    const expression = await renderer.render(output.root, substitutions, new Map());
    const key = `calc_final_${output.fixedDimension}`;
    if (formulas[key] !== undefined) throw new Error(`${key}: dimensão final duplicada.`);
    formulas[key] = expression;
    characters += key.length + expression.length;
    if (BigInt(characters) > options.maxCharacters) throw new Gemma4FlatFormulaLimitError(BigInt(characters), options.maxCharacters, outputs.length);
  }
  const invalid = [...renderer.freeVariables].filter((value) => !/^x\[.+\]$/.test(value));
  if (invalid.length) throw new Error(`Fórmula plana possui variáveis livres diferentes de x: ${invalid.slice(0, 8).join(", ")}.`);
  return { formulas, freeVariables: [...renderer.freeVariables].sort(), characters };
}

export class Gemma4FlatFormulaLimitError extends Error {
  constructor(readonly conservativeCharacters: bigint, readonly limit: bigint, readonly outputs: number) {
    super(`Expansão plana de ${outputs} saída(s) requer pelo menos ${conservativeCharacters} caracteres; limite configurado: ${limit}.`);
    this.name = "Gemma4FlatFormulaLimitError";
  }
}

class FlatRenderer {
  readonly nodes: ReadonlyMap<string, Gemma4ParametricRealNode>;
  readonly functions: ReadonlyMap<string, Gemma4ParametricExactRealProgram["operationFunctions"][number]>;
  readonly freeVariables = new Set<string>();

  constructor(program: Gemma4ParametricExactRealProgram, readonly options: Gemma4FlatFormulaGeneratorOptions) {
    this.nodes = new Map(program.expressionGraph.nodes.map((node) => [node.id, node]));
    this.functions = new Map(program.operationFunctions.map((entry) => [entry.functionId, entry]));
  }

  async render(id: string, substitutions: ReadonlyMap<string, string>, bounds: ReadonlyMap<string, string>): Promise<string> {
    const substituted = substitutions.get(id); if (substituted !== undefined) return substituted;
    const node = this.nodes.get(id); if (!node) throw new Error(`${id}: nó ausente durante substituição plana.`);
    const value = (dependency: string) => this.render(dependency, substitutions, bounds);
    const binary = async (operator: string, left: string, right: string) => `(${await value(left)} ${operator} ${await value(right)})`;
    switch (node.kind) {
      case "integer-constant": return String(node.value);
      case "integer-parameter": return node.name;
      case "integer-bound-index": return bounds.get(node.name) ?? node.name;
      case "integer-add": return binary("+", node.left, node.right); case "integer-subtract": return binary("-", node.left, node.right);
      case "integer-multiply": return binary("*", node.left, node.right); case "integer-floor-divide": return `floor(${await value(node.left)} / ${await value(node.right)})`;
      case "integer-modulo": return `mod(${await value(node.left)}, ${await value(node.right)})`;
      case "tensor-axis": throw new Error(`Fórmula plana requer dimensão fixa, encontrou axis(${node.tensor},${node.axis}).`);
      case "stable-true-prefix-rank": throw new Error(`Fórmula plana não resolveu stable_true_prefix_rank(${node.tensor}).`);
      case "rational": return rational(node.value.numerator, node.value.denominator);
      case "boolean": return String(node.value); case "negative-infinity": return "(-Infinity)"; case "positive-infinity": return "Infinity";
      case "input-element": {
        const variable = this.options.inputVariable(node.tensor, await Promise.all(node.coordinates.map(value)), node.valueType);
        this.freeVariables.add(variable); return variable;
      }
      case "learned-rational-element": {
        const rendered = await Promise.all(node.coordinates.map(value));
        const coordinates = rendered.map((coordinate) => {
          const parsed = Number(coordinate); if (!Number.isSafeInteger(parsed) || String(parsed) !== coordinate) throw new Error(`${node.tensor}: peso ainda possui coordenada simbólica ${coordinate}.`);
          return parsed;
        });
        const literal = await this.options.learnedLiteral(node.tensor, node.storageDtype, coordinates, node.decoderId);
        if (!/^-?(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?|\d+\/\d+)$/i.test(literal)) throw new Error(`${node.tensor}: learnedLiteral não retornou número objetivo: ${literal}.`);
        return literal.includes("/") ? `(${literal})` : literal;
      }
      case "function-call": {
        const function_ = this.functions.get(node.functionId); if (!function_) throw new Error(`${node.functionId}: função ausente.`);
        const arguments_ = await Promise.all(node.arguments.map(value));
        if (arguments_.length !== function_.parameters.length) throw new Error(`${node.functionId}: aridade divergente.`);
        const called = new Map<string, string>(); function_.parameters.forEach((parameter, index) => called.set(parameter.node, arguments_[index]!));
        return this.render(function_.root, called, new Map());
      }
      case "add": return parenthesized(" + ", await Promise.all(node.arguments.map(value)));
      case "multiply": return parenthesized(" * ", await Promise.all(node.arguments.map(value)));
      case "minimum": return `min(${(await Promise.all(node.arguments.map(value))).join(", ")})`;
      case "maximum": return `max(${(await Promise.all(node.arguments.map(value))).join(", ")})`;
      case "divide": return binary("/", node.numerator, node.denominator); case "integer-power": return `(${await value(node.base)} ** ${node.exponent})`;
      case "power": return `(${await value(node.base)} ** ${await value(node.exponent)})`;
      case "unary-function": return `${node.function}(${await value(node.argument)})`;
      case "compare": return binary(comparison(node.comparison), node.left, node.right);
      case "select": return `select(${await value(node.condition)}, ${await value(node.whenTrue)}, ${await value(node.whenFalse)})`;
      case "finite-sum": case "finite-maximum": {
        const start = exactInteger(await value(node.startInclusive)), end = exactInteger(await value(node.endExclusive));
        if (end < start || end - start > this.options.maxUnrolledReductionTerms) throw new Error(`${node.kind}: redução ${start}..${end} excede o limite de desenrolamento.`);
        const terms: string[] = [];
        for (let index = start; index < end; index += 1) {
          const iteration = new Map(bounds).set(node.index, String(index));
          if (node.predicate) terms.push(`select(${await this.render(node.predicate, substitutions, iteration)}, ${await this.render(node.body, substitutions, iteration)}, ${node.kind === "finite-sum" ? "0" : "(-Infinity)"})`);
          else terms.push(await this.render(node.body, substitutions, iteration));
        }
        return node.kind === "finite-sum" ? parenthesized(" + ", terms) : `max(${terms.join(", ")})`;
      }
    }
  }
}

function parenthesized(separator: string, values: readonly string[]): string { return values.length === 1 ? values[0]! : `(${values.join(separator)})`; }
function exactInteger(value: string): number { const parsed = Number(value); if (!Number.isSafeInteger(parsed) || String(parsed) !== value) throw new Error(`Inteiro fixo esperado, recebeu ${value}.`); return parsed; }
function rational(numerator: string, denominator: string): string { return denominator === "1" ? numerator : `(${numerator}/${denominator})`; }
function comparison(value: "equal" | "not-equal" | "less" | "less-equal" | "greater" | "greater-equal"): string { return ({ equal: "==", "not-equal": "!=", less: "<", "less-equal": "<=", greater: ">", "greater-equal": ">=" } as const)[value]; }
