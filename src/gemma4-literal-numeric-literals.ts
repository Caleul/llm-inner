import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4LiteralGenerationForwardCalculationContract,
  Gemma4LiteralGenerationScalarCalculations,
} from "./gemma4-literal-generation-calculations.js";
import type { Gemma4LiteralScalarCalculations } from "./gemma4-literal-scalar-calculations.js";

export interface Gemma4LiteralNumericLiteralUse {
  section: "forward" | "generation" | "cache-transition";
  definitionId: string;
  scope?: string;
}

/**
 * Exact binary interpretations for one numeric token used by the serialized
 * formula language.  The formula's surrounding cast selects the relevant
 * representation; carrying all three prevents a reader from reparsing a
 * decimal through a host-specific conversion or guessing a BF16 tie rule.
 */
export interface Gemma4LiteralNumericLiteral {
  id: string;
  token: string;
  kind: "decimal-token" | "named-mathematical-constant";
  finite: boolean;
  binary64Hex: string;
  binary32Hex: string;
  bfloat16Hex: string;
  bfloat16Rounding: "round-to-nearest-ties-to-even";
  uses: Gemma4LiteralNumericLiteralUse[];
}

export interface Gemma4LiteralNumericLiterals {
  kind: "gemma4-literal-numeric-literals";
  schemaVersion: 1;
  tokenSemantics: "formula token decoded from declared IEEE bits; surrounding F32/BF16/F64 cast selects representation";
  literals: Gemma4LiteralNumericLiteral[];
}

interface CollectedUse extends Gemma4LiteralNumericLiteralUse {
  token: string;
  kind: Gemma4LiteralNumericLiteral["kind"];
}

const NUMBER_TOKEN = /(?<![A-Za-z0-9_.])(?:-Infinity|Infinity|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?![A-Za-z0-9_.])/g;
const NAMED_CONSTANT = /(?<![A-Za-z0-9_.])pi(?![A-Za-z0-9_.])/g;

export function buildGemma4LiteralNumericLiterals(
  forward: Gemma4LiteralScalarCalculations,
  generation: Gemma4LiteralGenerationScalarCalculations,
  generationForward: Gemma4LiteralGenerationForwardCalculationContract,
): Gemma4LiteralNumericLiterals {
  const collected: CollectedUse[] = [];
  for (const calculation of forward.assignments) {
    collect(calculation.formula, {
      section: "forward",
      scope: calculation.scope,
      definitionId: calculation.definitionId,
    }, collected);
  }
  for (const calculation of generation.assignments) {
    for (const formula of calculation.scalarAssignments) {
      collect(formula, { section: "generation", definitionId: calculation.definitionId }, collected);
    }
  }
  for (const transition of generationForward.cacheTransitions) {
    for (const [phase, program] of [["prefill", transition.prefill], ["incremental", transition.incremental]] as const) {
      for (const formula of program.scalarAssignments) {
        collect(formula, {
          section: "cache-transition",
          scope: transition.ownership,
          definitionId: `layer_${transition.layer}_${phase}`,
        }, collected);
      }
    }
  }
  const byToken = new Map<string, { kind: Gemma4LiteralNumericLiteral["kind"]; uses: Gemma4LiteralNumericLiteralUse[] }>();
  for (const entry of collected) {
    const current = byToken.get(entry.token) ?? { kind: entry.kind, uses: [] };
    const use = { section: entry.section, definitionId: entry.definitionId, ...(entry.scope ? { scope: entry.scope } : {}) };
    if (!current.uses.some((candidate) => isDeepStrictEqual(candidate, use))) current.uses.push(use);
    byToken.set(entry.token, current);
  }
  const literals = [...byToken.entries()].sort(([left], [right]) => left.localeCompare(right, "en"))
    .map(([token, entry], index): Gemma4LiteralNumericLiteral => {
      const value = token === "pi" ? Math.PI : Number(token);
      if (Number.isNaN(value)) throw new Error(`Token numérico Gemma 4 inválido: ${token}.`);
      return {
        id: `numeric_literal_${String(index).padStart(4, "0")}`,
        token,
        kind: entry.kind,
        finite: Number.isFinite(value),
        binary64Hex: f64Bits(value),
        binary32Hex: f32Bits(value),
        bfloat16Hex: bf16Bits(value),
        bfloat16Rounding: "round-to-nearest-ties-to-even",
        uses: entry.uses,
      };
    });
  if (literals.length === 0) throw new Error("Programa Gemma 4 não possui literais numéricos endereçáveis.");
  return {
    kind: "gemma4-literal-numeric-literals",
    schemaVersion: 1,
    tokenSemantics: "formula token decoded from declared IEEE bits; surrounding F32/BF16/F64 cast selects representation",
    literals,
  };
}

export function validateGemma4LiteralNumericLiterals(
  actual: Gemma4LiteralNumericLiterals,
  forward: Gemma4LiteralScalarCalculations,
  generation: Gemma4LiteralGenerationScalarCalculations,
  generationForward: Gemma4LiteralGenerationForwardCalculationContract,
): void {
  if (!isDeepStrictEqual(actual, buildGemma4LiteralNumericLiterals(forward, generation, generationForward))) {
    throw new Error("Programa literal Gemma 4 possui tabela de bits numéricos ausente ou divergente.");
  }
}

function collect(formula: string, use: Gemma4LiteralNumericLiteralUse, output: CollectedUse[]): void {
  for (const match of formula.matchAll(NUMBER_TOKEN)) output.push({ ...use, token: match[0], kind: "decimal-token" });
  for (const match of formula.matchAll(NAMED_CONSTANT)) output.push({ ...use, token: match[0], kind: "named-mathematical-constant" });
}

function f64Bits(value: number): string {
  const bytes = new ArrayBuffer(8);
  new DataView(bytes).setFloat64(0, value, false);
  return `0x${new DataView(bytes).getBigUint64(0, false).toString(16).padStart(16, "0")}`;
}

function f32Bits(value: number): string {
  const bytes = new ArrayBuffer(4);
  new DataView(bytes).setFloat32(0, value, false);
  return `0x${new DataView(bytes).getUint32(0, false).toString(16).padStart(8, "0")}`;
}

function bf16Bits(value: number): string {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, false);
  const bits = view.getUint32(0, false);
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 0;
  return `0x${(rounded >>> 16).toString(16).padStart(4, "0")}`;
}
