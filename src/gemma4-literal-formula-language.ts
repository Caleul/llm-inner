import { isDeepStrictEqual } from "node:util";
import type { Gemma4LiteralGenerationScalarCalculations } from "./gemma4-literal-generation-calculations.js";
import type { Gemma4LiteralScalarCalculations } from "./gemma4-literal-scalar-calculations.js";

export interface Gemma4LiteralFormulaLanguageContract {
  kind: "gemma4-literal-formula-language-contract";
  schemaVersion: 1;
  languageId: "indexed-ieee754-expression-v1";
  authority: {
    forwardAssignments: "/scalarCalculations/assignments";
    generationAssignments: "/generation/scalarCalculations/assignments";
    instantiatedForwardOrder: "/calculationGraph/assignments";
    generationForwardOrder: "/generation/forwardCalculation/operationOrder";
    learnedOperandBindings: "/learnedOperands/assignments";
    storageDecoders: "/storageDecoders";
    numericLiteralBits: "/numericLiterals/literals";
    calculationDomains: "/calculationDomains/assignments";
  };
  evaluation: {
    dependencyOrder: string;
    coordinateOrder: string;
    inputBinding: string;
    numericTokenBinding: string;
    learnedValueBinding: string;
    generationOrder: string;
    invalidOperation: string;
  };
  indexing: {
    origin: 0;
    endConvention: string;
    tensorLayout: string;
    bounds: string;
    aliases: string;
  };
  scalarTypes: Array<{ name: "F64" | "F32" | "BF16" | "I32" | "BOOL"; semantics: string }>;
  operators: Array<{ notation: string; semantics: string }>;
  reductions: {
    ascendingLexicographic: string;
    operationDeclared: string;
    runtimeDefined: string;
    multipleDomains: string;
  };
  intrinsics: Array<{ notation: string; semantics: string }>;
}

/**
 * The formulas have always been stored in the artifact, but the meaning of
 * their notation used to live in this repository's reader and documentation.
 * This declaration makes the interpretation itself artifact data. JSON
 * pointers bind every rule to the finite sections needed for independent
 * execution; no source checkpoint or implicit host arithmetic is authoritative.
 */
export function buildGemma4LiteralFormulaLanguageContract(): Gemma4LiteralFormulaLanguageContract {
  return {
    kind: "gemma4-literal-formula-language-contract",
    schemaVersion: 1,
    languageId: "indexed-ieee754-expression-v1",
    authority: {
      forwardAssignments: "/scalarCalculations/assignments",
      generationAssignments: "/generation/scalarCalculations/assignments",
      instantiatedForwardOrder: "/calculationGraph/assignments",
      generationForwardOrder: "/generation/forwardCalculation/operationOrder",
      learnedOperandBindings: "/learnedOperands/assignments",
      storageDecoders: "/storageDecoders",
      numericLiteralBits: "/numericLiterals/literals",
      calculationDomains: "/calculationDomains/assignments",
    },
    evaluation: {
      dependencyOrder: "evaluate instantiated assignments by ascending ordinal; every predecessor must already exist",
      coordinateOrder: "row-major lexicographic over the complete declared output domain; preview windows never change evaluation",
      inputBinding: "bind orderedInputs positionally at each instantiated call site before evaluating the indexed formula",
      numericTokenBinding: "resolve every decimal or named mathematical token through numericLiterals and select bits by its surrounding F64/F32/BF16 cast",
      learnedValueBinding: "resolve decode(role)[indices] through learnedOperands then the matching storageDecoder and embedded constant",
      generationOrder: "evaluate generation scalarAssignments in array order and iterations in ascending step order until the declared stop predicate",
      invalidOperation: "fail closed before producing an output; never infer a default, host reduction, tensor layout, cast, or missing intrinsic",
    },
    indexing: {
      origin: 0,
      endConvention: "inclusive when written a..b; exclusive only when explicitly written endExclusive",
      tensorLayout: "row-major unless the referenced calculation domain or cache transition declares another layout",
      bounds: "every symbolic coordinate is bounded by calculationDomains; out-of-domain reads are errors unless the formula explicitly defines padding",
      aliases: "reshape, transpose, row_major_alias and indexed predecessor references preserve exact elements and perform no arithmetic cast",
    },
    scalarTypes: [
      { name: "F64", semantics: "IEEE-754 binary64 round-to-nearest ties-to-even; F64(expr) materializes one rounding boundary" },
      { name: "F32", semantics: "IEEE-754 binary32 round-to-nearest ties-to-even; F32(expr) materializes one rounding boundary" },
      { name: "BF16", semantics: "bfloat16 round-to-nearest ties-to-even from F32 bits; BF16(expr) materializes then widens exactly to F32 for JSON-visible values" },
      { name: "I32", semantics: "signed exact 32-bit integer; token IDs and positions must additionally satisfy each declared input domain" },
      { name: "BOOL", semantics: "exact false or true; comparisons and mask predicates do not perform numeric narrowing" },
    ],
    operators: [
      { notation: "a=b", semantics: "assign the right scalar exactly once to the indexed scalar on the left" },
      { notation: "a+b, a-b, a*b, a/b, -a", semantics: "evaluate operands left-to-right; arithmetic precision changes only at an explicit cast or declared reduction/FMA boundary" },
      { notation: "a**b", semantics: "real exponentiation followed by the surrounding declared cast; package formulas additionally pin source-visible pow/rsqrt decompositions where fidelity requires them" },
      { notation: "predicate ? a : b", semantics: "evaluate the predicate then only the selected branch" },
      { notation: "==, >, >=, <, <=, &&, ||, !", semantics: "exact comparison or short-circuit Boolean operation over already materialized operands" },
      { notation: "tensor[i,j,...]", semantics: "zero-based indexed read using the referenced domain and layout; commas order axes exactly as declared" },
    ],
    reductions: {
      ascendingLexicographic: "initialize with the formula-declared identity and evaluate every declared index in ascending lexicographic order with each visible cast applied immediately",
      operationDeclared: "execute the serialized reduction.schedule literally, including product rounding, lane assignment, block/tile order, FMA behavior, fold order, tails, accumulation dtype and output cast",
      runtimeDefined: "not executable: reproducibility must be fail-closed-runtime-reduction and every scalar renderer or replay claiming literal fidelity must reject it",
      multipleDomains: "nested REDUCE domains execute left-to-right as written; a schedule attached to the assignment overrides only the reduction indices named beside it",
    },
    intrinsics: [
      { notation: "decode(role)[indices]", semantics: "resolve role through the current learnedOperands binding; apply its logical index expression, row-major offset and matching exact storageDecoder to embedded bytes" },
      { notation: "REDUCE(index-domain, expression)", semantics: "evaluate the complete domain using the assignment reduction declaration; absence of a reduction schedule means ascending lexicographic order" },
      { notation: "F32_FMA(acc,a,b)", semantics: "compute exact a*b+acc then round once to IEEE binary32" },
      { notation: "min, max, floor", semantics: "IEEE minimum/maximum over materialized operands and mathematical floor; NaN is invalid unless an assignment explicitly permits it" },
      { notation: "sqrt, rsqrt", semantics: "sqrt is correctly rounded by the pinned runtime policy; rsqrt(x) is F32(1/F32(sqrt(x))) unless the assignment declares another serialized implementation" },
      { notation: "SLEEF_EXP_F32, SLEEF_SIN_F32, SLEEF_COS_F32, SLEEF_TANH_F32", semantics: "execute the exact SLEEF implementation identifier and cast boundaries serialized by the owning operation; lowercase exp/sin/cos/tanh in a formula are aliases only when that same operation declares the corresponding implementation" },
      { notation: "log, log1p", semantics: "evaluate the mathematical function at the operation-declared compute dtype and materialize every surrounding cast in formula order" },
      { notation: "concat, tuple, STRUCT", semantics: "construct values in argument order without arithmetic conversion; concat uses the axis named by the formula or cache transition" },
      { notation: "row_major_alias, reshape, transpose", semantics: "change only logical indexing/layout exactly as written; preserve every source bit" },
      { notation: "EVALUATE(reference in ordinal order)", semantics: "inline the finite referenced calculationGraph assignments with positional bindings; it is never a generic architecture or hidden decoder invocation" },
      { notation: "argmax-lowest-token-id", semantics: "scan token IDs in ascending order and replace the winner only on strictly greater F32 logits; equality retains the lowest ID" },
      { notation: "exact_safe_integer", semantics: "perform exact integer arithmetic and fail if the result is not a safe integer or violates its declared input domain" },
      { notation: "padding/mask/rotate/theta helpers named by a formula", semantics: "use only the explicit index mapping, predicate and constants written in that formula plus its calculation domain; no host or framework default is permitted" },
    ],
  };
}

export function validateGemma4LiteralFormulaLanguageContract(
  contract: Gemma4LiteralFormulaLanguageContract,
  forward: Gemma4LiteralScalarCalculations,
  generation: Gemma4LiteralGenerationScalarCalculations,
): void {
  if (!isDeepStrictEqual(contract, buildGemma4LiteralFormulaLanguageContract()) ||
    forward.formulaLanguage !== contract.languageId || generation.formulaLanguage !== contract.languageId) {
    throw new Error("Programa literal Gemma 4 possui linguagem de fórmulas ausente ou divergente.");
  }
}
