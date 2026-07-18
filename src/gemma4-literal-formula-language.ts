import { isDeepStrictEqual } from "node:util";
import type { Gemma4LiteralGenerationScalarCalculations } from "./gemma4-literal-generation-calculations.js";
import type { Gemma4LiteralScalarCalculations } from "./gemma4-literal-scalar-calculations.js";
import { gemma4LiteralNormalizationReductionPrograms } from "./gemma4-literal-normalization-reduction-view.js";

export interface Gemma4LiteralFormulaLanguageContract {
  kind: "gemma4-literal-formula-language-contract";
  schemaVersion: 8;
  languageId: "indexed-ieee754-expression-v1";
  authority: {
    forwardAssignments: "/scalarCalculations/assignments";
    generationAssignments: "/generation/scalarCalculations/assignments";
    instantiatedForwardOrder: "/calculationGraph/assignments";
    generationForwardOrder: "/generation/forwardCalculation/operationOrder";
    learnedOperandBindings: "/learnedOperands/assignments";
    learnedIndexLanguage: "/learnedOperands/indexLanguage";
    storageDecoders: "/storageDecoders";
    denseDecoderLanguage: "/denseDecoderLanguage";
    transcendentalPrograms: "/transcendentalPrograms";
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
    programs: Gemma4LiteralIndexingPrograms;
  };
  scalarTypes: Array<{ name: "F64" | "F32" | "BF16" | "I32" | "BOOL"; semantics: string }>;
  operators: Array<{ notation: string; semantics: string }>;
  reductions: {
    ascendingLexicographic: string;
    operationDeclared: string;
    runtimeDefined: string;
    multipleDomains: string;
    stagedReductions: string;
    normalizationPrograms: ReturnType<typeof gemma4LiteralNormalizationReductionPrograms>;
    softmaxPrograms: Gemma4LiteralSoftmaxReductionPrograms;
  };
  intrinsics: Array<{ notation: string; semantics: string }>;
}

export interface Gemma4LiteralIndexingPrograms {
  kind: "gemma4-literal-indexing-programs";
  schemaVersion: 2;
  contiguousVisionGroupId: string[];
  stableTrueCount: string[];
  stableTruePrefixRank: string[];
  stableTrueCoordinateAtRank: string[];
}

export function gemma4LiteralIndexingPrograms(): Gemma4LiteralIndexingPrograms {
  return {
    kind: "gemma4-literal-indexing-programs",
    schemaVersion: 2,
    contiguousVisionGroupId: [
      "require mm_token_type_ids to be an I32 row and sequence to be an in-bounds I32 coordinate",
      "group=I32(-1); previous_vision=false",
      "for index=0..sequence in ascending order: vision=(mm_token_type_ids[index]==1 || mm_token_type_ids[index]==2); if vision && !previous_vision then group=I32(group+1); previous_vision=vision",
      "result=(mm_token_type_ids[sequence]==1 || mm_token_type_ids[sequence]==2) ? group : I32(-1)",
    ],
    stableTrueCount: [
      "require mask to be a rectangular BOOL matrix",
      "count=I32(0)",
      "for batch=0..rows-1 ascending: for sequence=0..columns-1 ascending: if mask[batch,sequence] count=I32(count+1)",
      "result=count",
    ],
    stableTruePrefixRank: [
      "require mask to be a rectangular BOOL matrix and [batch,sequence] to be an in-bounds I32 coordinate",
      "rank=I32(0)",
      "for prior_batch=0..batch ascending: for prior_sequence=0..columns-1 ascending: stop before [batch,sequence]; if mask[prior_batch,prior_sequence] rank=I32(rank+1)",
      "result=mask[batch,sequence] ? rank : I32(-1)",
    ],
    stableTrueCoordinateAtRank: [
      "require mask to be a rectangular BOOL matrix and rank to be I32 in 0..STABLE_TRUE_COUNT(mask)-1",
      "current=I32(0)",
      "for batch=0..rows-1 ascending: for sequence=0..columns-1 ascending: if mask[batch,sequence] and current==rank return STRUCT(batch=batch,sequence=sequence); if mask[batch,sequence] current=I32(current+1)",
      "absence of a returned coordinate is fail-closed",
    ],
  };
}

/** Counts true rows in the exact stable batch-major order serialized in the artifact. */
export function executeGemma4LiteralStableTrueCount(
  programs: Gemma4LiteralIndexingPrograms,
  mask: readonly (readonly boolean[])[],
): number {
  assertCanonicalIndexingPrograms(programs);
  const columns = validateBooleanMatrix(mask);
  let count = 0;
  for (let batch = 0; batch < mask.length; batch += 1) for (let sequence = 0; sequence < columns; sequence += 1) {
    if (mask[batch]![sequence]) count += 1;
  }
  return count;
}

/** Returns the zero-based true-row rank before a selected coordinate, or -1 when false. */
export function executeGemma4LiteralStableTruePrefixRank(
  programs: Gemma4LiteralIndexingPrograms,
  mask: readonly (readonly boolean[])[],
  batch: number,
  sequence: number,
): number {
  assertCanonicalIndexingPrograms(programs);
  const columns = validateBooleanMatrix(mask);
  if (!Number.isSafeInteger(batch) || !Number.isSafeInteger(sequence) || batch < 0 || batch >= mask.length || sequence < 0 || sequence >= columns) {
    throw new Error("Programa de prefixo BOOL Gemma 4 requer coordenada válida.");
  }
  if (!mask[batch]![sequence]) return -1;
  let rank = 0;
  for (let priorBatch = 0; priorBatch <= batch; priorBatch += 1) for (let priorSequence = 0; priorSequence < columns; priorSequence += 1) {
    if (priorBatch === batch && priorSequence === sequence) return rank;
    if (mask[priorBatch]![priorSequence]) rank += 1;
  }
  throw new Error("Programa de prefixo BOOL Gemma 4 não alcançou a coordenada declarada.");
}

/** Selects the batch/sequence coordinate of a true row by stable zero-based rank. */
export function executeGemma4LiteralStableTrueCoordinateAtRank(
  programs: Gemma4LiteralIndexingPrograms,
  mask: readonly (readonly boolean[])[],
  rank: number,
): [number, number] {
  assertCanonicalIndexingPrograms(programs);
  const columns = validateBooleanMatrix(mask);
  const count = executeGemma4LiteralStableTrueCount(programs, mask);
  if (!Number.isSafeInteger(rank) || rank < 0 || rank >= count) {
    throw new Error("Programa de seleção BOOL Gemma 4 requer rank válido.");
  }
  let current = 0;
  for (let batch = 0; batch < mask.length; batch += 1) for (let sequence = 0; sequence < columns; sequence += 1) {
    if (!mask[batch]![sequence]) continue;
    if (current === rank) return [batch, sequence];
    current += 1;
  }
  throw new Error("Programa de seleção BOOL Gemma 4 não encontrou o rank declarado.");
}

function assertCanonicalIndexingPrograms(programs: Gemma4LiteralIndexingPrograms): void {
  if (!isDeepStrictEqual(programs, gemma4LiteralIndexingPrograms())) {
    throw new Error("Programa de indexação multimodal Gemma 4 ausente ou alterado.");
  }
}

function validateBooleanMatrix(mask: readonly (readonly boolean[])[]): number {
  const columns = mask[0]?.length;
  if (columns === undefined || columns === 0 || mask.some((row) => row.length !== columns || row.some((value) => typeof value !== "boolean"))) {
    throw new Error("Programa de indexação BOOL Gemma 4 requer matriz retangular não vazia.");
  }
  return columns;
}

/** Executes only the serialized contiguous-run contract used by every Gemma 4 vision-block mask. */
export function executeGemma4LiteralContiguousVisionGroupId(
  programs: Gemma4LiteralIndexingPrograms,
  mmTokenTypeIds: readonly number[],
  sequence: number,
): number {
  assertCanonicalIndexingPrograms(programs);
  if (!Number.isSafeInteger(sequence) || sequence < 0 || sequence >= mmTokenTypeIds.length ||
    mmTokenTypeIds.some((value) => !Number.isSafeInteger(value))) {
    throw new Error("Programa de grupos vision Gemma 4 requer linha I32 e coordenada válida.");
  }
  let group = -1;
  let previousVision = false;
  for (let index = 0; index <= sequence; index += 1) {
    const type = mmTokenTypeIds[index]!;
    const vision = type === 1 || type === 2;
    if (vision && !previousVision) group += 1;
    previousVision = vision;
  }
  const current = mmTokenTypeIds[sequence]!;
  return current === 1 || current === 2 ? group : -1;
}

export interface Gemma4LiteralSoftmaxReductionPrograms {
  kind: "gemma4-literal-softmax-reduction-programs";
  schemaVersion: 1;
  orderedF32Maximum: string[];
  orderedF32Sum: string[];
  pytorchF32VectorPairwiseMaximum: string[];
  pytorchF32VectorPairwiseSum: string[];
}

export type Gemma4LiteralSoftmaxReductionProgramName =
  | "ORDERED_F32_REDUCE_MAX"
  | "ORDERED_F32_REDUCE_SUM"
  | "PYTORCH_F32_VECTOR_REDUCE_MAX"
  | "PYTORCH_F32_VECTOR_REDUCE_SUM";

export function gemma4LiteralSoftmaxReductionPrograms(): Gemma4LiteralSoftmaxReductionPrograms {
  return {
    kind: "gemma4-literal-softmax-reduction-programs",
    schemaVersion: 1,
    orderedF32Maximum: [
      "acc=-Infinity",
      "for index in the complete declared domain in ascending order: if predicate is absent or true, acc=max(acc,value[index]); NaN is fail-closed",
      "result=acc; result==-Infinity is fail-closed when the owning softmax requires at least one valid element",
    ],
    orderedF32Sum: [
      "acc=F32(0)",
      "for index in the complete declared domain in ascending order: if predicate is absent or true, acc=F32(acc+value[index])",
      "result=acc",
    ],
    pytorchF32VectorPairwiseMaximum: [
      "lanes=4; if length==0 result=-Infinity; if 0<length<4 scan values[0..length-1] with max in ascending order",
      "otherwise accumulator[lane]=values[lane] for lane=0..3",
      "for base=4; base<length-(length%4); base+=4: accumulator[lane]=max(accumulator[lane],values[base+lane]) for lane=0..3 ascending",
      "for lane=0 while base+lane<length: accumulator[lane]=max(accumulator[lane],values[base+lane])",
      "result=max(accumulator[0],accumulator[1],accumulator[2],accumulator[3]) evaluated left-to-right; NaN is fail-closed",
    ],
    pytorchF32VectorPairwiseSum: [
      "lanes=4; if length==0 result=F32(0); if 0<length<4 scan values[0..length-1] with F32(acc+value) in ascending order",
      "otherwise accumulator[lane]=values[lane] for lane=0..3",
      "for base=4; base<length-(length%4); base+=4: accumulator[lane]=F32(accumulator[lane]+values[base+lane]) for lane=0..3 ascending",
      "for lane=0 while base+lane<length: accumulator[lane]=F32(accumulator[lane]+values[base+lane])",
      "result=F32(F32(accumulator[0]+accumulator[1])+F32(accumulator[2]+accumulator[3]))",
    ],
  };
}

/** Executes only the canonical serialized contract; altered artifact programs fail closed. */
export function executeGemma4LiteralSoftmaxReductionProgram(
  programs: Gemma4LiteralSoftmaxReductionPrograms,
  name: Gemma4LiteralSoftmaxReductionProgramName,
  values: readonly number[],
  predicate?: readonly boolean[],
): number {
  if (!isDeepStrictEqual(programs, gemma4LiteralSoftmaxReductionPrograms())) {
    throw new Error("Programa de redução softmax Gemma 4 ausente ou alterado.");
  }
  if (predicate && predicate.length !== values.length) throw new Error("Predicado de redução softmax possui cardinalidade divergente.");
  if (predicate && name.startsWith("PYTORCH_F32_VECTOR_")) {
    throw new Error("Programa vetorial PyTorch não aceita compactação implícita por predicado.");
  }
  const selected = predicate ? values.filter((_, index) => predicate[index]) : [...values];
  if (selected.some((value) => Number.isNaN(value))) throw new Error("Redução softmax Gemma 4 não aceita NaN.");
  if (name === "ORDERED_F32_REDUCE_MAX") return orderedMaximum(selected);
  if (name === "ORDERED_F32_REDUCE_SUM") return orderedSum(selected);
  return pytorchPairwiseReduce(selected, name === "PYTORCH_F32_VECTOR_REDUCE_MAX" ? "maximum" : "sum");
}

function orderedMaximum(values: readonly number[]): number {
  let result = -Infinity;
  for (const value of values) result = Math.max(result, value);
  return result;
}

function orderedSum(values: readonly number[]): number {
  let result = Math.fround(0);
  for (const value of values) result = Math.fround(result + value);
  return result;
}

function pytorchPairwiseReduce(values: readonly number[], operation: "maximum" | "sum"): number {
  if (values.length === 0) return operation === "maximum" ? -Infinity : Math.fround(0);
  if (values.length < 4) {
    let result = values[0]!;
    for (let index = 1; index < values.length; index += 1) {
      result = operation === "maximum" ? Math.max(result, values[index]!) : Math.fround(result + values[index]!);
    }
    return result;
  }
  const accumulators = values.slice(0, 4);
  let index = 4;
  for (; index < values.length - (values.length % 4); index += 4) for (let lane = 0; lane < 4; lane += 1) {
    accumulators[lane] = operation === "maximum"
      ? Math.max(accumulators[lane]!, values[index + lane]!)
      : Math.fround(accumulators[lane]! + values[index + lane]!);
  }
  for (let lane = 0; index + lane < values.length; lane += 1) {
    accumulators[lane] = operation === "maximum"
      ? Math.max(accumulators[lane]!, values[index + lane]!)
      : Math.fround(accumulators[lane]! + values[index + lane]!);
  }
  if (operation === "maximum") return Math.max(accumulators[0]!, accumulators[1]!, accumulators[2]!, accumulators[3]!);
  return Math.fround(Math.fround(accumulators[0]! + accumulators[1]!) + Math.fround(accumulators[2]! + accumulators[3]!));
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
    schemaVersion: 8,
    languageId: "indexed-ieee754-expression-v1",
    authority: {
      forwardAssignments: "/scalarCalculations/assignments",
      generationAssignments: "/generation/scalarCalculations/assignments",
      instantiatedForwardOrder: "/calculationGraph/assignments",
      generationForwardOrder: "/generation/forwardCalculation/operationOrder",
      learnedOperandBindings: "/learnedOperands/assignments",
      learnedIndexLanguage: "/learnedOperands/indexLanguage",
      storageDecoders: "/storageDecoders",
      denseDecoderLanguage: "/denseDecoderLanguage",
      transcendentalPrograms: "/transcendentalPrograms",
      numericLiteralBits: "/numericLiterals/literals",
      calculationDomains: "/calculationDomains/assignments",
    },
    evaluation: {
      dependencyOrder: "evaluate instantiated assignments by ascending ordinal; every predecessor must already exist",
      coordinateOrder: "row-major lexicographic over the complete declared output domain; preview windows never change evaluation",
      inputBinding: "bind orderedInputs positionally at each instantiated call site before evaluating the indexed formula",
      numericTokenBinding: "resolve forward/generation decimal or named mathematical tokens through numericLiterals; resolve transcendental-program names through transcendentalPrograms.constants; select bits by the surrounding F64/F32/BF16 cast",
      learnedValueBinding: "evaluate learnedOperands.logicalIndices with its embedded integer-expression AST, then execute the matching storageDecoder address and decode expression ASTs under denseDecoderLanguage over the embedded constant bytes",
      generationOrder: "evaluate generation scalarAssignments in array order and iterations in ascending step order until the declared stop predicate",
      invalidOperation: "fail closed before producing an output; never infer a default, host reduction, tensor layout, cast, or missing intrinsic",
    },
    indexing: {
      origin: 0,
      endConvention: "inclusive when written a..b; exclusive only when explicitly written endExclusive",
      tensorLayout: "row-major unless the referenced calculation domain or cache transition declares another layout",
      bounds: "every symbolic coordinate is bounded by calculationDomains; out-of-domain reads are errors unless the formula explicitly defines padding",
      aliases: "reshape, transpose, row_major_alias and indexed predecessor references preserve exact elements and perform no arithmetic cast",
      programs: gemma4LiteralIndexingPrograms(),
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
      stagedReductions: "execute reductionStages in array order; each stage binds exactly the indices, identity, predicate, program and optional operation-declared schedule named by the formula",
      normalizationPrograms: gemma4LiteralNormalizationReductionPrograms(),
      softmaxPrograms: gemma4LiteralSoftmaxReductionPrograms(),
    },
    intrinsics: [
      { notation: "decode(role)[indices]", semantics: "resolve role through the current learnedOperands binding; evaluate its gemma4-learned-index-expression-v1 AST, then execute storageDecoder.address for the exact byte range and storageDecoder.decode for the exact F32 result bits" },
      { notation: "REDUCE(index-domain, expression)", semantics: "evaluate the complete domain using the assignment reduction declaration; absence of a reduction schedule means ascending lexicographic order" },
      { notation: "ORDERED_F32_REDUCE_MAX, ORDERED_F32_REDUCE_SUM", semantics: "execute reductions.softmaxPrograms.orderedF32Maximum or orderedF32Sum over the complete named stage domain and predicate" },
      { notation: "PYTORCH_F32_VECTOR_REDUCE_MAX, PYTORCH_F32_VECTOR_REDUCE_SUM", semantics: "execute reductions.softmaxPrograms.pytorchF32VectorPairwiseMaximum or pytorchF32VectorPairwiseSum exactly with four lanes and the serialized tail/fold order" },
      { notation: "ORDERED_F32_DOT", semantics: "initialize F32(0), visit the complete named index domain in ascending order, materialize each F32 product, and immediately assign acc=F32(acc+product)" },
      { notation: "ARM_NEON_BF16_DOT_F32", semantics: "execute the owning reductionStages schedule literally; products, lanes, FMA boundaries, block order, tail and horizontal fold are all taken from that serialized schedule" },
      { notation: "exact_product(a*b)", semantics: "retain the exact real product of the two already materialized operands until the immediately enclosing operation-declared FMA/add boundary; it is invalid outside a reduction schedule whose product boundary is fused" },
      { notation: "F32_FMA(acc,a,b)", semantics: "compute exact a*b+acc then round once to IEEE binary32" },
      { notation: "min, max, floor", semantics: "IEEE minimum/maximum over materialized operands and mathematical floor; NaN is invalid unless an assignment explicitly permits it" },
      { notation: "ARM_SQRT_F32, PYTORCH_POW_NEGATIVE_HALF_F32", semantics: "execute the matching finite /transcendentalPrograms bit-search or reciprocal-square-root program; no host sqrt/pow fallback is permitted" },
      { notation: "SLEEF_EXP_F32, SLEEF_SIN_F32, SLEEF_COS_F32, SLEEF_TANH_F32, SLEEF_LOG1P_F32", semantics: "execute the matching finite /transcendentalPrograms program, its exact binary32 constants, pair/FMA subprograms, special-value branches and embedded rempi table; no external SLEEF source, binary or host libm fallback is permitted" },
      { notation: "concat, tuple, STRUCT", semantics: "construct values in argument order without arithmetic conversion; concat uses the axis named by the formula or cache transition" },
      { notation: "row_major_alias, reshape, transpose", semantics: "change only logical indexing/layout exactly as written; preserve every source bit" },
      { notation: "CONTIGUOUS_VISION_GROUP_ID", semantics: "execute indexing.programs.contiguousVisionGroupId over the declared mm_token_type_ids row through the requested sequence coordinate" },
      { notation: "STABLE_TRUE_COUNT", semantics: "execute indexing.programs.stableTrueCount over the complete rectangular BOOL matrix in batch-major order" },
      { notation: "STABLE_TRUE_PREFIX_RANK", semantics: "execute indexing.programs.stableTruePrefixRank and return the number of true coordinates preceding [batch,sequence], or I32(-1) when the requested coordinate is false" },
      { notation: "STABLE_TRUE_COORDINATE_AT_RANK", semantics: "execute indexing.programs.stableTrueCoordinateAtRank and return the unique [batch,sequence] coordinate of the requested zero-based true-row rank" },
      { notation: "EVALUATE(reference in ordinal order)", semantics: "inline the finite referenced calculationGraph assignments with positional bindings; it is never a generic architecture or hidden decoder invocation" },
      { notation: "argmax-lowest-token-id", semantics: "scan token IDs in ascending order and replace the winner only on strictly greater F32 logits; equality retains the lowest ID" },
      { notation: "exact_safe_integer", semantics: "perform exact integer arithmetic and fail if the result is not a safe integer or violates its declared input domain" },
      { notation: "padding, mask, modal-token, RoPE pairing and angle expressions", semantics: "these must be written explicitly in each formula from indexed inputs and declared constants; an unregistered function-like helper is invalid" },
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
  validateGemma4LiteralFormulaFunctionCoverage([
    ...forward.assignments.map((assignment) => assignment.formula),
    ...generation.assignments.flatMap((assignment) => assignment.scalarAssignments),
  ]);
}

const REGISTERED_FUNCTIONS = new Set([
  "ARM_NEON_BF16_DOT_F32", "ARM_SQRT_F32", "BF16", "BOOL", "CONTIGUOUS_VISION_GROUP_ID",
  "EVALUATE", "F32", "F64", "I32", "ORDERED_F32_DOT", "ORDERED_F32_REDUCE_MAX",
  "ORDERED_F32_REDUCE_SUM", "PYTORCH_F32_VECTOR_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_SUM",
  "PYTORCH_POW_NEGATIVE_HALF_F32", "REDUCE", "SLEEF_COS_F32", "SLEEF_EXP_F32", "SLEEF_LOG1P_F32",
  "SLEEF_SIN_F32", "SLEEF_TANH_F32", "STABLE_TRUE_COORDINATE_AT_RANK", "STABLE_TRUE_COUNT",
  "STABLE_TRUE_PREFIX_RANK", "STRUCT", "concat", "decode", "exact_product",
  "exact_safe_integer", "floor", "max", "min", "row_major_alias", "tuple",
]);

/** Rejects formula helpers whose executable meaning is absent from the embedded language contract. */
export function validateGemma4LiteralFormulaFunctionCoverage(formulas: readonly string[]): void {
  for (const formula of formulas) {
    for (const match of formula.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)) {
      if (!REGISTERED_FUNCTIONS.has(match[1]!)) {
        throw new Error(`Fórmula Gemma 4 contém helper opaco sem programa incorporado: ${match[1]}.`);
      }
    }
  }
}
