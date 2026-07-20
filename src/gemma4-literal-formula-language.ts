import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4LiteralGenerationForwardCalculationContract,
  Gemma4LiteralGenerationScalarCalculations,
} from "./gemma4-literal-generation-calculations.js";
import type { Gemma4LiteralScalarCalculations } from "./gemma4-literal-scalar-calculations.js";
import { gemma4LiteralNormalizationReductionPrograms } from "./gemma4-literal-normalization-reduction-view.js";
import {
  gemma4LiteralReductionDomainLanguage,
  type Gemma4LiteralReductionDomainLanguage,
} from "./gemma4-literal-reduction-domains.js";

export interface Gemma4LiteralFormulaLanguageContract {
  kind: "gemma4-literal-formula-language-contract";
  schemaVersion: 24;
  languageId: "indexed-ieee754-expression-v1";
  authority: {
    forwardAssignments: "/scalarCalculations/assignments";
    forwardScalarExecution: "/calculationGraph/assignments/*/scalarCalculation/scalarAssignments";
    forwardScalarDataflow: "/calculationGraph/assignments/*/scalarCalculation/statementDataflow";
    forwardScalarPrograms: "/calculationGraph/assignments/*/scalarCalculation/statementPrograms";
    forwardControlProgram: "/forwardControl";
    inputContract: "/inputContract";
    outputContract: "/outputContract";
    generationAssignments: "/generation/scalarCalculations/assignments";
    generationControlProgram: "/generation/controlProgram";
    instantiatedForwardOrder: "/calculationGraph/assignments";
    outputCoordinateWrite: "/calculationGraph/assignments/*/outputCoordinate/write";
    coordinateExpressionLanguage: "/calculationGraph/coordinateLanguage";
    predecessorCoordinateAccesses: "/calculationGraph/assignments/*/predecessors/*/accesses";
    consumerCoordinateAccesses: "/calculationGraph/assignments/*/consumerCoordinates/*/accesses";
    generationForwardOrder: "/generation/forwardCalculation/operationOrder";
    cacheTransitions: "/generation/forwardCalculation/cacheTransitions";
    learnedOperandBindings: "/learnedOperands/assignments";
    learnedIndexLanguage: "/learnedOperands/indexLanguage";
    storageDecoders: "/storageDecoders";
    denseDecoderLanguage: "/denseDecoderLanguage";
    transcendentalPrograms: "/transcendentalPrograms";
    numericLiteralBits: "/numericLiterals/literals";
    calculationDomains: "/calculationDomains/assignments";
    dimensionLanguage: "/calculationDomains/dimensionLanguage";
    dimensionPrograms: "/calculationDomains/dimensionPrograms";
    reductionDomains: "/scalarCalculations/assignments/*/(reduction|reductionStages/*)/domains";
  };
  evaluation: {
    dependencyOrder: string;
    scalarIntermediateNavigation: string;
    predecessorNavigation: string;
    outputAndConsumerNavigation: string;
    coordinateOrder: string;
    inputBinding: string;
    operandClosure: string;
    dimensionBinding: string;
    reductionBinding: string;
    numericTokenBinding: string;
    learnedValueBinding: string;
    forwardOrder: string;
    generationOrder: string;
    cacheTransitionOrder: string;
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
  scalarPrograms: {
    authority: string;
    nodeKinds: string;
    evaluationOrder: string;
    calls: string;
    namedArguments: string;
    filteredDomains: string;
    orderedLoops: string;
    evaluateInvocation: string;
    sourceRendering: string;
  };
  scalarTypes: Array<{ name: "F64" | "F32" | "BF16" | "I32" | "BOOL"; semantics: string }>;
  operators: Array<{ notation: string; semantics: string }>;
  reductions: {
    ascendingLexicographic: string;
    operationDeclared: string;
    runtimeDefined: string;
    multipleDomains: string;
    stagedReductions: string;
    domainLanguage: Gemma4LiteralReductionDomainLanguage;
    normalizationPrograms: ReturnType<typeof gemma4LiteralNormalizationReductionPrograms>;
    softmaxPrograms: Gemma4LiteralSoftmaxReductionPrograms;
  };
  intrinsics: Array<{ notation: string; semantics: string }>;
}

export interface Gemma4LiteralIndexingPrograms {
  kind: "gemma4-literal-indexing-programs";
  schemaVersion: 3;
  contiguousVisionGroupId: string[];
  stableTrueCount: string[];
  stableTruePrefixRank: string[];
  stableTrueCoordinateAtRank: string[];
  visionPoolSlot: string[];
  visionPoolCellHasPatch: string[];
  audioRelativeShiftSource: string[];
}

export function gemma4LiteralIndexingPrograms(): Gemma4LiteralIndexingPrograms {
  return {
    kind: "gemma4-literal-indexing-programs",
    schemaVersion: 3,
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
    visionPoolSlot: [
      "require pixel_position_ids to be a non-empty row of exact I32 [x,y] pairs; padding is only [-1,-1]; require patch, kernel and pool_cells to be positive in-bounds I32 values except patch may be zero",
      "if pixel_position_ids[patch]==[-1,-1] result=I32(-1)",
      "max_x=I32(1+max(pixel_position_ids[index].x for index=0..patches-1 ascending where pixel_position_ids[index]!=[-1,-1])); require at least one non-padding patch",
      "columns=I32(floor(max_x/kernel)); slot=I32(floor(pixel_position_ids[patch].x/kernel)+columns*floor(pixel_position_ids[patch].y/kernel))",
      "require slot in 0..pool_cells-1; result=slot",
    ],
    visionPoolCellHasPatch: [
      "require pool_cell to be I32 in 0..pool_cells-1 and validate the complete row through VISION_POOL_SLOT",
      "result=false",
      "for patch=0..patches-1 ascending: if VISION_POOL_SLOT(pixel_position_ids,patch,kernel,pool_cells)==pool_cell result=true",
      "return BOOL(result) after visiting the complete patch domain",
    ],
    audioRelativeShiftSource: [
      "require query_in_block and key_slot to be non-negative I32; require context and relative_length to be positive I32 with key_slot<context and relative_length<=context+1",
      "padded_length=I32(context+1); flattened=I32(query_in_block*context+key_slot)",
      "source_query=I32(floor(flattened/padded_length)); source_relative=I32(flattened%padded_length)",
      "result=source_relative<relative_length ? STRUCT(valid=true,query_in_block=source_query,relative_index=source_relative) : STRUCT(valid=false,query_in_block=source_query,relative_index=I32(-1))",
    ],
  };
}

export interface Gemma4LiteralAudioRelativeShiftSource {
  valid: boolean;
  queryInBlock: number;
  relativeIndex: number;
}

/** Maps one patch to the exact source-visible Gemma 4 pooling cell, or -1 for padding. */
export function executeGemma4LiteralVisionPoolSlot(
  programs: Gemma4LiteralIndexingPrograms,
  pixelPositionIds: readonly (readonly [number, number])[],
  patch: number,
  kernel: number,
  poolCells: number,
): number {
  assertCanonicalIndexingPrograms(programs);
  validateVisionPoolRequest(pixelPositionIds, patch, kernel, poolCells);
  const [x, y] = pixelPositionIds[patch]!;
  const validXs = pixelPositionIds.filter(([candidateX, candidateY]) => candidateX !== -1 || candidateY !== -1).map(([candidateX]) => candidateX);
  if (validXs.length === 0) throw new Error("Programa de pool vision Gemma 4 requer ao menos um patch válido.");
  if (x === -1 && y === -1) return -1;
  const maxX = Math.max(...validXs) + 1;
  const slot = Math.floor(x / kernel) + Math.floor(maxX / kernel) * Math.floor(y / kernel);
  if (!Number.isSafeInteger(slot) || slot < 0 || slot >= poolCells) {
    throw new Error(`Programa de pool vision Gemma 4 produziu slot ${slot} fora de ${poolCells}.`);
  }
  return slot;
}

/** Executes the finite existential pool-mask program without an implicit host reduction. */
export function executeGemma4LiteralVisionPoolCellHasPatch(
  programs: Gemma4LiteralIndexingPrograms,
  pixelPositionIds: readonly (readonly [number, number])[],
  poolCell: number,
  kernel: number,
  poolCells: number,
): boolean {
  assertCanonicalIndexingPrograms(programs);
  if (!Number.isSafeInteger(poolCell) || poolCell < 0 || poolCell >= poolCells) {
    throw new Error("Programa de máscara pool vision Gemma 4 requer célula válida.");
  }
  let result = false;
  for (let patch = 0; patch < pixelPositionIds.length; patch += 1) {
    if (executeGemma4LiteralVisionPoolSlot(programs, pixelPositionIds, patch, kernel, poolCells) === poolCell) result = true;
  }
  return result;
}

/** Mirrors the exact pad/view/slice/view source coordinate used by Gemma 4 audio. */
export function executeGemma4LiteralAudioRelativeShiftSource(
  programs: Gemma4LiteralIndexingPrograms,
  queryInBlock: number,
  keySlot: number,
  context: number,
  relativeLength: number,
): Gemma4LiteralAudioRelativeShiftSource {
  assertCanonicalIndexingPrograms(programs);
  if (![queryInBlock, keySlot, context, relativeLength].every(Number.isSafeInteger) || queryInBlock < 0 || keySlot < 0 ||
    context <= 0 || keySlot >= context || relativeLength <= 0 || relativeLength > context + 1) {
    throw new Error("Programa de relative shift audio Gemma 4 requer coordenadas e domínios válidos.");
  }
  const paddedLength = context + 1;
  const flattened = queryInBlock * context + keySlot;
  if (!Number.isSafeInteger(flattened)) throw new Error("Programa de relative shift audio Gemma 4 excedeu I32 seguro.");
  const sourceQuery = Math.floor(flattened / paddedLength);
  const sourceRelative = flattened % paddedLength;
  return {
    valid: sourceRelative < relativeLength,
    queryInBlock: sourceQuery,
    relativeIndex: sourceRelative < relativeLength ? sourceRelative : -1,
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

function validateVisionPoolRequest(
  positions: readonly (readonly [number, number])[],
  patch: number,
  kernel: number,
  poolCells: number,
): void {
  const validPositions = positions.length > 0 && positions.every((position) => position.length === 2 &&
    position.every(Number.isSafeInteger) && ((position[0] === -1 && position[1] === -1) || (position[0] >= 0 && position[1] >= 0)));
  if (!validPositions || !Number.isSafeInteger(patch) || patch < 0 || patch >= positions.length ||
    !Number.isSafeInteger(kernel) || kernel <= 0 || !Number.isSafeInteger(poolCells) || poolCells <= 0) {
    throw new Error("Programa de pool vision Gemma 4 requer posições e domínios I32 válidos.");
  }
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
    schemaVersion: 24,
    languageId: "indexed-ieee754-expression-v1",
    authority: {
      forwardAssignments: "/scalarCalculations/assignments",
      forwardScalarExecution: "/calculationGraph/assignments/*/scalarCalculation/scalarAssignments",
      forwardScalarDataflow: "/calculationGraph/assignments/*/scalarCalculation/statementDataflow",
      forwardScalarPrograms: "/calculationGraph/assignments/*/scalarCalculation/statementPrograms",
      forwardControlProgram: "/forwardControl",
      inputContract: "/inputContract",
      outputContract: "/outputContract",
      generationAssignments: "/generation/scalarCalculations/assignments",
      generationControlProgram: "/generation/controlProgram",
      instantiatedForwardOrder: "/calculationGraph/assignments",
      outputCoordinateWrite: "/calculationGraph/assignments/*/outputCoordinate/write",
      coordinateExpressionLanguage: "/calculationGraph/coordinateLanguage",
      predecessorCoordinateAccesses: "/calculationGraph/assignments/*/predecessors/*/accesses",
      consumerCoordinateAccesses: "/calculationGraph/assignments/*/consumerCoordinates/*/accesses",
      generationForwardOrder: "/generation/forwardCalculation/operationOrder",
      cacheTransitions: "/generation/forwardCalculation/cacheTransitions",
      learnedOperandBindings: "/learnedOperands/assignments",
      learnedIndexLanguage: "/learnedOperands/indexLanguage",
      storageDecoders: "/storageDecoders",
      denseDecoderLanguage: "/denseDecoderLanguage",
      transcendentalPrograms: "/transcendentalPrograms",
      numericLiteralBits: "/numericLiterals/literals",
      calculationDomains: "/calculationDomains/assignments",
      dimensionLanguage: "/calculationDomains/dimensionLanguage",
      dimensionPrograms: "/calculationDomains/dimensionPrograms",
      reductionDomains: "/scalarCalculations/assignments/*/(reduction|reductionStages/*)/domains",
    },
    evaluation: {
      dependencyOrder: "evaluate instantiated assignments by ascending ordinal; within each assignment evaluate statementPrograms in ordinal order, where every local and precondition precedes the final output assignment; scalarAssignments is the audit rendering and every predecessor must already exist",
      scalarIntermediateNavigation: "statementDataflow has one entry per scalarAssignments ordinal; writes identifies each local or terminal output coordinate, reads binds every distinct local access to its earlier producerStatementOrdinal, coordinatePrograms are executable under calculationGraph.coordinateLanguage, and consumerStatementOrdinals are the exact reverse edges",
      predecessorNavigation: "for each predecessor, accesses lists every distinct tensor-element or tensor-shape expression in first-use order; coordinatePrograms and axisProgram are closed under the owning output axes, reduction domains, prior locals and explicit tensor-axis reads in calculationGraph.coordinateLanguage; free host extent aliases are invalid; whole-value marks an unindexed structured/control read, while an empty list with scalarUse=shape-or-control-only declares that the dependency affects domain or branch selection rather than the scalar expression",
      outputAndConsumerNavigation: "outputCoordinate.write is the unique left-hand tensor element assigned by the scalar program and carries executable coordinatePrograms; shapeAssertions carry executable axisProgram values, and consumerCoordinates repeats every downstream read and program grouped by consumer operation so traversal is exact in both dependency directions without parsing the human expression string",
      coordinateOrder: "row-major lexicographic over the complete declared output domain; preview windows never change evaluation",
      inputBinding: "bind orderedInputs positionally at each instantiated call site before evaluating the indexed formula",
      operandClosure: "every source tensor read names one ordered input and an explicit coordinate expression; free aliases such as input, x, q, k, value, padded_input, ellipsis and prose branch descriptions are invalid",
      dimensionBinding: "resolve every symbolic output bound through calculationDomains.dimensionPrograms and the embedded safe-integer dimension language; missing bindings are invalid and must never be guessed",
      reductionBinding: "resolve every reduction index interval through its domains entry and reductions.domainLanguage after calculation-graph call-site tensor binding; prose aliases such as width, patches, context, head_dim and in_features never supply an extent",
      numericTokenBinding: "resolve forward, generation and cache-transition decimal or named mathematical tokens through numericLiterals; resolve transcendental-program names through transcendentalPrograms.constants; select bits by the surrounding F64/F32/BF16 cast",
      learnedValueBinding: "evaluate learnedOperands.logicalIndices with its embedded integer-expression AST, then execute the matching storageDecoder address and decode expression ASTs under denseDecoderLanguage over the embedded constant bytes",
      forwardOrder: "execute forwardControl to select optional modality branches, exact absent-branch identity aliases and attention-mask mode; then evaluate only its selected calculationGraph assignments in serialized dependency order",
      generationOrder: "execute generation.controlProgram; scalarCalculations are its indexed audit rendering and may not replace or override structured control fields",
      cacheTransitionOrder: "for every forward invocation execute cacheTransitions in ascending layer order; within each transition execute prefill or incremental scalarAssignments in array order over its complete BHSD coordinate domain",
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
    scalarPrograms: {
      authority: "statementPrograms[*].expression is the executable syntax authority for its same-ordinal scalarAssignments audit rendering",
      nodeKinds: "literal, identifier, array, unary, binary, conditional, call, index, member, range-inclusive, named-argument, filtered-domain, ordered-loop and evaluate-invocation are a closed tagged union; an unknown kind is invalid",
      evaluationOrder: "evaluate child nodes left-to-right; binary && and || short-circuit, conditional evaluates only its selected branch, index coordinates are evaluated in axis order and explicit casts round immediately",
      calls: "call evaluates its callee then positional arguments left-to-right; the callee must resolve to an intrinsic, cast, reduction program, decoder, indexing program or transcendental program registered by this artifact",
      namedArguments: "named-argument binds a reduction index, range, lane count or schedule label by exact name inside the owning registered call; duplicate or unrecognized names are invalid",
      filteredDomains: "filtered-domain enumerates its range in declared order and evaluates the predicate at each bound index; false coordinates are skipped without evaluating the reduction body",
      orderedLoops: "ordered-loop enumerates its inclusive domain in ascending order and evaluates body once per index; loop-carried indexed locals observe only values written by earlier iterations",
      evaluateInvocation: "evaluate-invocation selects calculationGraph assignments whose invocationId equals predicateValue, executes them by ascending ordinal with orderedInputs bound positionally, and reads terminalOutput at terminalCoordinates",
      sourceRendering: "source is non-authoritative audit text and must correspond byte-for-byte to scalarAssignments[ordinal]; readers execute the tagged tree and never reparse source",
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
      { notation: "a+b, a-b, a*b, a/b, a%b, -a", semantics: "evaluate operands left-to-right; % is exact non-negative I32 remainder for the indexed programs; arithmetic precision changes only at an explicit cast or declared reduction/FMA boundary" },
      { notation: "a**b", semantics: "real exponentiation followed by the surrounding declared cast; package formulas additionally pin source-visible pow/rsqrt decompositions where fidelity requires them" },
      { notation: "predicate ? a : b", semantics: "evaluate the predicate then only the selected branch" },
      { notation: "require predicate", semantics: "evaluate predicate as BOOL and fail closed before any dependent assignment when it is false" },
      { notation: "==, >, >=, <, <=, &&, ||, !", semantics: "exact comparison or short-circuit Boolean operation over already materialized operands" },
      { notation: "tensor[i,j,...]", semantics: "zero-based indexed read using the referenced domain and layout; commas order axes exactly as declared" },
    ],
    reductions: {
      ascendingLexicographic: "initialize with the formula-declared identity and evaluate every declared index in ascending lexicographic order with each visible cast applied immediately",
      operationDeclared: "execute the serialized reduction.schedule literally, including product rounding, lane assignment, block/tile order, FMA behavior, fold order, tails, accumulation dtype and output cast",
      runtimeDefined: "not executable: reproducibility must be fail-closed-runtime-reduction and every scalar renderer or replay claiming literal fidelity must reject it",
      multipleDomains: "nested REDUCE domains execute left-to-right as written; a schedule attached to the assignment overrides only the reduction indices named beside it",
      stagedReductions: "execute reductionStages in array order; each stage binds exactly the indices, identity, predicate, program and optional operation-declared schedule named by the formula",
      domainLanguage: gemma4LiteralReductionDomainLanguage(),
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
      { notation: "VISION_POOL_SLOT", semantics: "execute indexing.programs.visionPoolSlot over the complete position row and return the exact pool cell, or I32(-1) for padding" },
      { notation: "VISION_POOL_CELL_HAS_PATCH", semantics: "execute indexing.programs.visionPoolCellHasPatch over the complete ascending patch domain and return whether the requested pool cell receives a valid patch" },
      { notation: "AUDIO_RELATIVE_SHIFT_SOURCE", semantics: "execute indexing.programs.audioRelativeShiftSource and return the exact valid flag plus source query/relative coordinates for the audio pad-view-slice-view transform" },
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
  generationForward: Gemma4LiteralGenerationForwardCalculationContract,
): void {
  if (!isDeepStrictEqual(contract, buildGemma4LiteralFormulaLanguageContract()) ||
    forward.formulaLanguage !== contract.languageId || generation.formulaLanguage !== contract.languageId) {
    throw new Error("Programa literal Gemma 4 possui linguagem de fórmulas ausente ou divergente.");
  }
  validateGemma4LiteralFormulaFunctionCoverage([
    ...forward.assignments.flatMap((assignment) => assignment.scalarAssignments),
    ...generation.assignments.flatMap((assignment) => assignment.scalarAssignments),
    ...generationForward.cacheTransitions.flatMap((transition) => [
      ...transition.prefill.scalarAssignments,
      ...transition.incremental.scalarAssignments,
    ]),
  ]);
}

const REGISTERED_FUNCTIONS = new Set([
  "ARM_NEON_BF16_DOT_F32", "ARM_SQRT_F32", "AUDIO_RELATIVE_SHIFT_SOURCE", "BF16", "BOOL", "CONTIGUOUS_VISION_GROUP_ID",
  "EVALUATE", "F32", "F32_FMA", "F64", "I32", "ORDERED_F32_DOT", "ORDERED_F32_REDUCE_MAX",
  "ORDERED_F32_REDUCE_SUM", "PYTORCH_F32_VECTOR_REDUCE_MAX", "PYTORCH_F32_VECTOR_REDUCE_SUM",
  "PYTORCH_POW_NEGATIVE_HALF_F32", "REDUCE", "SLEEF_COS_F32", "SLEEF_EXP_F32", "SLEEF_LOG1P_F32",
  "SLEEF_SIN_F32", "SLEEF_TANH_F32", "STABLE_TRUE_COORDINATE_AT_RANK", "STABLE_TRUE_COUNT",
  "STABLE_TRUE_PREFIX_RANK", "STRUCT", "VISION_POOL_CELL_HAS_PATCH", "VISION_POOL_SLOT", "concat", "decode", "exact_product",
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
