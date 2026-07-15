import type {
  DenseF32Tensor,
  DenseTensor,
  DifferentialComparisonReport,
  DifferentialGenerationComparisonReport,
  DifferentialGenerationReferenceTrace,
  DifferentialGenerationStepComparison,
  DifferentialKeyValueCacheSample,
  DifferentialKeyValueCacheComparison,
  DifferentialOperationComparison,
  DifferentialReferenceTrace,
  DifferentialTensorMetrics,
  DifferentialTolerance,
  ModelIR,
  ReferenceExecutionResult,
  ReferenceF32ExecutionResult,
  ReferenceF32GenerationResult,
  ReferenceGenerationResult,
} from "./types.js";
import { selectGreedyToken } from "./generation.js";

type ComparableTensor = DenseTensor | DenseF32Tensor;
type ExecutionResult = ReferenceExecutionResult | ReferenceF32ExecutionResult;
type GenerationResult = ReferenceGenerationResult | ReferenceF32GenerationResult;
type GenerationCacheMap = ReadonlyMap<number, { key: ComparableTensor; value: ComparableTensor }>;

const DEFAULT_TOLERANCE: DifferentialTolerance = { maxAbsoluteError: 0, maxRelativeError: 0 };

/**
 * Compare every emitted IR operation against a trace captured by an
 * authoritative runtime. Missing captures and mismatched shapes are evidence
 * of an incomplete comparison, never treated as a pass.
 */
export function compareExecutionTrace(
  ir: ModelIR,
  candidate: ExecutionResult,
  reference: DifferentialReferenceTrace,
  options: { candidateRuntime: string; tolerance?: DifferentialTolerance; topK?: number },
): DifferentialComparisonReport {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  validateTolerance(tolerance);
  const topK = options.topK ?? 10;
  if (!Number.isInteger(topK) || topK <= 0) throw new Error("topK deve ser inteiro positivo.");

  const expected = [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
  const referenceById = new Map<string, (typeof reference.operations)[number]>();
  for (const sample of reference.operations) {
    if (referenceById.has(sample.operationId)) throw new Error(`Trace de referência contém operação duplicada: ${sample.operationId}.`);
    referenceById.set(sample.operationId, sample);
  }

  const operations: DifferentialOperationComparison[] = [];
  const missingReferenceOperationIds: string[] = [];
  let firstDivergentOperation: string | null = null;
  for (const operation of expected) {
    const sample = referenceById.get(operation.id);
    if (!sample) {
      missingReferenceOperationIds.push(operation.id);
      operations.push({ operationId: operation.id, output: operation.output, status: "missing-reference" });
      firstDivergentOperation ??= operation.id;
      continue;
    }
    if (sample.output !== operation.output) {
      throw new Error(`Trace de referência ${operation.id} declara output '${sample.output}', esperado '${operation.output}'.`);
    }
    const actual = candidate.values.get(operation.output);
    if (!actual) throw new Error(`Executor candidato não produziu '${operation.output}' para ${operation.id}.`);
    if (!sameShape(actual.shape, sample.tensor.shape)) {
      operations.push({ operationId: operation.id, output: operation.output, status: "shape-mismatch" });
      firstDivergentOperation ??= operation.id;
      continue;
    }
    const metrics = compareTensor(actual, sample.tensor, topK);
    const status = passes(metrics, tolerance) ? "pass" : "diverged";
    if (status === "diverged") firstDivergentOperation ??= operation.id;
    operations.push({ operationId: operation.id, output: operation.output, status, metrics });
  }

  const expectedIds = new Set(expected.map((operation) => operation.id));
  const unexpectedReferenceOperationIds = [...referenceById.keys()].filter((id) => !expectedIds.has(id)).sort();
  if (unexpectedReferenceOperationIds.length > 0) firstDivergentOperation ??= unexpectedReferenceOperationIds[0]!;

  const referenceCache = new Map<number, (typeof reference.pastKeyValues)[number]>();
  for (const cache of reference.pastKeyValues) {
    if (referenceCache.has(cache.layer)) throw new Error(`Trace de referência contém cache KV duplicado para camada ${cache.layer}.`);
    referenceCache.set(cache.layer, cache);
  }
  const kvCache: DifferentialKeyValueCacheComparison[] = [];
  for (const layer of new Set([...referenceCache.keys(), ...candidate.pastKeyValues.keys()])) {
    const expectedCache = referenceCache.get(layer);
    const actualCache = candidate.pastKeyValues.get(layer);
    if (!expectedCache) {
      kvCache.push({ layer, status: "missing-reference" });
      firstDivergentOperation ??= `kv-cache:layer-${layer}`;
      continue;
    }
    if (!actualCache) {
      kvCache.push({ layer, status: "missing-candidate" });
      firstDivergentOperation ??= `kv-cache:layer-${layer}`;
      continue;
    }
    if (!sameShape(actualCache.key.shape, expectedCache.key.shape) || !sameShape(actualCache.value.shape, expectedCache.value.shape)) {
      kvCache.push({ layer, status: "shape-mismatch" });
      firstDivergentOperation ??= `kv-cache:layer-${layer}`;
      continue;
    }
    const key = compareTensor(actualCache.key, expectedCache.key, topK);
    const value = compareTensor(actualCache.value, expectedCache.value, topK);
    const status = passes(key, tolerance) && passes(value, tolerance) ? "pass" : "diverged";
    if (status === "diverged") firstDivergentOperation ??= `kv-cache:layer-${layer}`;
    kvCache.push({ layer, status, key, value });
  }

  // `candidate.logits` is the terminal decoder result, which is softcapped
  // when the IR emits `final_logit_softcap`.  Looking up lm_head first would
  // compare post-softcap candidate logits with pre-softcap reference logits;
  // worse, the operation comparisons could still all pass and incorrectly
  // classify the complete trace as equivalent.  Derive the comparison target
  // from the emitted IR rather than assuming a particular epilogue shape.
  const terminalOperations = [...expected].reverse();
  const logitsOperation = terminalOperations.find((operation) => operation.output === "softcapped_logits") ??
    terminalOperations.find((operation) => operation.output === "logits");
  const logitsSample = logitsOperation ? referenceById.get(logitsOperation.id) : undefined;
  const logits = logitsSample && sameShape(candidate.logits.shape, logitsSample.tensor.shape)
    ? compareTensor(candidate.logits, logitsSample.tensor, topK)
    : null;
  const complete = missingReferenceOperationIds.length === 0 && unexpectedReferenceOperationIds.length === 0 &&
    operations.every((comparison) => comparison.status === "pass") && kvCache.every((comparison) => comparison.status === "pass") &&
    logits !== null && passes(logits, tolerance);
  const exact = operations.every((comparison) => exactMetrics(comparison.metrics)) &&
    kvCache.every((comparison) => exactMetrics(comparison.key) && exactMetrics(comparison.value)) && exactMetrics(logits ?? undefined);
  const fidelityClass = !complete ? "incomplete" : exact ? "lossless-within-dtype" : "numerically-equivalent";
  return {
    reference: {
      runtime: reference.runtime,
      model: reference.model,
      revisionOrChecksum: reference.revisionOrChecksum,
      containerFormat: reference.containerFormat,
      quantization: reference.quantization,
      inputTokens: reference.inputTokens.map((row) => [...row]),
      dtypePolicy: reference.dtypePolicy,
    },
    candidateRuntime: options.candidateRuntime,
    tolerance: { ...tolerance },
    operations,
    kvCache,
    missingReferenceOperationIds,
    unexpectedReferenceOperationIds,
    firstDivergentOperation,
    logits,
    fidelityClass,
  };
}

/**
 * Compare a complete greedy decode against an authoritative capture. This is
 * intentionally separate from forward-operation comparison: a matching final
 * argmax cannot prove that each emitted token used the same absolute position
 * or left an equivalent KV cache for continuation.
 */
export function compareGenerationTrace(
  candidate: GenerationResult,
  reference: DifferentialGenerationReferenceTrace,
  options: { candidateRuntime: string; tolerance?: DifferentialTolerance; topK?: number },
): DifferentialGenerationComparisonReport {
  const tolerance = options.tolerance ?? DEFAULT_TOLERANCE;
  validateTolerance(tolerance);
  const topK = options.topK ?? 10;
  if (!Number.isInteger(topK) || topK <= 0) throw new Error("topK deve ser inteiro positivo.");
  validateGenerationReference(reference);
  validateCandidateGeneration(candidate);

  const generatedTokenIds: DifferentialGenerationStepComparison[] = [];
  let firstDivergence: string | null = null;
  const promptMatches = candidate.inputIds.length >= reference.inputTokens.length &&
    reference.inputTokens.every((token, index) => candidate.inputIds[index] === token);
  if (!promptMatches) firstDivergence = "generation:prompt";
  const count = Math.max(candidate.steps.length, reference.steps.length);
  for (let index = 0; index < count; index += 1) {
    const actual = candidate.steps[index];
    const expected = reference.steps[index];
    const status = !expected ? "missing-reference" : !actual ? "missing-candidate" :
      actual.tokenId === expected.tokenId && actual.positionId === expected.positionId ? "pass" : "diverged";
    if (status !== "pass") firstDivergence ??= `generation:step-${index}`;
    const candidateSelectionLogits = candidate.selectionLogits[index];
    const referenceSelectionLogits = reference.selectionLogits[index];
    const selectionLogits = candidateSelectionLogits && referenceSelectionLogits && sameShape(candidateSelectionLogits.shape, referenceSelectionLogits.shape)
      ? compareTensor(candidateSelectionLogits, referenceSelectionLogits, topK)
      : null;
    if (!selectionLogits) firstDivergence ??= `generation:selection-logits-${index}-shape`;
    else if (!passes(selectionLogits, tolerance)) firstDivergence ??= `generation:selection-logits-${index}`;
    const stepKvCache = compareGenerationCache(candidate.stepPastKeyValues[index], reference.stepPastKeyValues[index], tolerance, topK);
    const stepCacheDivergence = stepKvCache.find((cache) => cache.status !== "pass");
    if (stepCacheDivergence) firstDivergence ??= `generation:step-${index}-kv-cache-layer-${stepCacheDivergence.layer}`;
    generatedTokenIds.push({ index, status, ...(actual ? { candidate: { ...actual } } : {}), ...(expected ? { reference: { ...expected } } : {}), selectionLogits, kvCache: stepKvCache });
  }

  const terminalLogits = sameShape(candidate.logits.shape, reference.logits.shape)
    ? compareTensor(candidate.logits, reference.logits, topK)
    : null;
  if (!terminalLogits) firstDivergence ??= "generation:terminal-logits-shape";
  else if (!passes(terminalLogits, tolerance)) firstDivergence ??= "generation:terminal-logits";

  const referenceCache = uniqueGenerationCache(reference.pastKeyValues);
  const kvCache: DifferentialKeyValueCacheComparison[] = [];
  for (const layer of new Set([...referenceCache.keys(), ...candidate.pastKeyValues.keys()])) {
    const expected = referenceCache.get(layer);
    const actual = candidate.pastKeyValues.get(layer);
    if (!expected) {
      kvCache.push({ layer, status: "missing-reference" });
      firstDivergence ??= `generation:kv-cache-layer-${layer}`;
      continue;
    }
    if (!actual) {
      kvCache.push({ layer, status: "missing-candidate" });
      firstDivergence ??= `generation:kv-cache-layer-${layer}`;
      continue;
    }
    if (!sameShape(actual.key.shape, expected.key.shape) || !sameShape(actual.value.shape, expected.value.shape)) {
      kvCache.push({ layer, status: "shape-mismatch" });
      firstDivergence ??= `generation:kv-cache-layer-${layer}`;
      continue;
    }
    const key = compareTensor(actual.key, expected.key, topK);
    const value = compareTensor(actual.value, expected.value, topK);
    const status = passes(key, tolerance) && passes(value, tolerance) ? "pass" : "diverged";
    if (status !== "pass") firstDivergence ??= `generation:kv-cache-layer-${layer}`;
    kvCache.push({ layer, status, key, value });
  }

  const incomplete = generatedTokenIds.some((step) => step.status.startsWith("missing") || step.selectionLogits === null ||
    step.kvCache?.some((cache) => cache.status.startsWith("missing") || cache.status === "shape-mismatch")) ||
    kvCache.some((cache) => cache.status.startsWith("missing") || cache.status === "shape-mismatch") || terminalLogits === null;
  const numericallyEquivalent = !incomplete && firstDivergence === null;
  const exact = numericallyEquivalent && generatedTokenIds.every((step) => exactMetrics(step.selectionLogits ?? undefined) &&
    step.kvCache?.every((cache) => exactMetrics(cache.key) && exactMetrics(cache.value))) && exactMetrics(terminalLogits ?? undefined) &&
    kvCache.every((cache) => exactMetrics(cache.key) && exactMetrics(cache.value));
  return {
    reference: {
      runtime: reference.runtime,
      model: reference.model,
      revisionOrChecksum: reference.revisionOrChecksum,
      containerFormat: reference.containerFormat,
      quantization: reference.quantization,
      inputTokens: [...reference.inputTokens],
      promptPositionIds: [...reference.promptPositionIds],
      dtypePolicy: reference.dtypePolicy,
      maxNewTokens: reference.maxNewTokens,
      ...(reference.eosTokenId !== undefined ? { eosTokenId: reference.eosTokenId } : {}),
    },
    candidateRuntime: options.candidateRuntime,
    tolerance: { ...tolerance },
    promptMatches,
    generatedTokenIds,
    terminalLogits,
    kvCache,
    firstDivergence,
    fidelityClass: incomplete ? "incomplete" : exact ? "lossless-within-dtype" : numericallyEquivalent ? "numerically-equivalent" : "approximate",
  };
}

function validateCandidateGeneration(candidate: GenerationResult): void {
  if (candidate.steps.length !== candidate.generatedTokenIds.length) {
    throw new Error("Resultado candidato de geração requer um step para cada token emitido.");
  }
  if (candidate.selectionLogits.length !== candidate.generatedTokenIds.length) {
    throw new Error("Resultado candidato de geração requer logits de seleção para cada token emitido.");
  }
  if (candidate.stepPastKeyValues.length !== candidate.generatedTokenIds.length) {
    throw new Error("Resultado candidato de geração requer cache KV pós-decode para cada token emitido.");
  }
  if (candidate.inputIds.length < candidate.generatedTokenIds.length ||
    !candidate.generatedTokenIds.every((token, index) => candidate.inputIds[candidate.inputIds.length - candidate.generatedTokenIds.length + index] === token)) {
    throw new Error("Resultado candidato de geração não contém os tokens emitidos no sufixo de inputIds.");
  }
  for (const [index, step] of candidate.steps.entries()) {
    if (!Number.isInteger(step.tokenId) || step.tokenId < 0 || !Number.isInteger(step.positionId) || step.positionId < 0 || step.tokenId !== candidate.generatedTokenIds[index]) {
      throw new Error(`Resultado candidato de geração contém step inválido no índice ${index}.`);
    }
    if (selectGreedyToken(candidate.selectionLogits[index]!) !== step.tokenId) {
      throw new Error(`Resultado candidato de geração seleciona token ${step.tokenId} no índice ${index}, mas os logits determinísticos exigem outro argmax.`);
    }
  }
}

function validateGenerationReference(reference: DifferentialGenerationReferenceTrace): void {
  for (const field of ["runtime", "model", "revisionOrChecksum", "dtypePolicy"] as const) {
    if (reference[field].trim() === "") throw new Error(`Trace de geração requer ${field} não vazio.`);
  }
  if (reference.inputTokens.length === 0 || reference.inputTokens.some((token) => !Number.isInteger(token) || token < 0)) {
    throw new Error("Trace de geração requer inputTokens não vazio com IDs inteiros não negativos.");
  }
  if (reference.promptPositionIds.length !== reference.inputTokens.length || reference.promptPositionIds.some((position) => !Number.isInteger(position) || position < 0)) {
    throw new Error("Trace de geração requer promptPositionIds inteiros não negativos com o comprimento do prompt.");
  }
  if (!Number.isInteger(reference.maxNewTokens) || reference.maxNewTokens < 0 || reference.generatedTokenIds.length > reference.maxNewTokens) {
    throw new Error("Trace de geração contém maxNewTokens inválido ou mais tokens do que o limite declarado.");
  }
  if (reference.eosTokenId !== undefined && (!Number.isInteger(reference.eosTokenId) || reference.eosTokenId < 0)) {
    throw new Error("Trace de geração contém eosTokenId inválido.");
  }
  if (reference.steps.length !== reference.generatedTokenIds.length) {
    throw new Error("Trace de geração requer um step para cada token emitido.");
  }
  if (reference.selectionLogits.length !== reference.generatedTokenIds.length) {
    throw new Error("Trace de geração requer logits de seleção para cada token emitido.");
  }
  if (reference.stepPastKeyValues.length !== reference.generatedTokenIds.length) {
    throw new Error("Trace de geração requer cache KV pós-decode para cada token emitido.");
  }
  for (const [index, step] of reference.steps.entries()) {
    if (!Number.isInteger(step.tokenId) || step.tokenId < 0 || !Number.isInteger(step.positionId) || step.positionId < 0 || step.tokenId !== reference.generatedTokenIds[index]) {
      throw new Error(`Trace de geração contém step inválido no índice ${index}.`);
    }
    if (selectGreedyToken(reference.selectionLogits[index]!) !== step.tokenId) {
      throw new Error(`Trace de geração seleciona token ${step.tokenId} no índice ${index}, mas os logits determinísticos exigem outro argmax.`);
    }
  }
  const eosIndex = reference.eosTokenId === undefined ? -1 : reference.generatedTokenIds.indexOf(reference.eosTokenId);
  if (eosIndex >= 0 && eosIndex !== reference.generatedTokenIds.length - 1) {
    throw new Error("Trace de geração não pode emitir tokens após EOS.");
  }
}

function uniqueGenerationCache(samples: readonly DifferentialGenerationReferenceTrace["pastKeyValues"][number][]): Map<number, DifferentialGenerationReferenceTrace["pastKeyValues"][number]> {
  const result = new Map<number, DifferentialGenerationReferenceTrace["pastKeyValues"][number]>();
  for (const sample of samples) {
    if (!Number.isInteger(sample.layer) || sample.layer < 0) throw new Error("Trace de geração contém camada KV inválida.");
    if (result.has(sample.layer)) throw new Error(`Trace de geração contém cache KV duplicado para camada ${sample.layer}.`);
    result.set(sample.layer, sample);
  }
  return result;
}

function compareGenerationCache(
  candidate: GenerationCacheMap | undefined,
  reference: readonly DifferentialKeyValueCacheSample[] | undefined,
  tolerance: DifferentialTolerance,
  topK: number,
): DifferentialKeyValueCacheComparison[] {
  const referenceCache = uniqueGenerationCache(reference ?? []);
  const comparisons: DifferentialKeyValueCacheComparison[] = [];
  for (const layer of new Set([...referenceCache.keys(), ...(candidate?.keys() ?? [])])) {
    const expected = referenceCache.get(layer);
    const actual = candidate?.get(layer);
    if (!expected) {
      comparisons.push({ layer, status: "missing-reference" });
      continue;
    }
    if (!actual) {
      comparisons.push({ layer, status: "missing-candidate" });
      continue;
    }
    if (!sameShape(actual.key.shape, expected.key.shape) || !sameShape(actual.value.shape, expected.value.shape)) {
      comparisons.push({ layer, status: "shape-mismatch" });
      continue;
    }
    const key = compareTensor(actual.key, expected.key, topK);
    const value = compareTensor(actual.value, expected.value, topK);
    comparisons.push({ layer, status: passes(key, tolerance) && passes(value, tolerance) ? "pass" : "diverged", key, value });
  }
  return comparisons;
}

function compareTensor(actual: ComparableTensor, expected: ComparableTensor, topK: number): DifferentialTensorMetrics {
  let maxAbsoluteError = 0;
  let maxRelativeError = 0;
  let nonFiniteMismatchCount = 0;
  let dot = 0;
  let actualNorm = 0;
  let expectedNorm = 0;
  for (let index = 0; index < actual.values.length; index += 1) {
    const a = actual.values[index]!;
    const b = expected.values[index]!;
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      if (!Object.is(a, b)) nonFiniteMismatchCount += 1;
      continue;
    }
    const absolute = Math.abs(a - b);
    maxAbsoluteError = Math.max(maxAbsoluteError, absolute);
    maxRelativeError = Math.max(maxRelativeError, absolute / Math.max(Math.abs(b), Number.MIN_VALUE));
    dot += a * b;
    actualNorm += a * a;
    expectedNorm += b * b;
  }
  return {
    shape: [...actual.shape],
    elementCount: actual.values.length,
    maxAbsoluteError,
    maxRelativeError,
    cosineSimilarity: actualNorm === 0 || expectedNorm === 0 ? null : dot / Math.sqrt(actualNorm * expectedNorm),
    topKOverlap: actual.shape.length === 0 ? null : topKOverlap(actual.values, expected.values, topK),
    argmaxAgreement: actual.shape.length === 0 ? null : argmax(actual.values) === argmax(expected.values),
    nonFiniteMismatchCount,
  };
}

function passes(metrics: DifferentialTensorMetrics, tolerance: DifferentialTolerance): boolean {
  return metrics.nonFiniteMismatchCount === 0 &&
    (metrics.maxAbsoluteError <= tolerance.maxAbsoluteError || metrics.maxRelativeError <= tolerance.maxRelativeError);
}

function exactMetrics(metrics: DifferentialTensorMetrics | undefined): boolean {
  return metrics?.maxAbsoluteError === 0 && metrics.nonFiniteMismatchCount === 0;
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}

function argmax(values: Float64Array | Float32Array): number {
  let result = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index]! > values[result]!) result = index;
  return result;
}

function topKOverlap(left: Float64Array | Float32Array, right: Float64Array | Float32Array, requested: number): number {
  const count = Math.min(requested, left.length);
  const indices = (values: Float64Array | Float32Array) => [...values.keys()].sort((a, b) => values[b]! - values[a]!).slice(0, count);
  const expected = new Set(indices(right));
  return indices(left).filter((index) => expected.has(index)).length / count;
}

function validateTolerance(tolerance: DifferentialTolerance): void {
  if (!Number.isFinite(tolerance.maxAbsoluteError) || tolerance.maxAbsoluteError < 0 || !Number.isFinite(tolerance.maxRelativeError) || tolerance.maxRelativeError < 0) {
    throw new Error("Tolerâncias diferencial devem ser números finitos não negativos.");
  }
}
