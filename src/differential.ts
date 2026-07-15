import type {
  DenseF32Tensor,
  DenseTensor,
  DifferentialComparisonReport,
  DifferentialKeyValueCacheComparison,
  DifferentialOperationComparison,
  DifferentialReferenceTrace,
  DifferentialTensorMetrics,
  DifferentialTolerance,
  ModelIR,
  ReferenceExecutionResult,
  ReferenceF32ExecutionResult,
} from "./types.js";

type ComparableTensor = DenseTensor | DenseF32Tensor;
type ExecutionResult = ReferenceExecutionResult | ReferenceF32ExecutionResult;

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

  const logitsSample = referenceById.get("lm_head") ?? referenceById.get("final_logit_softcap");
  const logits = logitsSample && sameShape(candidate.logits.shape, logitsSample.tensor.shape)
    ? compareTensor(candidate.logits, logitsSample.tensor, topK)
    : null;
  const complete = missingReferenceOperationIds.length === 0 && unexpectedReferenceOperationIds.length === 0 &&
    operations.every((comparison) => comparison.status === "pass") && kvCache.every((comparison) => comparison.status === "pass");
  const exact = operations.every((comparison) => exactMetrics(comparison.metrics)) &&
    kvCache.every((comparison) => exactMetrics(comparison.key) && exactMetrics(comparison.value));
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
