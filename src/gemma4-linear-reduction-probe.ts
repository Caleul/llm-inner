import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { createPagedDenseF32Matrix, pagedLinearF32 } from "./paged-dense.js";
import { fingerprintIR, readExecutionTraceBundle } from "./trace.js";
import type { DenseF32Tensor, DenseTensor, LinearOp, ModelIR, Operation, ReductionSchedule, TensorInfo } from "./types.js";

export interface Gemma4LinearReductionProfile {
  id: string;
  accumulationDtype: "F32" | "F64";
  reduction: ReductionSchedule;
}

export interface Gemma4LinearReductionProbeProfileResult {
  id: string;
  accumulationDtype: "F32" | "F64";
  reduction: ReductionSchedule;
  exact: boolean;
  mismatchedElements: number;
  maxAbsoluteError: number;
  firstMismatchedElement: number | null;
}

export interface Gemma4LinearReductionProbeInputGroup {
  /**
   * Every trace in a group is an independently captured repeat of precisely
   * these declared model inputs.  A group is the smallest unit that can show
   * a native capture is stable; different groups are evidence against a
   * schedule that only happens to fit one activation vector.
   */
  inputTokens: number[][];
  positionIds?: number[][];
  traces: string[];
  profiles: Gemma4LinearReductionProbeProfileResult[];
}

export interface Gemma4LinearReductionOutputFeatureProfileSpan {
  /** Inclusive output-feature index. */
  start: number;
  /** Exclusive output-feature index. */
  endExclusive: number;
  /**
   * Candidate profiles that matched every observed row for every distinct
   * declared input in this campaign. This is evidence only: it is not an
   * adapter policy and cannot authorize a coordinate-specific runtime rule.
   */
  profileIds: string[];
}

export interface Gemma4LinearReductionOutputFeatureCoverage {
  /** Number of distinct token/position input groups covered by this result. */
  inputGroupCount: number;
  /** The final dimension of the traced linear result. */
  outputFeatures: number;
  /** Run-length encoded common exact-profile sets, ordered by output feature. */
  spans: Gemma4LinearReductionOutputFeatureProfileSpan[];
  /** Features for which no one candidate profile matched every captured row/input. */
  uncoveredOutputFeatures: number[];
}

export interface Gemma4LinearReductionProbeReport {
  kind: "gemma4-linear-reduction-profile-probe";
  artifact: string;
  /** Independent native captures that agreed on the traced producer/result. */
  traces: string[];
  traceCount: number;
  operationId: string;
  inputOperationId: string;
  sourceCheckpointAccessed: false;
  reference: {
    runtime: string;
    executionDevice?: string;
    model: string;
    revisionOrChecksum: string;
    containerFormat: string;
    quantization: string;
    inputTokens: number[][];
    dtypePolicy: string;
  };
  /** Stable repeated captures, partitioned by declared prompt/positions. */
  inputGroups: Gemma4LinearReductionProbeInputGroup[];
  profiles: Gemma4LinearReductionProbeProfileResult[];
  exactProfileIds: string[];
  /**
   * Cross-input, per-output-feature evidence. This exposes a possible
   * kernel-output-tile boundary without pretending that a fitted feature map
   * is established model semantics.
   */
  outputFeatureCoverage: Gemma4LinearReductionOutputFeatureCoverage;
}

/**
 * Measures only a declared linear assignment against an integrity-bound
 * native operation trace.  It is deliberately diagnostic: a profile is never
 * installed into an adapter by this function, and a partial coordinate match
 * is never promoted to an exact schedule.
 */
export async function probeGemma4LiteralLinearReductionProfiles(options: {
  artifact: string;
  traces: readonly string[];
  operationId: string;
  profiles: readonly Gemma4LinearReductionProfile[];
  maxReadBytes: number;
  /** Require evidence over this many distinct declared prompt/position pairs. */
  minDistinctInputs?: number;
}): Promise<Gemma4LinearReductionProbeReport> {
  if (!Number.isSafeInteger(options.maxReadBytes) || options.maxReadBytes <= 0) throw new Error("Probe de redução Gemma 4 requer maxReadBytes positivo seguro.");
  const traces = validateTracePaths(options.traces);
  const profiles = validateProfiles(options.profiles);
  const decoded = await Promise.all(traces.map((trace) => readExecutionTraceBundle(trace)));
  const minDistinctInputs = options.minDistinctInputs ?? 1;
  if (!Number.isSafeInteger(minDistinctInputs) || minDistinctInputs < 1) throw new Error("Probe de redução Gemma 4 requer minDistinctInputs inteiro positivo.");
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    const fingerprint = fingerprintIR(artifact.program.textProgram);
    for (const trace of decoded) assertCompatibleTrace(trace, fingerprint);
    for (let index = 1; index < decoded.length; index += 1) assertSameProbeContract(decoded[0]!, decoded[index]!);
    const operation = findLinearOperation(artifact.program.textProgram, options.operationId);
    const inputOperation = findProducerOperation(artifact.program.textProgram, operation.input);
    const matrix = createPagedDenseF32Matrix(tensorInfo(artifact, operation.weight.name, operation.weight.shape, operation.weight.storageDtype), artifact, options.maxReadBytes);
    const groups = groupByDeclaredInputs(traces, decoded);
    if (groups.length < minDistinctInputs) throw new Error(`Probe de redução Gemma 4 requer ${minDistinctInputs} entradas declaradas distintas; recebeu ${groups.length}.`);
    const outputFeatureCoverage = createOutputFeatureCoverage(profiles);
    // Keep one bounded matrix read active at a time.  `maxReadBytes` is an
    // evidence/runtime limit, not a per-prompt suggestion that a campaign may
    // multiply through Promise.all.
    const inputGroups: Gemma4LinearReductionProbeInputGroup[] = [];
    for (const group of groups) {
      const first = group.captures[0]!;
      const input = traceTensor(first.decoded.reference.operations, inputOperation.id, inputOperation.output);
      const reference = traceTensor(first.decoded.reference.operations, operation.id, operation.output);
      for (let index = 1; index < group.captures.length; index += 1) {
        const repeated = group.captures[index]!.decoded;
        assertSameTraceIdentity(first.decoded, repeated);
        assertSameTensor(input, traceTensor(repeated.reference.operations, inputOperation.id, inputOperation.output), `${inputOperation.id}: produtor nativo não repetível`);
        assertSameTensor(reference, traceTensor(repeated.reference.operations, operation.id, operation.output), `${operation.id}: resultado nativo não repetível`);
      }
      const results: Gemma4LinearReductionProbeProfileResult[] = [];
      for (const profile of profiles) {
        const candidate = await pagedLinearF32(input, matrix, {
          outputDtype: operation.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32",
          accumulationDtype: profile.accumulationDtype,
          reduction: profile.reduction,
        });
        results.push(compareProfile(profile, candidate, reference, outputFeatureCoverage));
      }
      inputGroups.push({
        inputTokens: cloneInputs(first.decoded.reference.inputTokens),
        ...(first.decoded.reference.positionIds ? { positionIds: cloneInputs(first.decoded.reference.positionIds) } : {}),
        traces: group.captures.map((capture) => capture.trace),
        profiles: results,
      });
    }
    const results = profiles.map((profile) => aggregateProfile(profile, inputGroups.map((group) => group.profiles.find((result) => result.id === profile.id)!)));
    return {
      kind: "gemma4-linear-reduction-profile-probe",
      artifact: artifact.artifact,
      traces: [...traces],
      traceCount: traces.length,
      operationId: operation.id,
      inputOperationId: inputOperation.id,
      sourceCheckpointAccessed: false,
      reference: {
        runtime: decoded[0]!.reference.runtime,
        ...(decoded[0]!.reference.executionDevice === undefined ? {} : { executionDevice: decoded[0]!.reference.executionDevice }),
        model: decoded[0]!.reference.model,
        revisionOrChecksum: decoded[0]!.reference.revisionOrChecksum,
        containerFormat: decoded[0]!.reference.containerFormat,
        quantization: decoded[0]!.reference.quantization,
        inputTokens: cloneInputs(decoded[0]!.reference.inputTokens),
        dtypePolicy: decoded[0]!.reference.dtypePolicy,
      },
      inputGroups,
      profiles: results,
      exactProfileIds: results.filter((result) => result.exact).map((result) => result.id),
      outputFeatureCoverage: finalizeOutputFeatureCoverage(outputFeatureCoverage, groups.length),
    };
  } finally {
    await artifact.close();
  }
}

function groupByDeclaredInputs(
  traces: readonly string[],
  decoded: readonly Awaited<ReturnType<typeof readExecutionTraceBundle>>[],
): Array<{ captures: Array<{ trace: string; decoded: Awaited<ReturnType<typeof readExecutionTraceBundle>> }> }> {
  const groups = new Map<string, Array<{ trace: string; decoded: Awaited<ReturnType<typeof readExecutionTraceBundle>> }>>();
  const captureIds = new Set<string>();
  for (let index = 0; index < decoded.length; index += 1) {
    const entry = decoded[index]!;
    if (captureIds.has(entry.bundle.captureId!)) throw new Error("Probe de redução requer captureId distinto por trace.");
    captureIds.add(entry.bundle.captureId!);
    const key = JSON.stringify({ inputTokens: entry.reference.inputTokens, positionIds: entry.reference.positionIds ?? null });
    const group = groups.get(key) ?? [];
    group.push({ trace: traces[index]!, decoded: entry });
    groups.set(key, group);
  }
  for (const group of groups.values()) if (group.length < 2) throw new Error("Probe de redução requer ao menos dois traces nativos independentes por entrada declarada.");
  return [...groups.values()].map((captures) => ({ captures }));
}

function cloneInputs(inputs: readonly number[][]): number[][] {
  return inputs.map((row) => [...row]);
}

function aggregateProfile(profile: Gemma4LinearReductionProfile, results: readonly Gemma4LinearReductionProbeProfileResult[]): Gemma4LinearReductionProbeProfileResult {
  if (results.length === 0) throw new Error(`${profile.id}: probe sem resultados por entrada.`);
  const firstMismatch = results.find((result) => result.firstMismatchedElement !== null)?.firstMismatchedElement ?? null;
  return {
    id: profile.id,
    accumulationDtype: profile.accumulationDtype,
    reduction: structuredClone(profile.reduction),
    exact: results.every((result) => result.exact),
    mismatchedElements: results.reduce((total, result) => total + result.mismatchedElements, 0),
    maxAbsoluteError: Math.max(...results.map((result) => result.maxAbsoluteError)),
    firstMismatchedElement: firstMismatch,
  };
}

function validateTracePaths(traces: readonly string[]): string[] {
  if (traces.length < 2) throw new Error("Probe de redução Gemma 4 requer ao menos dois traces nativos independentes.");
  if (traces.some((trace) => !trace)) throw new Error("Probe de redução Gemma 4 recebeu caminho de trace vazio.");
  if (new Set(traces).size !== traces.length) throw new Error("Probe de redução Gemma 4 requer arquivos de trace distintos.");
  return [...traces];
}

function assertCompatibleTrace(trace: Awaited<ReturnType<typeof readExecutionTraceBundle>>, fingerprint: string): void {
  if (trace.bundle.candidatePolicy.dtype !== "F32" || trace.bundle.candidatePolicy.runtime !== "llm-inner paged Gemma4Text literal F32") {
    throw new Error("Probe de redução requer trace Gemma4Text F32 paginado compatível.");
  }
  if (trace.reference.executionDevice !== "cpu" && trace.reference.executionDevice !== "mps") {
    throw new Error("Probe de redução requer executionDevice explícito cpu ou mps no trace nativo.");
  }
  if (trace.bundle.irFingerprint !== fingerprint) throw new Error("Trace de redução Gemma 4 não corresponde ao programa textual do artefato literal.");
  if (!trace.bundle.captureId) throw new Error("Probe de redução requer captureId por trace para provar capturas independentes.");
}

function assertSameTraceIdentity(
  left: Awaited<ReturnType<typeof readExecutionTraceBundle>>,
  right: Awaited<ReturnType<typeof readExecutionTraceBundle>>,
): void {
  if (left.bundle.captureId === right.bundle.captureId) throw new Error("Probe de redução requer captureId distinto por trace.");
  const identity = (trace: Awaited<ReturnType<typeof readExecutionTraceBundle>>) => JSON.stringify({
    source: trace.bundle.source,
    irFingerprint: trace.bundle.irFingerprint,
    candidatePolicy: trace.bundle.candidatePolicy,
    reference: {
      runtime: trace.reference.runtime,
      executionDevice: trace.reference.executionDevice ?? null,
      model: trace.reference.model,
      revisionOrChecksum: trace.reference.revisionOrChecksum,
      containerFormat: trace.reference.containerFormat,
      quantization: trace.reference.quantization,
      inputTokens: trace.reference.inputTokens,
      positionIds: trace.reference.positionIds,
      dtypePolicy: trace.reference.dtypePolicy,
    },
  });
  if (identity(left) !== identity(right)) throw new Error("Probe de redução recebeu traces com identidade de referência diferente.");
}

/**
 * A multi-input campaign may vary only declared tokens/positions and capture
 * UUIDs.  Source identity, runtime, artifact semantics and dtype contract
 * must remain fixed so a profile cannot combine unrelated native behaviours.
 */
function assertSameProbeContract(
  left: Awaited<ReturnType<typeof readExecutionTraceBundle>>,
  right: Awaited<ReturnType<typeof readExecutionTraceBundle>>,
): void {
  const contract = (trace: Awaited<ReturnType<typeof readExecutionTraceBundle>>) => JSON.stringify({
    source: trace.bundle.source,
    irFingerprint: trace.bundle.irFingerprint,
    candidatePolicy: trace.bundle.candidatePolicy,
    reference: {
      runtime: trace.reference.runtime,
      executionDevice: trace.reference.executionDevice ?? null,
      model: trace.reference.model,
      revisionOrChecksum: trace.reference.revisionOrChecksum,
      containerFormat: trace.reference.containerFormat,
      quantization: trace.reference.quantization,
      dtypePolicy: trace.reference.dtypePolicy,
    },
  });
  if (contract(left) !== contract(right)) throw new Error("Probe de redução recebeu traces com contrato de referência diferente.");
}

function validateProfiles(profiles: readonly Gemma4LinearReductionProfile[]): Gemma4LinearReductionProfile[] {
  if (profiles.length === 0) throw new Error("Probe de redução requer ao menos um perfil.");
  const ids = new Set<string>();
  return profiles.map((profile) => {
    if (!profile.id || ids.has(profile.id)) throw new Error("Probe de redução requer IDs de perfil únicos e não vazios.");
    ids.add(profile.id);
    const reduction = profile.reduction;
    if (reduction.kind === "ordered-scalar" || reduction.kind === "ordered-fma") {
      if (reduction.indexOrder !== "ascending") throw new Error(`${profile.id}: redução escalar não canônica.`);
      if (reduction.kind === "ordered-fma" && profile.accumulationDtype !== "F32") throw new Error(`${profile.id}: redução FMA escalar requer acumulador F32.`);
      return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(reduction) };
    }
    if (reduction.kind === "blocked-f32-terms") {
      if (profile.accumulationDtype !== "F32" || !Number.isSafeInteger(reduction.termsPerBlock) || reduction.termsPerBlock < 2 ||
        reduction.inputBlock !== "contiguous-terms" || reduction.blockOrder !== "ascending" ||
        (reduction.termOrder !== "ascending" && reduction.termOrder !== "descending") ||
        (reduction.productBoundary !== "separately-rounded-f32" && reduction.productBoundary !== "fused-fma")) {
        throw new Error(`${profile.id}: perfil de blocos F32 inválido.`);
      }
      return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(reduction) };
    }
    if (reduction.kind === "blocked-tiled-f32-lanes") {
      if (profile.accumulationDtype !== "F32" || !Number.isSafeInteger(reduction.laneCount) || reduction.laneCount < 2 ||
        !Number.isSafeInteger(reduction.termsPerLane) || reduction.termsPerLane < 2 || reduction.inputBlock !== "tile-contiguous-terms" || reduction.blockOrder !== "ascending" ||
        (reduction.laneReductionOrder !== "ascending" && reduction.laneReductionOrder !== "descending" && reduction.laneReductionOrder !== "balanced-pairwise") ||
        (reduction.productBoundary !== "separately-rounded-f32" && reduction.productBoundary !== "fused-fma")) {
        throw new Error(`${profile.id}: perfil de blocos tiled F32 inválido.`);
      }
      return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(reduction) };
    }
    if (profile.accumulationDtype !== "F32" || !Number.isSafeInteger(reduction.laneCount) || reduction.laneCount < 2 ||
      (reduction.laneReductionOrder !== "ascending" && reduction.laneReductionOrder !== "descending" && reduction.laneReductionOrder !== "balanced-pairwise")) {
      throw new Error(`${profile.id}: perfil de lanes F32 inválido.`);
    }
    const tiled = reduction.kind === "tiled-f32-lanes" || reduction.kind === "tiled-fma-lanes";
    if ((!tiled && reduction.inputLane !== "index-modulo-lane-count") ||
      (tiled && (!Number.isSafeInteger(reduction.termsPerLane) || reduction.termsPerLane < 2 || reduction.inputLane !== "tile-contiguous-terms"))) {
      throw new Error(`${profile.id}: mapeamento de lanes F32 inválido.`);
    }
    return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(reduction) };
  });
}

function findLinearOperation(program: ModelIR, operationId: string): LinearOp {
  const operation = operations(program).find((entry) => entry.id === operationId);
  if (!operation || operation.op !== "linear" || !operation.transposeWeight || operation.bias) throw new Error(`${operationId}: probe requer linear declarado [out,in] sem bias.`);
  return operation as LinearOp;
}

function findProducerOperation(program: ModelIR, output: string): Pick<Operation, "id" | "output"> {
  const producer = operations(program).find((entry) => entry.output === output);
  if (!producer) throw new Error(`Probe de redução não encontrou atribuição produtora para '${output}'.`);
  return producer;
}

function traceTensor(samples: readonly { operationId: string; output: string; tensor: DenseF32Tensor | DenseTensor }[], operationId: string, output: string): DenseF32Tensor {
  const sample = samples.find((entry) => entry.operationId === operationId);
  if (!sample || sample.output !== output) throw new Error(`Trace de redução não contém ${operationId}/${output}.`);
  if (!(sample.tensor.values instanceof Float32Array)) throw new Error(`Trace de redução ${operationId} não declara tensor F32.`);
  return sample.tensor as DenseF32Tensor;
}

function assertSameTensor(left: DenseF32Tensor, right: DenseF32Tensor, label: string): void {
  if (left.shape.length !== right.shape.length || left.shape.some((dimension, index) => dimension !== right.shape[index]) || left.values.length !== right.values.length) {
    throw new Error(`${label}: shape divergente entre traces.`);
  }
  for (let index = 0; index < left.values.length; index += 1) {
    if (!Object.is(left.values[index], right.values[index])) throw new Error(`${label}: valor divergente no índice ${index}.`);
  }
}

function tensorInfo(
  artifact: Awaited<ReturnType<typeof openGemma4CompositeLiteralArtifact>>,
  name: string,
  shape: readonly number[],
  storageDtype: string,
): TensorInfo {
  const constant = artifact.constants.get(name);
  if (!constant || constant.storageDtype !== storageDtype || constant.logicalShape.length !== shape.length || constant.logicalShape.some((dimension, index) => dimension !== shape[index])) {
    throw new Error(`Probe de redução não encontrou constante densa compatível para '${name}'.`);
  }
  return { name: constant.name, storageDtype: constant.storageDtype, storageShape: [...constant.storageShape], logicalShape: [...constant.logicalShape] };
}

function operations(program: ModelIR): Operation[] {
  return [...program.prelude, ...program.layers.flatMap((layer) => layer.operations), ...program.epilogue];
}

interface MutableOutputFeatureCoverage {
  readonly profileMatches: Map<string, boolean[]>;
  outputFeatures: number | null;
}

function createOutputFeatureCoverage(profiles: readonly Gemma4LinearReductionProfile[]): MutableOutputFeatureCoverage {
  return { profileMatches: new Map(profiles.map((profile) => [profile.id, []])), outputFeatures: null };
}

function finalizeOutputFeatureCoverage(coverage: MutableOutputFeatureCoverage, inputGroupCount: number): Gemma4LinearReductionOutputFeatureCoverage {
  const outputFeatures = coverage.outputFeatures;
  if (outputFeatures === null) throw new Error("Probe de redução Gemma 4 não produziu cobertura de output.");
  const profileIdsByFeature = Array.from({ length: outputFeatures }, (_unused, outputFeature) =>
    [...coverage.profileMatches].filter(([, matches]) => matches[outputFeature]).map(([id]) => id),
  );
  const spans: Gemma4LinearReductionOutputFeatureProfileSpan[] = [];
  for (let start = 0; start < outputFeatures;) {
    const profileIds = profileIdsByFeature[start]!;
    let endExclusive = start + 1;
    while (endExclusive < outputFeatures && sameIds(profileIds, profileIdsByFeature[endExclusive]!)) endExclusive += 1;
    spans.push({ start, endExclusive, profileIds });
    start = endExclusive;
  }
  return {
    inputGroupCount,
    outputFeatures,
    spans,
    uncoveredOutputFeatures: profileIdsByFeature.flatMap((profileIds, outputFeature) => profileIds.length === 0 ? [outputFeature] : []),
  };
}

function sameIds(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((id, index) => id === right[index]);
}

function compareProfile(
  profile: Gemma4LinearReductionProfile,
  candidate: DenseF32Tensor,
  reference: DenseF32Tensor,
  coverage: MutableOutputFeatureCoverage,
): Gemma4LinearReductionProbeProfileResult {
  if (candidate.shape.length !== reference.shape.length || candidate.shape.some((dimension, index) => dimension !== reference.shape[index]) || candidate.values.length !== reference.values.length) {
    throw new Error(`${profile.id}: perfil produziu shape incompatível com o checkpoint nativo.`);
  }
  if (candidate.shape.length === 0) throw new Error(`${profile.id}: perfil produziu tensor escalar sem dimensão de output.`);
  const outputFeatures = candidate.shape[candidate.shape.length - 1]!;
  if (!Number.isSafeInteger(outputFeatures) || outputFeatures <= 0) throw new Error(`${profile.id}: perfil produziu dimensão de output inválida.`);
  if (coverage.outputFeatures === null) {
    coverage.outputFeatures = outputFeatures;
    for (const matches of coverage.profileMatches.values()) matches.push(...Array.from({ length: outputFeatures }, () => true));
  } else if (coverage.outputFeatures !== outputFeatures) {
    throw new Error(`${profile.id}: perfil produziu dimensão de output incompatível entre entradas.`);
  }
  const featureMatches = coverage.profileMatches.get(profile.id);
  if (!featureMatches) throw new Error(`${profile.id}: perfil não possui cobertura de output registrada.`);
  let mismatchedElements = 0, maxAbsoluteError = 0, firstMismatchedElement: number | null = null;
  for (let index = 0; index < candidate.values.length; index += 1) {
    const actual = candidate.values[index]!, expected = reference.values[index]!;
    if (Object.is(actual, expected)) continue;
    featureMatches[index % outputFeatures] = false;
    mismatchedElements += 1;
    firstMismatchedElement ??= index;
    const error = Math.abs(actual - expected);
    if (error > maxAbsoluteError) maxAbsoluteError = error;
  }
  return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(profile.reduction), exact: mismatchedElements === 0, mismatchedElements, maxAbsoluteError, firstMismatchedElement };
}
