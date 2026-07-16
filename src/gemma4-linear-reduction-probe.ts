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

export interface Gemma4LinearReductionProbeReport {
  kind: "gemma4-linear-reduction-profile-probe";
  artifact: string;
  operationId: string;
  inputOperationId: string;
  sourceCheckpointAccessed: false;
  reference: {
    runtime: string;
    model: string;
    revisionOrChecksum: string;
    containerFormat: string;
    quantization: string;
    inputTokens: number[][];
    dtypePolicy: string;
  };
  profiles: Gemma4LinearReductionProbeProfileResult[];
  exactProfileIds: string[];
}

/**
 * Measures only a declared linear assignment against an integrity-bound
 * native operation trace.  It is deliberately diagnostic: a profile is never
 * installed into an adapter by this function, and a partial coordinate match
 * is never promoted to an exact schedule.
 */
export async function probeGemma4LiteralLinearReductionProfiles(options: {
  artifact: string;
  trace: string;
  operationId: string;
  profiles: readonly Gemma4LinearReductionProfile[];
  maxReadBytes: number;
}): Promise<Gemma4LinearReductionProbeReport> {
  if (!Number.isSafeInteger(options.maxReadBytes) || options.maxReadBytes <= 0) throw new Error("Probe de redução Gemma 4 requer maxReadBytes positivo seguro.");
  const profiles = validateProfiles(options.profiles);
  const decoded = await readExecutionTraceBundle(options.trace);
  if (decoded.bundle.candidatePolicy.dtype !== "F32" || decoded.bundle.candidatePolicy.runtime !== "llm-inner paged Gemma4Text literal F32") {
    throw new Error("Probe de redução requer trace Gemma4Text F32 paginado compatível.");
  }
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    if (fingerprintIR(artifact.program.textProgram) !== decoded.bundle.irFingerprint) throw new Error("Trace de redução Gemma 4 não corresponde ao programa textual do artefato literal.");
    const operation = findLinearOperation(artifact.program.textProgram, options.operationId);
    const inputOperation = findProducerOperation(artifact.program.textProgram, operation.input);
    const input = traceTensor(decoded.reference.operations, inputOperation.id, inputOperation.output);
    const reference = traceTensor(decoded.reference.operations, operation.id, operation.output);
    const matrix = createPagedDenseF32Matrix(tensorInfo(artifact, operation.weight.name, operation.weight.shape, operation.weight.storageDtype), artifact, options.maxReadBytes);
    const results: Gemma4LinearReductionProbeProfileResult[] = [];
    for (const profile of profiles) {
      const candidate = await pagedLinearF32(input, matrix, {
        outputDtype: operation.dtypePolicy.outputDtype === "BF16" ? "BF16" : "F32",
        accumulationDtype: profile.accumulationDtype,
        reduction: profile.reduction,
      });
      results.push(compareProfile(profile, candidate, reference));
    }
    return {
      kind: "gemma4-linear-reduction-profile-probe",
      artifact: artifact.artifact,
      operationId: operation.id,
      inputOperationId: inputOperation.id,
      sourceCheckpointAccessed: false,
      reference: {
        runtime: decoded.reference.runtime,
        model: decoded.reference.model,
        revisionOrChecksum: decoded.reference.revisionOrChecksum,
        containerFormat: decoded.reference.containerFormat,
        quantization: decoded.reference.quantization,
        inputTokens: decoded.reference.inputTokens.map((row) => [...row]),
        dtypePolicy: decoded.reference.dtypePolicy,
      },
      profiles: results,
      exactProfileIds: results.filter((result) => result.exact).map((result) => result.id),
    };
  } finally {
    await artifact.close();
  }
}

function validateProfiles(profiles: readonly Gemma4LinearReductionProfile[]): Gemma4LinearReductionProfile[] {
  if (profiles.length === 0) throw new Error("Probe de redução requer ao menos um perfil.");
  const ids = new Set<string>();
  return profiles.map((profile) => {
    if (!profile.id || ids.has(profile.id)) throw new Error("Probe de redução requer IDs de perfil únicos e não vazios.");
    ids.add(profile.id);
    if (profile.reduction.kind === "ordered-scalar") {
      if (profile.reduction.indexOrder !== "ascending") throw new Error(`${profile.id}: redução escalar não canônica.`);
    } else if (profile.accumulationDtype !== "F32" || !Number.isSafeInteger(profile.reduction.laneCount) || profile.reduction.laneCount < 2 ||
      profile.reduction.inputLane !== "index-modulo-lane-count" || profile.reduction.laneReductionOrder !== "ascending") {
      throw new Error(`${profile.id}: perfil de lanes F32 inválido.`);
    }
    return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(profile.reduction) };
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

function compareProfile(profile: Gemma4LinearReductionProfile, candidate: DenseF32Tensor, reference: DenseF32Tensor): Gemma4LinearReductionProbeProfileResult {
  if (candidate.shape.length !== reference.shape.length || candidate.shape.some((dimension, index) => dimension !== reference.shape[index]) || candidate.values.length !== reference.values.length) {
    throw new Error(`${profile.id}: perfil produziu shape incompatível com o checkpoint nativo.`);
  }
  let mismatchedElements = 0, maxAbsoluteError = 0, firstMismatchedElement: number | null = null;
  for (let index = 0; index < candidate.values.length; index += 1) {
    const actual = candidate.values[index]!, expected = reference.values[index]!;
    if (Object.is(actual, expected)) continue;
    mismatchedElements += 1;
    firstMismatchedElement ??= index;
    const error = Math.abs(actual - expected);
    if (error > maxAbsoluteError) maxAbsoluteError = error;
  }
  return { id: profile.id, accumulationDtype: profile.accumulationDtype, reduction: structuredClone(profile.reduction), exact: mismatchedElements === 0, mismatchedElements, maxAbsoluteError, firstMismatchedElement };
}
