import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
  GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY,
  GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE,
  GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT,
  type Gemma4RuntimeReductionExecutionProtocol,
  type Gemma4RuntimeReductionExecutableReplayContract,
} from "./gemma4-authoritative-runtime.js";
import {
  gemma4RuntimeReductionInvocationProgram,
  gemma4RuntimeReductionInvocationProgramSha256,
  type Gemma4LiteralRuntimeReductionOperationClass,
} from "./gemma4-runtime-reduction-invocation.js";
import type { Gemma4AudioProgram } from "./gemma4-audio.js";
import type { Gemma4VisionProgram } from "./gemma4-vision.js";
import type { DenseF32Tensor } from "./types.js";

export type Gemma4RuntimeReductionRequest =
  | {
    scope: "vision";
    operationId: string;
    operation: "attention-score-matmul";
    program: Gemma4VisionProgram;
    operands: readonly [DenseF32Tensor, DenseF32Tensor];
  }
  | {
    scope: "vision";
    operationId: string;
    operation: "attention-value-matmul";
    program: Gemma4VisionProgram;
    operands: readonly [DenseF32Tensor, DenseF32Tensor];
  }
  | {
    scope: "audio";
    operationId: string;
    operation: "chunked-attention-content-matmul" | "relative-attention-position-matmul" | "chunked-relative-attention-values";
    program: Gemma4AudioProgram;
    operands: readonly [DenseF32Tensor, DenseF32Tensor];
  };

export interface Gemma4RuntimeReductionAttestation {
  runtime: "torch-2.12.1";
  torchBuildCommit: "7269437d655783a26cba32aa88195b741ff496aa";
  executionMode: "torch.inference_mode";
  device: "cpu";
  platform: "Darwin-arm64";
  backend: "Apple Accelerate SGEMM";
  blasBuildSetting: "BLAS_INFO=accelerate";
  runtimeProcessEnvironment: typeof GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT;
  runtimeEnvironmentIdentity: typeof GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY;
  runtimeExecutionState: typeof GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE;
}

export interface Gemma4RuntimeReductionTensorEvidence {
  encoding: "ieee-f32-little-endian";
  shape: number[];
  bytes: number;
  sha256: string;
}

export interface Gemma4RuntimeReductionExecutionEvidence {
  schemaVersion: 4;
  contractId: "torch-2.12.1-cpu-inference-matmul-v1";
  scope: Gemma4RuntimeReductionRequest["scope"];
  operationId: string;
  operation: Gemma4RuntimeReductionRequest["operation"];
  sourceCheckpointAccessed: false;
  adapterProgramSha256: typeof GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256;
  executionProtocolSha256: string;
  invocationProgramId: Gemma4LiteralRuntimeReductionOperationClass;
  invocationProgramSha256: string;
  runtimeAttestation: Gemma4RuntimeReductionAttestation;
  orderedOperands: [Gemma4RuntimeReductionTensorEvidence, Gemma4RuntimeReductionTensorEvidence];
  output: Gemma4RuntimeReductionTensorEvidence;
}

export interface Gemma4RuntimeReductionExecution {
  output: DenseF32Tensor;
  evidence: Gemma4RuntimeReductionExecutionEvidence;
}

export interface Gemma4RuntimeReductionEvidenceContract {
  providerContractId: Gemma4RuntimeReductionExecutableReplayContract["providerContractId"];
  adapterProgramSha256: typeof GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256;
  executionProtocolSha256: string;
  invocationPrograms: Gemma4RuntimeReductionExecutableReplayContract["invocationPrograms"];
  runtimeAttestation: Gemma4RuntimeReductionAttestation;
}

export function expectedGemma4RuntimeReductionAttestation(): Gemma4RuntimeReductionAttestation {
  return {
    runtime: "torch-2.12.1",
    torchBuildCommit: "7269437d655783a26cba32aa88195b741ff496aa",
    executionMode: "torch.inference_mode",
    device: "cpu",
    platform: "Darwin-arm64",
    backend: "Apple Accelerate SGEMM",
    blasBuildSetting: "BLAS_INFO=accelerate",
    runtimeProcessEnvironment: structuredClone(GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT),
    runtimeEnvironmentIdentity: structuredClone(GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY),
    runtimeExecutionState: structuredClone(GEMMA4_RUNTIME_REDUCTION_EXECUTION_STATE),
  };
}

export function gemma4RuntimeReductionEvidenceContract(
  replay: Gemma4RuntimeReductionExecutableReplayContract,
): Gemma4RuntimeReductionEvidenceContract {
  return {
    providerContractId: replay.providerContractId,
    adapterProgramSha256: replay.adapterProgram.sha256,
    executionProtocolSha256: gemma4RuntimeReductionExecutionProtocolSha256(replay.executionProtocol),
    invocationPrograms: structuredClone(replay.invocationPrograms),
    runtimeAttestation: {
      runtime: replay.runtime,
      torchBuildCommit: replay.torchBuildCommit,
      executionMode: replay.executionMode,
      device: replay.device,
      platform: replay.platform,
      backend: replay.backend,
      blasBuildSetting: replay.blasBuildSetting,
      runtimeProcessEnvironment: structuredClone(replay.runtimeProcessEnvironment),
      runtimeEnvironmentIdentity: structuredClone(replay.runtimeEnvironmentIdentity),
      runtimeExecutionState: structuredClone(replay.runtimeExecutionState),
    },
  };
}

export function gemma4RuntimeReductionExecutionProtocolSha256(
  protocol: Gemma4RuntimeReductionExecutionProtocol,
): string {
  return createHash("sha256").update(JSON.stringify(protocol), "utf8").digest("hex");
}

/**
 * Executes only a serialized runtime-defined reduction. Implementations do
 * not receive a checkpoint path, tensor catalog, weights, or trace outputs.
 */
export interface Gemma4RuntimeReductionProvider {
  readonly contractId: "torch-2.12.1-cpu-inference-matmul-v1";
  readonly evidenceContract: Gemma4RuntimeReductionEvidenceContract;
  readonly executions: readonly Gemma4RuntimeReductionExecutionEvidence[];
  execute(request: Gemma4RuntimeReductionRequest): Gemma4RuntimeReductionExecution;
}

export interface Gemma4RuntimeReductionReplayEvidence {
  contractId: "torch-2.12.1-cpu-inference-matmul-v1";
  executionCount: number;
  executions: Gemma4RuntimeReductionExecutionEvidence[];
}

export function executeGemma4RuntimeReduction(
  provider: Gemma4RuntimeReductionProvider,
  request: Gemma4RuntimeReductionRequest,
  expectedShape: readonly number[],
): DenseF32Tensor {
  const evidenceContract = provider.evidenceContract;
  if (provider.contractId !== "torch-2.12.1-cpu-inference-matmul-v1") {
    throw new Error(`${request.operationId}: provedor de redução Gemma 4 não corresponde ao contrato fixado.`);
  }
  if (evidenceContract.providerContractId !== provider.contractId) {
    throw new Error(`${request.operationId}: contrato de evidência da redução Gemma 4 diverge do provedor.`);
  }
  assertProgramOperation(request);
  const execution = provider.execute(request);
  const result = execution.output;
  const elements = expectedShape.reduce((total, dimension) => total * dimension, 1);
  if (result.shape.length !== expectedShape.length || result.shape.some((dimension, index) => dimension !== expectedShape[index]) ||
    result.values.length !== elements || result.values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${request.operationId}: provedor de redução Gemma 4 retornou tensor inválido ou shape divergente.`);
  }
  const invocation = gemma4RuntimeReductionInvocationProgram(
    evidenceContract.invocationPrograms,
    request.scope,
    request.operation,
  );
  const expectedEvidence: Gemma4RuntimeReductionExecutionEvidence = {
    schemaVersion: 4,
    contractId: provider.contractId,
    scope: request.scope,
    operationId: request.operationId,
    operation: request.operation,
    sourceCheckpointAccessed: false,
    adapterProgramSha256: evidenceContract.adapterProgramSha256,
    executionProtocolSha256: evidenceContract.executionProtocolSha256,
    invocationProgramId: invocation.id,
    invocationProgramSha256: gemma4RuntimeReductionInvocationProgramSha256(invocation),
    runtimeAttestation: structuredClone(evidenceContract.runtimeAttestation),
    orderedOperands: [tensorEvidence(request.operands[0]), tensorEvidence(request.operands[1])],
    output: tensorEvidence(result),
  };
  if (!isDeepStrictEqual(execution.evidence, expectedEvidence)) {
    throw new Error(`${request.operationId}: evidência do provedor de redução Gemma 4 está incompleta ou divergente.`);
  }
  return result;
}

export function gemma4RuntimeReductionTensorEvidence(tensor: DenseF32Tensor): Gemma4RuntimeReductionTensorEvidence {
  return tensorEvidence(tensor);
}

export function gemma4RuntimeReductionReplayEvidence(
  provider: Gemma4RuntimeReductionProvider,
  startOrdinal: number,
): Gemma4RuntimeReductionReplayEvidence {
  if (!Number.isSafeInteger(startOrdinal) || startOrdinal < 0 || startOrdinal > provider.executions.length) {
    throw new Error("Ordinal inicial de evidência de redução Gemma 4 inválido.");
  }
  const executions = structuredClone(provider.executions.slice(startOrdinal));
  return { contractId: provider.contractId, executionCount: executions.length, executions };
}

function assertProgramOperation(request: Gemma4RuntimeReductionRequest): void {
  const assignments = request.program.assignments;
  const matches = assignments.filter((assignment) => assignment.id === request.operationId);
  const assignment = matches[0];
  if (matches.length !== 1 || assignment?.operation !== request.operation || assignment.inputs.length !== 2 ||
    assignment.dtypePolicy?.accumulationDtype !== "runtime-defined") {
    throw new Error(`${request.operationId}: pedido de redução não corresponde a uma única BMM runtime-defined serializada.`);
  }
}

function tensorEvidence(tensor: DenseF32Tensor): Gemma4RuntimeReductionTensorEvidence {
  const bytes = Buffer.from(tensor.values.buffer, tensor.values.byteOffset, tensor.values.byteLength);
  return {
    encoding: "ieee-f32-little-endian",
    shape: [...tensor.shape],
    bytes: bytes.byteLength,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}
