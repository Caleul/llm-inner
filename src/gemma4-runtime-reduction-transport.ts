import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  gemma4RuntimeReductionExecutionProtocol,
  type Gemma4RuntimeReductionAttestation,
  type Gemma4RuntimeReductionExecutableReplayContract,
} from "./gemma4-authoritative-runtime.js";
import type {
  Gemma4LiteralRuntimeReductionOperationClass,
  Gemma4RuntimeReductionInvocationProgram,
} from "./gemma4-runtime-reduction-invocation.js";
import type { DenseF32Tensor } from "./types.js";

type ExecutionProtocol = Gemma4RuntimeReductionExecutableReplayContract["executionProtocol"];

interface Gemma4RuntimeReductionTransportRequest {
  scope: "vision" | "audio";
  operationId: string;
  operation: "attention-score-matmul" | "attention-value-matmul" |
    "chunked-attention-content-matmul" | "relative-attention-position-matmul" |
    "chunked-relative-attention-values";
  program: {
    runtimeDtype: "BF16" | "F32";
    tower: object;
  };
  operands: readonly [DenseF32Tensor, DenseF32Tensor];
}

export interface Gemma4RuntimeReductionSerializedTensor {
  dtype: "F32";
  bitPattern: "IEEE-754 binary32";
  byteOrder: "little-endian";
  layout: "row-major-contiguous";
  shape: number[];
  byteLength: number;
  dataBase64: string;
}

export interface Gemma4RuntimeReductionRequestEnvelope {
  schemaVersion: 2;
  contractId: "torch-2.12.1-cpu-inference-matmul-v1";
  scope: Gemma4RuntimeReductionTransportRequest["scope"];
  operationId: string;
  operation: Gemma4RuntimeReductionTransportRequest["operation"];
  invocationProgram: Gemma4RuntimeReductionInvocationProgram;
  tower: Record<string, number | string>;
  operands: [Gemma4RuntimeReductionSerializedTensor, Gemma4RuntimeReductionSerializedTensor];
}

export interface Gemma4RuntimeReductionResponseEnvelope {
  schemaVersion: 2;
  contractId: "torch-2.12.1-cpu-inference-matmul-v1";
  operationId: string;
  scope: Gemma4RuntimeReductionTransportRequest["scope"];
  operation: Gemma4RuntimeReductionTransportRequest["operation"];
  invocationProgramId: Gemma4LiteralRuntimeReductionOperationClass;
  sourceCheckpointAccessed: false;
  runtimeAttestation: Gemma4RuntimeReductionAttestation;
  output: Gemma4RuntimeReductionSerializedTensor;
}

export interface Gemma4RuntimeReductionTranscriptEvidence {
  schemaVersion: 1;
  encoding: "utf8";
  hashAlgorithm: "sha256";
  requestBytes: number;
  requestSha256: string;
  responseBytes: number;
  responseSha256: string;
}

export function serializeGemma4RuntimeReductionRequestEnvelope(
  request: Gemma4RuntimeReductionTransportRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
  protocol: ExecutionProtocol,
): Gemma4RuntimeReductionRequestEnvelope {
  const envelope: Gemma4RuntimeReductionRequestEnvelope = {
    schemaVersion: 2,
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    scope: request.scope,
    operationId: request.operationId,
    operation: request.operation,
    invocationProgram,
    tower: serializeInvocationEnvironment(request, invocationProgram),
    operands: [
      serializeGemma4RuntimeReductionTensor(request.operands[0], protocol.tensorEncoding, `${request.operationId}: operand 0`),
      serializeGemma4RuntimeReductionTensor(request.operands[1], protocol.tensorEncoding, `${request.operationId}: operand 1`),
    ],
  };
  assertExactObjectFields(envelope, protocol.requestEnvelope.fields, `${request.operationId}: request envelope`);
  for (const [index, operand] of envelope.operands.entries()) {
    assertExactObjectFields(operand, protocol.tensorEncoding.fields, `${request.operationId}: request operand ${index}`);
  }
  return envelope;
}

export function serializeGemma4RuntimeReductionResponseEnvelope(
  request: Gemma4RuntimeReductionTransportRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
  runtimeAttestation: Gemma4RuntimeReductionAttestation,
  output: DenseF32Tensor,
  protocol: ExecutionProtocol,
): Gemma4RuntimeReductionResponseEnvelope {
  const envelope: Gemma4RuntimeReductionResponseEnvelope = {
    schemaVersion: 2,
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    operationId: request.operationId,
    scope: request.scope,
    operation: request.operation,
    invocationProgramId: invocationProgram.id,
    sourceCheckpointAccessed: false,
    runtimeAttestation: structuredClone(runtimeAttestation),
    output: serializeGemma4RuntimeReductionTensor(output, protocol.tensorEncoding, `${request.operationId}: response output`),
  };
  assertExactObjectFields(envelope, protocol.responseEnvelope.fields, `${request.operationId}: response envelope`);
  assertExactObjectFields(envelope.output, protocol.tensorEncoding.fields, `${request.operationId}: response output`);
  return envelope;
}

export function parseGemma4RuntimeReductionResponseEnvelope(
  stdout: string,
  request: Gemma4RuntimeReductionTransportRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
  protocol: ExecutionProtocol,
): Gemma4RuntimeReductionResponseEnvelope {
  let parsed: Gemma4RuntimeReductionResponseEnvelope;
  try { parsed = JSON.parse(stdout) as Gemma4RuntimeReductionResponseEnvelope; } catch (error) {
    throw new Error(`${request.operationId}: helper PyTorch retornou JSON inválido: ${(error as Error).message}`);
  }
  if (JSON.stringify(parsed) !== stdout) {
    throw new Error(`${request.operationId}: helper PyTorch não respeitou a serialização JSON compacta sem bytes finais.`);
  }
  assertExactObjectFields(parsed, protocol.responseEnvelope.fields, `${request.operationId}: response envelope`);
  assertExactObjectFields(parsed.output, protocol.tensorEncoding.fields, `${request.operationId}: response output`);
  if (parsed.schemaVersion !== protocol.responseEnvelope.schemaVersion ||
    parsed.contractId !== "torch-2.12.1-cpu-inference-matmul-v1" ||
    parsed.operationId !== request.operationId || parsed.scope !== request.scope || parsed.operation !== request.operation ||
    parsed.invocationProgramId !== invocationProgram.id ||
    parsed.sourceCheckpointAccessed !== false || !parsed.runtimeAttestation || !parsed.output) {
    throw new Error(`${request.operationId}: helper PyTorch retornou envelope incompatível.`);
  }
  return parsed;
}

export function gemma4RuntimeReductionTranscriptEvidence(
  requestUtf8: string,
  responseUtf8: string,
  protocol: ExecutionProtocol,
): Gemma4RuntimeReductionTranscriptEvidence {
  if (!isDeepStrictEqual(protocol.transcriptCommitment, gemma4RuntimeReductionExecutionProtocol().transcriptCommitment)) {
    throw new Error("Compromisso de transcript da redução Gemma 4 não é suportado ou está divergente.");
  }
  assertCanonicalJson(requestUtf8, "request");
  assertCanonicalJson(responseUtf8, "response");
  return {
    schemaVersion: 1,
    encoding: "utf8",
    hashAlgorithm: "sha256",
    requestBytes: Buffer.byteLength(requestUtf8, "utf8"),
    requestSha256: sha256Utf8(requestUtf8),
    responseBytes: Buffer.byteLength(responseUtf8, "utf8"),
    responseSha256: sha256Utf8(responseUtf8),
  };
}

export function serializeGemma4RuntimeReductionTensor(
  tensor: DenseF32Tensor,
  contract: ExecutionProtocol["tensorEncoding"] = gemma4RuntimeReductionExecutionProtocol().tensorEncoding,
  label = "runtime-reduction tensor",
): Gemma4RuntimeReductionSerializedTensor {
  assertSupportedTensorEncoding(contract);
  const elements = tensor.shape.reduce((total, dimension) => total * dimension, 1);
  if (!tensor.shape.length || tensor.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
    !Number.isSafeInteger(elements) || elements !== tensor.values.length || tensor.values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${label}: tensor F32 inválido para transporte lossless.`);
  }
  const payload = Buffer.allocUnsafe(elements * 4);
  tensor.values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
  return {
    dtype: "F32",
    bitPattern: "IEEE-754 binary32",
    byteOrder: "little-endian",
    layout: "row-major-contiguous",
    shape: [...tensor.shape],
    byteLength: payload.byteLength,
    dataBase64: payload.toString("base64"),
  };
}

export function deserializeGemma4RuntimeReductionTensor(
  serialized: Gemma4RuntimeReductionSerializedTensor,
  contract: ExecutionProtocol["tensorEncoding"] = gemma4RuntimeReductionExecutionProtocol().tensorEncoding,
  label = "runtime-reduction tensor",
): DenseF32Tensor {
  assertSupportedTensorEncoding(contract);
  assertExactObjectFields(serialized, contract.fields, label);
  if (serialized.dtype !== contract.dtype || serialized.bitPattern !== contract.bitPattern ||
    serialized.byteOrder !== contract.byteOrder || serialized.layout !== contract.layout ||
    !Array.isArray(serialized.shape) || !serialized.shape.length ||
    serialized.shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
    !Number.isSafeInteger(serialized.byteLength) || serialized.byteLength < 0 ||
    typeof serialized.dataBase64 !== "string" || !isCanonicalBase64(serialized.dataBase64)) {
    throw new Error(`${label}: envelope F32 incompatível com o protocolo serializado.`);
  }
  const elements = serialized.shape.reduce((total, dimension) => total * dimension, 1);
  const expectedBytes = elements * 4;
  if (!Number.isSafeInteger(elements) || !Number.isSafeInteger(expectedBytes) || serialized.byteLength !== expectedBytes) {
    throw new Error(`${label}: shape e byteLength não correspondem.`);
  }
  const payload = Buffer.from(serialized.dataBase64, "base64");
  if (payload.byteLength !== expectedBytes || payload.toString("base64") !== serialized.dataBase64) {
    throw new Error(`${label}: payload Base64 não é canônico ou possui tamanho divergente.`);
  }
  const values = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) {
    const value = payload.readFloatLE(index * 4);
    if (!Number.isFinite(value)) throw new Error(`${label}: payload contém valor F32 não finito.`);
    values[index] = value;
  }
  return { shape: [...serialized.shape], values };
}

function serializeInvocationEnvironment(
  request: Gemma4RuntimeReductionTransportRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
): Record<string, number | string> {
  const tower = request.program.tower as unknown as Record<string, unknown>;
  const runtimeDtype = request.program.runtimeDtype;
  const runtimeContract = invocationProgram.environment.runtimeDtype;
  if (runtimeContract.source !== "program.runtimeDtype" || runtimeDtype !== runtimeContract.equals) {
    throw new Error(`${request.operationId}: dtype do ambiente de invocação diverge do programa serializado.`);
  }
  const result: Record<string, number | string> = { runtimeDtype };
  for (const binding of invocationProgram.environment.towerParameters) {
    const value = tower[binding.name];
    if (binding.source !== `program.tower.${binding.name}` || binding.numericDomain !== "safe-integer" ||
      !Number.isSafeInteger(value) || (value as number) < binding.minimumInclusive) {
      throw new Error(`${request.operationId}: parâmetro ${binding.name} não satisfaz o ambiente de invocação serializado.`);
    }
    result[binding.name] = value as number;
  }
  return result;
}

function assertSupportedTensorEncoding(contract: ExecutionProtocol["tensorEncoding"]): void {
  if (!isDeepStrictEqual(contract, gemma4RuntimeReductionExecutionProtocol().tensorEncoding)) {
    throw new Error("Encoding de tensor da redução Gemma 4 não é suportado ou está divergente.");
  }
}

function assertCanonicalJson(value: string, label: string): void {
  let parsed: unknown;
  try { parsed = JSON.parse(value) as unknown; } catch (error) {
    throw new Error(`Transcript ${label} da redução Gemma 4 não contém JSON UTF-8 válido: ${(error as Error).message}`);
  }
  if (JSON.stringify(parsed) !== value) {
    throw new Error(`Transcript ${label} da redução Gemma 4 não usa JSON compacto canônico sem bytes finais.`);
  }
}

function sha256Utf8(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function assertExactObjectFields(value: unknown, expected: readonly string[], location: string): void {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
    JSON.stringify(Object.keys(value)) !== JSON.stringify(expected)) {
    throw new Error(`${location}: campos não correspondem ao protocolo serializado.`);
  }
}

function isCanonicalBase64(value: string): boolean {
  return value.length % 4 === 0 && /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value);
}
