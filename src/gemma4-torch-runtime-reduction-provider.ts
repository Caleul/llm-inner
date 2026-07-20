import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";
import {
  gemma4RuntimeReductionExecutionProtocol,
  validateGemma4RuntimeReductionExecutableReplayContract,
  type Gemma4RuntimeReductionExecutableReplayContract,
} from "./gemma4-authoritative-runtime.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  gemma4RuntimeReductionTensorEvidence,
  gemma4RuntimeReductionEvidenceContract,
  gemma4RuntimeReductionExecutionProtocolSha256,
  type Gemma4RuntimeReductionAttestation,
  type Gemma4RuntimeReductionExecution,
  type Gemma4RuntimeReductionEvidenceContract,
  type Gemma4RuntimeReductionProvider,
  type Gemma4RuntimeReductionRequest,
} from "./gemma4-runtime-reduction-provider.js";
import type { DenseF32Tensor } from "./types.js";
import {
  gemma4RuntimeReductionInvocationProgram,
  gemma4RuntimeReductionInvocationProgramSha256,
  type Gemma4LiteralRuntimeReductionOperationClass,
  type Gemma4RuntimeReductionInvocationProgram,
} from "./gemma4-runtime-reduction-invocation.js";

export interface Gemma4RuntimeReductionSerializedTensor {
  dtype: "F32";
  bitPattern: "IEEE-754 binary32";
  byteOrder: "little-endian";
  layout: "row-major-contiguous";
  shape: number[];
  byteLength: number;
  dataBase64: string;
}
interface HelperResponse {
  schemaVersion: number;
  contractId: string;
  operationId: string;
  scope: string;
  operation: string;
  invocationProgramId: Gemma4LiteralRuntimeReductionOperationClass;
  sourceCheckpointAccessed: boolean;
  runtimeAttestation: Gemma4RuntimeReductionAttestation;
  output: Gemma4RuntimeReductionSerializedTensor;
}

/**
 * Source-independent adapter for the exact pinned PyTorch CPU matmul path.
 * Each request contains only already-produced literal intermediates and the
 * structured tower contract; the helper has no checkpoint argument or model
 * loading capability.
 */
export class Gemma4TorchRuntimeReductionProvider implements Gemma4RuntimeReductionProvider {
  readonly contractId = "torch-2.12.1-cpu-inference-matmul-v1" as const;
  readonly #executions: Gemma4RuntimeReductionExecution["evidence"][] = [];
  readonly #python: string;
  readonly #replayContract: Gemma4RuntimeReductionExecutableReplayContract;
  readonly #evidenceContract: Gemma4RuntimeReductionEvidenceContract;

  constructor(
    python: string,
    replayContract: Gemma4RuntimeReductionExecutableReplayContract,
  ) {
    if (!python) throw new Error("Provedor PyTorch Gemma 4 requer executável Python explícito.");
    validateGemma4RuntimeReductionExecutableReplayContract(replayContract);
    this.#python = resolve(python);
    this.#replayContract = structuredClone(replayContract);
    this.#evidenceContract = gemma4RuntimeReductionEvidenceContract(this.#replayContract);
  }

  static async fromArtifact(python: string, artifactPath: string): Promise<Gemma4TorchRuntimeReductionProvider> {
    const artifact = await openGemma4CompositeLiteralArtifact(artifactPath);
    try {
      return new Gemma4TorchRuntimeReductionProvider(
        python,
        artifact.authoritativeExecution.unresolvedNativeReduction.executableReplay,
      );
    } finally {
      await artifact.close();
    }
  }

  get executions(): readonly Gemma4RuntimeReductionExecution["evidence"][] {
    return structuredClone(this.#executions);
  }

  get evidenceContract(): Gemma4RuntimeReductionEvidenceContract {
    return structuredClone(this.#evidenceContract);
  }

  execute(request: Gemma4RuntimeReductionRequest): Gemma4RuntimeReductionExecution {
    const invocationProgram = gemma4RuntimeReductionInvocationProgram(
      this.#replayContract.invocationPrograms,
      request.scope,
      request.operation,
    );
    const launch = runtimeReductionLaunchContract(this.#replayContract);
    const directory = mkdtempSync(join(tmpdir(), launch.temporaryDirectoryPrefix));
    try {
      const requestPath = join(directory, launch.requestFileName);
      const helperPath = join(directory, launch.adapterFileName);
      const requestEnvelope = serializeRequest(request, invocationProgram, this.#replayContract.executionProtocol.tensorEncoding);
      assertExactObjectFields(requestEnvelope, this.#replayContract.executionProtocol.requestEnvelope.fields, `${request.operationId}: request envelope`);
      assertTensorFields(requestEnvelope, this.#replayContract.executionProtocol.tensorEncoding.fields, `${request.operationId}: request envelope`);
      writeFileSync(requestPath, JSON.stringify(requestEnvelope), "utf8");
      writeFileSync(helperPath, this.#replayContract.adapterProgram.sourceUtf8, "utf8");
      const paths = { "adapter-file": helperPath, "request-file": requestPath } as const;
      const child = spawnSync(this.#python, launch.arguments.map((argument) => paths[argument]), {
        encoding: "utf8",
        cwd: directory,
        env: runtimeReductionSpawnEnvironment(this.#replayContract),
        input: "",
        maxBuffer: launch.maxOutputBytes,
      });
      if (child.error) throw child.error;
      if (child.status !== 0) {
        throw new Error(`${request.operationId}: helper PyTorch encerrou com código ${String(child.status)}: ${child.stderr.trim()}`);
      }
      const response = parseResponse(
        child.stdout,
        request,
        invocationProgram,
        this.#replayContract.executionProtocol,
      );
      const output = deserializeGemma4RuntimeReductionTensor(
        response.output,
        this.#replayContract.executionProtocol.tensorEncoding,
        `${request.operationId}: response output`,
      );
      const execution: Gemma4RuntimeReductionExecution = {
        output,
        evidence: {
          schemaVersion: 4,
          contractId: this.contractId,
          scope: request.scope,
          operationId: request.operationId,
          operation: request.operation,
          sourceCheckpointAccessed: false,
          adapterProgramSha256: this.#replayContract.adapterProgram.sha256,
          executionProtocolSha256: gemma4RuntimeReductionExecutionProtocolSha256(this.#replayContract.executionProtocol),
          invocationProgramId: invocationProgram.id,
          invocationProgramSha256: gemma4RuntimeReductionInvocationProgramSha256(invocationProgram),
          runtimeAttestation: response.runtimeAttestation,
          orderedOperands: [
            gemma4RuntimeReductionTensorEvidence(request.operands[0]),
            gemma4RuntimeReductionTensorEvidence(request.operands[1]),
          ],
          output: gemma4RuntimeReductionTensorEvidence(output),
        },
      };
      this.#executions.push(structuredClone(execution.evidence));
      return execution;
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  }
}

interface RuntimeReductionLaunchContract {
  temporaryDirectoryPrefix: string;
  adapterFileName: string;
  requestFileName: string;
  arguments: Array<"adapter-file" | "request-file">;
  maxOutputBytes: number;
}

/** Materializes only the launch semantics serialized by the artifact. */
export function runtimeReductionLaunchContract(
  replayContract: Pick<Gemma4RuntimeReductionExecutableReplayContract, "executionProtocol">,
): RuntimeReductionLaunchContract {
  const protocol = replayContract.executionProtocol;
  if (!isDeepStrictEqual(protocol, gemma4RuntimeReductionExecutionProtocol())) {
    throw new Error("Protocolo de execução da redução Gemma 4 não é suportado ou está divergente.");
  }
  const adapter = protocol.files.find((file) => file.role === "adapter");
  const request = protocol.files.find((file) => file.role === "request");
  if (!adapter || !request) throw new Error("Protocolo de execução da redução Gemma 4 não declara arquivos adapter/request.");
  return {
    temporaryDirectoryPrefix: protocol.temporaryDirectory.prefix,
    adapterFileName: adapter.name,
    requestFileName: request.name,
    arguments: [...protocol.invocation.arguments],
    maxOutputBytes: protocol.invocation.maxOutputBytes,
  };
}

/** Materializes only the closed environment serialized by the artifact. */
export function runtimeReductionSpawnEnvironment(
  replayContract: Pick<Gemma4RuntimeReductionExecutableReplayContract, "runtimeProcessEnvironment">,
): Record<string, string> {
  const environment = replayContract.runtimeProcessEnvironment;
  if (environment.schemaVersion !== 1 || environment.inheritance !== "none") {
    throw new Error("Ambiente de processo da redução Gemma 4 não é um mapa fechado suportado.");
  }
  return { ...environment.variables };
}

function serializeRequest(
  request: Gemma4RuntimeReductionRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
  tensorEncoding: Gemma4RuntimeReductionExecutableReplayContract["executionProtocol"]["tensorEncoding"],
): object {
  const tower = serializeInvocationEnvironment(request, invocationProgram);
  return {
    schemaVersion: 2,
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    scope: request.scope,
    operationId: request.operationId,
    operation: request.operation,
    invocationProgram,
    tower,
    operands: request.operands.map((tensor, index) =>
      serializeGemma4RuntimeReductionTensor(tensor, tensorEncoding, `${request.operationId}: operand ${index}`)),
  };
}

function serializeInvocationEnvironment(
  request: Gemma4RuntimeReductionRequest,
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

export function serializeGemma4RuntimeReductionTensor(
  tensor: DenseF32Tensor,
  contract: Gemma4RuntimeReductionExecutableReplayContract["executionProtocol"]["tensorEncoding"] =
    gemma4RuntimeReductionExecutionProtocol().tensorEncoding,
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
  contract: Gemma4RuntimeReductionExecutableReplayContract["executionProtocol"]["tensorEncoding"] =
    gemma4RuntimeReductionExecutionProtocol().tensorEncoding,
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

function assertSupportedTensorEncoding(
  contract: Gemma4RuntimeReductionExecutableReplayContract["executionProtocol"]["tensorEncoding"],
): void {
  if (!isDeepStrictEqual(contract, gemma4RuntimeReductionExecutionProtocol().tensorEncoding)) {
    throw new Error("Encoding de tensor da redução Gemma 4 não é suportado ou está divergente.");
  }
}

function isCanonicalBase64(value: string): boolean {
  return value.length % 4 === 0 && /^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value);
}

function parseResponse(
  stdout: string,
  request: Gemma4RuntimeReductionRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
  protocol: Gemma4RuntimeReductionExecutableReplayContract["executionProtocol"],
): HelperResponse {
  let parsed: HelperResponse;
  try { parsed = JSON.parse(stdout) as HelperResponse; } catch (error) {
    throw new Error(`${request.operationId}: helper PyTorch retornou JSON inválido: ${(error as Error).message}`);
  }
  assertExactObjectFields(parsed, protocol.responseEnvelope.fields, `${request.operationId}: response envelope`);
  assertExactObjectFields(parsed.output, protocol.tensorEncoding.fields, `${request.operationId}: response output`);
  if (parsed.schemaVersion !== 2 || parsed.contractId !== "torch-2.12.1-cpu-inference-matmul-v1" ||
    parsed.operationId !== request.operationId || parsed.scope !== request.scope || parsed.operation !== request.operation ||
    parsed.invocationProgramId !== invocationProgram.id ||
    parsed.sourceCheckpointAccessed !== false || !parsed.runtimeAttestation || !parsed.output) {
    throw new Error(`${request.operationId}: helper PyTorch retornou envelope incompatível.`);
  }
  return parsed;
}

function assertTensorFields(
  envelope: object,
  expected: readonly string[],
  location: string,
): void {
  const operands = (envelope as { operands?: unknown }).operands;
  if (!Array.isArray(operands) || operands.length !== 2) throw new Error(`${location}: operands inválidos.`);
  for (const [index, operand] of operands.entries()) {
    assertExactObjectFields(operand, expected, `${location}: operand ${index}`);
  }
}

function assertExactObjectFields(value: unknown, expected: readonly string[], location: string): void {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
    JSON.stringify(Object.keys(value)) !== JSON.stringify(expected)) {
    throw new Error(`${location}: campos não correspondem ao protocolo serializado.`);
  }
}
