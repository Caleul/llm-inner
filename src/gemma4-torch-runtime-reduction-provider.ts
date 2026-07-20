import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
  GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT,
  validateGemma4RuntimeReductionAdapterProgram,
  type Gemma4RuntimeReductionAdapterProgram,
} from "./gemma4-authoritative-runtime.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import {
  gemma4RuntimeReductionTensorEvidence,
  type Gemma4RuntimeReductionAttestation,
  type Gemma4RuntimeReductionExecution,
  type Gemma4RuntimeReductionProvider,
  type Gemma4RuntimeReductionRequest,
} from "./gemma4-runtime-reduction-provider.js";
import type { DenseF32Tensor } from "./types.js";
import {
  gemma4RuntimeReductionInvocationProgram,
  gemma4RuntimeReductionInvocationProgramSha256,
  validateGemma4RuntimeReductionInvocationPrograms,
  type Gemma4LiteralRuntimeReductionOperationClass,
  type Gemma4RuntimeReductionInvocationProgram,
} from "./gemma4-runtime-reduction-invocation.js";

interface SerializedTensor { shape: number[]; values: number[] }
interface HelperResponse {
  schemaVersion: number;
  contractId: string;
  operationId: string;
  scope: string;
  operation: string;
  invocationProgramId: Gemma4LiteralRuntimeReductionOperationClass;
  sourceCheckpointAccessed: boolean;
  runtimeAttestation: Gemma4RuntimeReductionAttestation;
  output: SerializedTensor;
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
  readonly #adapterProgram: Gemma4RuntimeReductionAdapterProgram;
  readonly #invocationPrograms: readonly Gemma4RuntimeReductionInvocationProgram[];

  constructor(
    python: string,
    adapterProgram: Gemma4RuntimeReductionAdapterProgram,
    invocationPrograms: readonly Gemma4RuntimeReductionInvocationProgram[],
  ) {
    if (!python) throw new Error("Provedor PyTorch Gemma 4 requer executável Python explícito.");
    validateGemma4RuntimeReductionAdapterProgram(adapterProgram);
    this.#python = resolve(python);
    this.#adapterProgram = structuredClone(adapterProgram);
    validateGemma4RuntimeReductionInvocationPrograms(invocationPrograms);
    this.#invocationPrograms = structuredClone(invocationPrograms);
  }

  static async fromArtifact(python: string, artifactPath: string): Promise<Gemma4TorchRuntimeReductionProvider> {
    const artifact = await openGemma4CompositeLiteralArtifact(artifactPath);
    try {
      return new Gemma4TorchRuntimeReductionProvider(
        python,
        artifact.authoritativeExecution.unresolvedNativeReduction.executableReplay.adapterProgram,
        artifact.authoritativeExecution.unresolvedNativeReduction.executableReplay.invocationPrograms,
      );
    } finally {
      await artifact.close();
    }
  }

  get executions(): readonly Gemma4RuntimeReductionExecution["evidence"][] {
    return structuredClone(this.#executions);
  }

  execute(request: Gemma4RuntimeReductionRequest): Gemma4RuntimeReductionExecution {
    const invocationProgram = gemma4RuntimeReductionInvocationProgram(this.#invocationPrograms, request.scope, request.operation);
    const directory = mkdtempSync(join(tmpdir(), "llm-inner-gemma4-runtime-reduction-"));
    try {
      const requestPath = join(directory, "request.json");
      const helperPath = join(directory, "embedded-runtime-reduction.py");
      writeFileSync(requestPath, JSON.stringify(serializeRequest(request, invocationProgram)), "utf8");
      writeFileSync(helperPath, this.#adapterProgram.sourceUtf8, "utf8");
      const child = spawnSync(this.#python, [helperPath, requestPath], {
        encoding: "utf8",
        env: { ...GEMMA4_RUNTIME_REDUCTION_PROCESS_ENVIRONMENT.variables },
        maxBuffer: 128 * 1024 * 1024,
      });
      if (child.error) throw child.error;
      if (child.status !== 0) {
        throw new Error(`${request.operationId}: helper PyTorch encerrou com código ${String(child.status)}: ${child.stderr.trim()}`);
      }
      const response = parseResponse(child.stdout, request, invocationProgram);
      const output = { shape: [...response.output.shape], values: Float32Array.from(response.output.values) };
      const execution: Gemma4RuntimeReductionExecution = {
        output,
        evidence: {
          schemaVersion: 2,
          contractId: this.contractId,
          scope: request.scope,
          operationId: request.operationId,
          operation: request.operation,
          sourceCheckpointAccessed: false,
          adapterProgramSha256: this.#adapterProgram.sha256,
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

function serializeRequest(request: Gemma4RuntimeReductionRequest, invocationProgram: Gemma4RuntimeReductionInvocationProgram): object {
  const tower = serializeInvocationEnvironment(request, invocationProgram);
  return {
    schemaVersion: 1,
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    scope: request.scope,
    operationId: request.operationId,
    operation: request.operation,
    invocationProgram,
    tower,
    operands: request.operands.map(serializeTensor),
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

function serializeTensor(tensor: DenseF32Tensor): SerializedTensor {
  return { shape: [...tensor.shape], values: Array.from(tensor.values) };
}

function parseResponse(
  stdout: string,
  request: Gemma4RuntimeReductionRequest,
  invocationProgram: Gemma4RuntimeReductionInvocationProgram,
): HelperResponse {
  let parsed: HelperResponse;
  try { parsed = JSON.parse(stdout) as HelperResponse; } catch (error) {
    throw new Error(`${request.operationId}: helper PyTorch retornou JSON inválido: ${(error as Error).message}`);
  }
  if (parsed.schemaVersion !== 1 || parsed.contractId !== "torch-2.12.1-cpu-inference-matmul-v1" ||
    parsed.operationId !== request.operationId || parsed.scope !== request.scope || parsed.operation !== request.operation ||
    parsed.invocationProgramId !== invocationProgram.id ||
    parsed.sourceCheckpointAccessed !== false || !parsed.runtimeAttestation || !parsed.output ||
    !Array.isArray(parsed.output.shape) || parsed.output.shape.some((value) => !Number.isSafeInteger(value) || value <= 0) ||
    !Array.isArray(parsed.output.values) || parsed.output.values.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error(`${request.operationId}: helper PyTorch retornou envelope incompatível.`);
  }
  return parsed;
}
