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
  type Gemma4RuntimeReductionExecution,
  type Gemma4RuntimeReductionEvidenceContract,
  type Gemma4RuntimeReductionProvider,
  type Gemma4RuntimeReductionRequest,
} from "./gemma4-runtime-reduction-provider.js";
import {
  gemma4RuntimeReductionInvocationProgram,
  gemma4RuntimeReductionInvocationProgramSha256,
} from "./gemma4-runtime-reduction-invocation.js";
import {
  deserializeGemma4RuntimeReductionTensor,
  gemma4RuntimeReductionTranscriptEvidence,
  parseGemma4RuntimeReductionResponseEnvelope,
  serializeGemma4RuntimeReductionRequestEnvelope,
} from "./gemma4-runtime-reduction-transport.js";

export {
  deserializeGemma4RuntimeReductionTensor,
  serializeGemma4RuntimeReductionTensor,
} from "./gemma4-runtime-reduction-transport.js";

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
      const requestEnvelope = serializeGemma4RuntimeReductionRequestEnvelope(
        request,
        invocationProgram,
        this.#replayContract.executionProtocol,
      );
      const requestUtf8 = JSON.stringify(requestEnvelope);
      writeFileSync(requestPath, requestUtf8, "utf8");
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
      const response = parseGemma4RuntimeReductionResponseEnvelope(
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
          schemaVersion: 5,
          contractId: this.contractId,
          scope: request.scope,
          operationId: request.operationId,
          operation: request.operation,
          sourceCheckpointAccessed: false,
          adapterProgramSha256: this.#replayContract.adapterProgram.sha256,
          executionProtocolSha256: gemma4RuntimeReductionExecutionProtocolSha256(this.#replayContract.executionProtocol),
          invocationProgramId: invocationProgram.id,
          invocationProgramSha256: gemma4RuntimeReductionInvocationProgramSha256(invocationProgram),
          executionTranscript: gemma4RuntimeReductionTranscriptEvidence(
            requestUtf8,
            child.stdout,
            this.#replayContract.executionProtocol,
          ),
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
