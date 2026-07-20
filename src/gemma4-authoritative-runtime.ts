import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

export const GEMMA4_AUDIO_REFERENCE_RUNTIME =
  "transformers-5.5.0/torch-2.12.1-Gemma4Audio-CPU-eager-inference-mode";

export const GEMMA4_VISION_REFERENCE_RUNTIME =
  "transformers-5.5.0/torch-2.12.1-Gemma4Vision-CPU-eager-inference-mode";

export const GEMMA4_COMPOSITE_REFERENCE_RUNTIME =
  "transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode";

export type Gemma4AuthoritativeRuntimeScope = "audio" | "vision" | "composite";

export interface Gemma4AuthoritativeTraceContext {
  runtime: string;
  executionMode: unknown;
  attentionImplementation: unknown;
}

export interface Gemma4AuthoritativeExecutionContract {
  kind: "gemma4-authoritative-execution-contract";
  schemaVersion: 2;
  canonicalCompositeRuntime: typeof GEMMA4_COMPOSITE_REFERENCE_RUNTIME;
  diagnosticSubprogramRuntimes: {
    audio: typeof GEMMA4_AUDIO_REFERENCE_RUNTIME;
    vision: typeof GEMMA4_VISION_REFERENCE_RUNTIME;
  };
  executionMode: "torch.inference_mode";
  attentionImplementation: "eager";
  unresolvedNativeReduction: {
    provider: "Apple Accelerate SGEMM";
    scalarSchedule: "unpublished-fail-closed";
    operationClasses: [
      "vision-attention-score",
      "vision-attention-value",
      "audio-content-attention-score",
      "audio-position-attention-score",
      "audio-attention-value",
    ];
    executableReplay: {
      providerContractId: "torch-2.12.1-cpu-inference-matmul-v1";
      runtime: "torch-2.12.1";
      torchBuildCommit: "7269437d655783a26cba32aa88195b741ff496aa";
      executionMode: "torch.inference_mode";
      device: "cpu";
      platform: "Darwin-arm64";
      operation: "torch.matmul";
      backend: "Apple Accelerate SGEMM";
      blasBuildSetting: "BLAS_INFO=accelerate";
      checkpointInput: "forbidden";
      operandSource: "dependency-ordered artifact intermediates only";
      adapterProgram: Gemma4RuntimeReductionAdapterProgram;
      operationClasses: [
        { operationClass: "vision-attention-score"; inputDtype: "BF16"; outputDtype: "BF16"; transform: "[B,H,Q,D] @ transpose([B,H,K,D],2,3) -> [B,H,Q,K]" },
        { operationClass: "vision-attention-value"; inputDtype: "BF16"; outputDtype: "BF16"; transform: "transpose([B,H,Q,K] @ [B,H,K,D],1,2) -> reshape [B,Q,H*D]" },
        { operationClass: "audio-content-attention-score"; inputDtype: "F32"; outputDtype: "F32"; transform: "block(Q,[B,H,blocks,chunk,D]) @ transpose(context(K),D,key)" },
        { operationClass: "audio-position-attention-score"; inputDtype: "F32"; outputDtype: "F32"; transform: "reshape(block(Q),[B,H,blocks*chunk,D]) @ transpose(relativeK,H,D,R)" },
        { operationClass: "audio-attention-value"; inputDtype: "F32"; outputDtype: "F32"; transform: "weights[B,H,blocks,chunk,context] @ context(V) -> transpose/reshape/trim [B,S,H*D]" },
      ];
    };
  };
}

export interface Gemma4RuntimeReductionAdapterProgram {
  kind: "gemma4-runtime-reduction-adapter-program";
  schemaVersion: 1;
  language: "python3";
  encoding: "utf8";
  entrypoint: "main";
  bytes: number;
  sha256: typeof GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256;
  sourceUtf8: string;
}

export const GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256 = "6a2d8ab61c3ae2d2b5ca90c908e1d76ff3761615f0f7c662ae63323b0b611674" as const;

/** Loads the build-time adapter once; the emitted artifact embeds it and no longer needs this file. */
export async function loadGemma4RuntimeReductionAdapterProgram(
  helper = resolve(dirname(fileURLToPath(import.meta.url)), "../../helpers/torch_gemma4_runtime_reductions.py"),
): Promise<Gemma4RuntimeReductionAdapterProgram> {
  return buildGemma4RuntimeReductionAdapterProgram(await readFile(helper, "utf8"));
}

export function buildGemma4RuntimeReductionAdapterProgram(sourceUtf8: string): Gemma4RuntimeReductionAdapterProgram {
  const bytes = Buffer.byteLength(sourceUtf8, "utf8");
  const sha256 = createHash("sha256").update(sourceUtf8, "utf8").digest("hex");
  if (sha256 !== GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256) {
    throw new Error(`Adapter de redução Gemma 4 diverge do programa fixado: ${sha256}.`);
  }
  return {
    kind: "gemma4-runtime-reduction-adapter-program",
    schemaVersion: 1,
    language: "python3",
    encoding: "utf8",
    entrypoint: "main",
    bytes,
    sha256: GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
    sourceUtf8,
  };
}

export function validateGemma4RuntimeReductionAdapterProgram(program: Gemma4RuntimeReductionAdapterProgram): void {
  if (!isDeepStrictEqual(program, buildGemma4RuntimeReductionAdapterProgram(program.sourceUtf8))) {
    throw new Error("Programa adapter de redução Gemma 4 está incompleto ou divergente.");
  }
}

export function gemma4AuthoritativeExecutionContract(
  adapterProgram: Gemma4RuntimeReductionAdapterProgram,
): Gemma4AuthoritativeExecutionContract {
  validateGemma4RuntimeReductionAdapterProgram(adapterProgram);
  return {
    kind: "gemma4-authoritative-execution-contract",
    schemaVersion: 2,
    canonicalCompositeRuntime: GEMMA4_COMPOSITE_REFERENCE_RUNTIME,
    diagnosticSubprogramRuntimes: {
      audio: GEMMA4_AUDIO_REFERENCE_RUNTIME,
      vision: GEMMA4_VISION_REFERENCE_RUNTIME,
    },
    executionMode: "torch.inference_mode",
    attentionImplementation: "eager",
    unresolvedNativeReduction: {
      provider: "Apple Accelerate SGEMM",
      scalarSchedule: "unpublished-fail-closed",
      operationClasses: [
        "vision-attention-score",
        "vision-attention-value",
        "audio-content-attention-score",
        "audio-position-attention-score",
        "audio-attention-value",
      ],
      executableReplay: {
        providerContractId: "torch-2.12.1-cpu-inference-matmul-v1",
        runtime: "torch-2.12.1",
        torchBuildCommit: "7269437d655783a26cba32aa88195b741ff496aa",
        executionMode: "torch.inference_mode",
        device: "cpu",
        platform: "Darwin-arm64",
        operation: "torch.matmul",
        backend: "Apple Accelerate SGEMM",
        blasBuildSetting: "BLAS_INFO=accelerate",
        checkpointInput: "forbidden",
        operandSource: "dependency-ordered artifact intermediates only",
        adapterProgram: structuredClone(adapterProgram),
        operationClasses: [
          { operationClass: "vision-attention-score", inputDtype: "BF16", outputDtype: "BF16", transform: "[B,H,Q,D] @ transpose([B,H,K,D],2,3) -> [B,H,Q,K]" },
          { operationClass: "vision-attention-value", inputDtype: "BF16", outputDtype: "BF16", transform: "transpose([B,H,Q,K] @ [B,H,K,D],1,2) -> reshape [B,Q,H*D]" },
          { operationClass: "audio-content-attention-score", inputDtype: "F32", outputDtype: "F32", transform: "block(Q,[B,H,blocks,chunk,D]) @ transpose(context(K),D,key)" },
          { operationClass: "audio-position-attention-score", inputDtype: "F32", outputDtype: "F32", transform: "reshape(block(Q),[B,H,blocks*chunk,D]) @ transpose(relativeK,H,D,R)" },
          { operationClass: "audio-attention-value", inputDtype: "F32", outputDtype: "F32", transform: "weights[B,H,blocks,chunk,context] @ context(V) -> transpose/reshape/trim [B,S,H*D]" },
        ],
      },
    },
  };
}

export function validateGemma4AuthoritativeExecutionContract(
  contract: Gemma4AuthoritativeExecutionContract,
): void {
  const adapter = contract.unresolvedNativeReduction.executableReplay.adapterProgram;
  validateGemma4RuntimeReductionAdapterProgram(adapter);
  const expected = gemma4AuthoritativeExecutionContract(adapter);
  if (!isDeepStrictEqual(contract, expected)) {
    throw new Error("Artefato Gemma 4 não fixa o contrato autoritativo de execução e a fronteira BMM nativa.");
  }
}

/**
 * A trace is numeric evidence only for the exact execution contract which
 * produced it. PyTorch no-grad/inference-mode and SDPA/eager mask construction
 * are not interchangeable around the audio and native matmul boundaries.
 */
export function assertGemma4AuthoritativeRuntime(
  scope: Gemma4AuthoritativeRuntimeScope,
  actual: Gemma4AuthoritativeTraceContext,
): void {
  const expected = scope === "audio" ? GEMMA4_AUDIO_REFERENCE_RUNTIME
    : scope === "vision" ? GEMMA4_VISION_REFERENCE_RUNTIME
      : GEMMA4_COMPOSITE_REFERENCE_RUNTIME;
  if (actual.runtime !== expected || actual.executionMode !== "torch.inference_mode" || actual.attentionImplementation !== "eager") {
    throw new Error(
      `Trace Gemma 4 ${scope} usa contexto runtime='${actual.runtime}', executionMode='${String(actual.executionMode)}', ` +
      `attentionImplementation='${String(actual.attentionImplementation)}'; esperado runtime='${expected}', ` +
      "executionMode='torch.inference_mode', attentionImplementation='eager'.",
    );
  }
}
