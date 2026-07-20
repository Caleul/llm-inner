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
  schemaVersion: 1;
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

export function gemma4AuthoritativeExecutionContract(): Gemma4AuthoritativeExecutionContract {
  return {
    kind: "gemma4-authoritative-execution-contract",
    schemaVersion: 1,
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
  const expected = gemma4AuthoritativeExecutionContract();
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
import { isDeepStrictEqual } from "node:util";
