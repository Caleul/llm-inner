export const GEMMA4_AUDIO_REFERENCE_RUNTIME =
  "transformers-5.5.0/torch-2.12.1-Gemma4Audio-CPU-eager-inference-mode";

export const GEMMA4_VISION_REFERENCE_RUNTIME =
  "transformers-5.5.0/torch-2.12.1-Gemma4Vision-CPU-eager-inference-mode";

export const GEMMA4_COMPOSITE_REFERENCE_RUNTIME =
  "transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode";

export type Gemma4AuthoritativeRuntimeScope = "audio" | "vision" | "composite";

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
 * produced it.  In particular, PyTorch no-grad and inference-mode are not
 * interchangeable around the native Apple Accelerate matmul boundaries.
 */
export function assertGemma4AuthoritativeRuntime(
  scope: Gemma4AuthoritativeRuntimeScope,
  actual: string,
): void {
  const expected = scope === "audio" ? GEMMA4_AUDIO_REFERENCE_RUNTIME
    : scope === "vision" ? GEMMA4_VISION_REFERENCE_RUNTIME
      : GEMMA4_COMPOSITE_REFERENCE_RUNTIME;
  if (actual !== expected) {
    throw new Error(`Trace Gemma 4 ${scope} usa runtime '${actual}', esperado '${expected}'.`);
  }
}
import { isDeepStrictEqual } from "node:util";
