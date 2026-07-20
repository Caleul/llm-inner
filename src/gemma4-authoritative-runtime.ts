import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import {
  gemma4RuntimeReductionInvocationPrograms,
  validateGemma4RuntimeReductionInvocationPrograms,
  type Gemma4RuntimeReductionInvocationProgram,
} from "./gemma4-runtime-reduction-invocation.js";

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
  schemaVersion: 3;
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
      invocationPrograms: Gemma4RuntimeReductionInvocationProgram[];
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

export const GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256 = "29537b870ca3e28caaec2e03456227a25027822cc6f1ae90a45bfc9769b08594" as const;

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
    schemaVersion: 3,
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
        invocationPrograms: gemma4RuntimeReductionInvocationPrograms(),
      },
    },
  };
}

export function validateGemma4AuthoritativeExecutionContract(
  contract: Gemma4AuthoritativeExecutionContract,
): void {
  const adapter = contract.unresolvedNativeReduction.executableReplay.adapterProgram;
  validateGemma4RuntimeReductionAdapterProgram(adapter);
  validateGemma4RuntimeReductionInvocationPrograms(contract.unresolvedNativeReduction.executableReplay.invocationPrograms);
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
