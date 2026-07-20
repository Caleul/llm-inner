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
  schemaVersion: 7;
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
      runtimeEnvironmentIdentity: typeof GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY;
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

export const GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256 = "6cb4b2d076a08536eabeb4bfaa2a5368de75f88b904aad291fd847e39d725b08" as const;

/**
 * Exact host identity for the still-opaque Apple Accelerate reduction path.
 * Darwin-arm64 alone is not sufficient: Accelerate dispatch can change with
 * the OS build and Apple CPU generation even when the Torch build is fixed.
 */
export const GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY = {
  pythonImplementation: "CPython",
  pythonVersion: "3.14.3",
  operatingSystem: "macOS",
  operatingSystemVersion: "26.5.2",
  operatingSystemBuild: "25F84",
  kernelRelease: "25.5.0",
  machineModel: "Mac15,10",
  cpuBrand: "Apple M3 Max",
  torchBuildConfigSha256: "606e3853213dea3faabc6d58b66ed7e419ee4452a6d53c2b27495a2ecc4e07a7",
  runtimeDependencyIdentity: {
    schemaVersion: 1,
    files: [
      {
        role: "cpython-runtime",
        locator: "sys.base_prefix/Python",
        bytes: 5_438_400,
        sha256: "e5728c35bdc26dee85e45b3fb94780afc1c9f97ced6b0af64d54e4eab3422e0a",
      },
      {
        role: "torch-python-extension",
        locator: "torch._C.__file__",
        bytes: 50_232,
        sha256: "c48ade47e58bf4d28f4f41bd59b5be4b37e4931b36a1a90c44e9d8f5cb6ee434",
      },
      {
        role: "torch-python-library",
        locator: "torch.package/lib/libtorch_python.dylib",
        bytes: 29_929_032,
        sha256: "cb0f00560a29f0ff82cc125013c4fe5dfd544fba9cc728aa922efa56cd047557",
      },
      {
        role: "torch-cpu-kernel-library",
        locator: "torch.package/lib/libtorch_cpu.dylib",
        bytes: 248_507_328,
        sha256: "791f549846676c37c778a6bb043b6fafae54d3d32112a782a388be4ba6e6d52f",
      },
      {
        role: "torch-tensor-runtime-library",
        locator: "torch.package/lib/libc10.dylib",
        bytes: 1_072_704,
        sha256: "935940fedf52ad9d3aa40f1570ec4e6529b6be7d860bf0daaced344927d7e657",
      },
      {
        role: "torch-core-library",
        locator: "torch.package/lib/libtorch.dylib",
        bytes: 16_752,
        sha256: "4eab0bfaef1b14044359cefebb0030f1dcc5ad5f757d514a2aa7fff6dd08b032",
      },
      {
        role: "torch-shared-memory-library",
        locator: "torch.package/lib/libshm.dylib",
        bytes: 64_016,
        sha256: "43e8d43211fdbc270a2a5a6d580ed0344f74f5f15cf34f8abe5da6f958323dfe",
      },
      {
        role: "openmp-runtime-library",
        locator: "torch.package/lib/libomp.dylib",
        bytes: 856_096,
        sha256: "6256bee09e93c28d71c65711cc69224d69994c6965648b628b70a22772fe98d4",
      },
      {
        role: "torch-global-dependencies-library",
        locator: "torch.package/lib/libtorch_global_deps.dylib",
        bytes: 16_760,
        sha256: "63504e19a4f955eb4abe956a3f90578e343b033be75f26aa168aa4991db437fd",
      },
    ],
    pythonSourceTrees: [
      {
        role: "cpython-standard-library",
        locator: "os.__file__/..",
        includeSuffixes: [".py", ".pyi"],
        excludePathParts: ["__pycache__", "site-packages"],
        canonicalLeafEncoding: "relative-posix-path\\0byte-count\\0sha256-hex\\n",
        files: 1_848,
        bytes: 35_754_693,
        sha256: "3179ebdc3d1f5bbb1f3612d64fd0feb137af43688523c8c6eb0ff12eb9b4254d",
      },
      {
        role: "torch-python-package",
        locator: "torch.__file__/..",
        includeSuffixes: [".py", ".pyi"],
        excludePathParts: ["__pycache__"],
        canonicalLeafEncoding: "relative-posix-path\\0byte-count\\0sha256-hex\\n",
        files: 2_230,
        bytes: 46_148_427,
        sha256: "31caad9097d0c18d1ea1067589d973544965f4e91d27f223c865608ffbc9c8e7",
      },
    ],
    sharedCacheImages: [
      {
        role: "accelerate-blas",
        installName: "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib",
        architecture: "arm64e",
        machoUuid: "F078C775-D8DC-3C4D-879F-A9BB228DBE06",
      },
    ],
  },
} as const;

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
    schemaVersion: 7,
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
        runtimeEnvironmentIdentity: structuredClone(GEMMA4_RUNTIME_REDUCTION_ENVIRONMENT_IDENTITY),
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
