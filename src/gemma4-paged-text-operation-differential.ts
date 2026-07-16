import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { compareCapturedOperationCheckpoints } from "./differential.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { executeGemma4PagedTextLiteralF32 } from "./gemma4-paged-text.js";
import { fingerprintIR, readExecutionTraceBundle } from "./trace.js";
import type { DifferentialCheckpointComparisonReport } from "./types.js";

/** Compare a source-independent literal prefill against native module checkpoints. */
export async function compareGemma4PagedTextLiteralOperationCheckpoints(options: {
  artifact: string;
  trace: string;
  maxReadBytes: number;
  topK?: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
  assertSourceUnavailable?: string;
}): Promise<DifferentialCheckpointComparisonReport> {
  if (!Number.isSafeInteger(options.maxReadBytes) || options.maxReadBytes <= 0) throw new Error("Comparação de checkpoints Gemma 4 requer maxReadBytes positivo seguro.");
  if (options.assertSourceUnavailable) {
    let exists = true;
    try { await access(options.assertSourceUnavailable, constants.F_OK); } catch { exists = false; }
    if (exists) throw new Error(`Comparação de checkpoints Gemma 4 requer source indisponível, mas '${options.assertSourceUnavailable}' ainda existe.`);
  }
  const decoded = await readExecutionTraceBundle(options.trace);
  if (decoded.bundle.candidatePolicy.dtype !== "F32" || decoded.bundle.candidatePolicy.runtime !== "llm-inner paged Gemma4Text literal F32") {
    throw new Error("Trace de checkpoints não declara o candidato paginado Gemma4Text F32 esperado.");
  }
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    if (fingerprintIR(artifact.program.textProgram) !== decoded.bundle.irFingerprint) throw new Error("Trace de checkpoints Gemma 4 não corresponde ao programa textual embutido no artefato literal.");
    const candidate = await executeGemma4PagedTextLiteralF32(artifact, {
      inputIds: decoded.reference.inputTokens,
      ...(decoded.reference.positionIds ? { positionIds: decoded.reference.positionIds } : {}),
    }, { maxReadBytes: options.maxReadBytes });
    return compareCapturedOperationCheckpoints(candidate.values, decoded.reference, {
      candidateRuntime: decoded.bundle.candidatePolicy.runtime,
      ...(options.topK === undefined ? {} : { topK: options.topK }),
      tolerance: { maxAbsoluteError: options.maxAbsoluteError ?? 0, maxRelativeError: options.maxRelativeError ?? 0 },
    });
  } finally {
    await artifact.close();
  }
}
