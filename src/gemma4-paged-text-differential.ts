import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { compareGenerationTrace } from "./differential.js";
import { openGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { generateGemma4PagedTextLiteralF32 } from "./gemma4-paged-text.js";
import { fingerprintIR, readGenerationTraceBundle } from "./trace.js";
import type { DifferentialGenerationComparisonReport } from "./types.js";

export async function compareGemma4PagedTextLiteralGenerationTrace(options: {
  artifact: string;
  trace: string;
  maxReadBytes: number;
  topK?: number;
  maxAbsoluteError?: number;
  maxRelativeError?: number;
  assertSourceUnavailable?: string;
}): Promise<DifferentialGenerationComparisonReport> {
  if (!Number.isSafeInteger(options.maxReadBytes) || options.maxReadBytes <= 0) throw new Error("Comparação Gemma 4 paginada requer maxReadBytes positivo seguro.");
  if (options.assertSourceUnavailable) {
    let sourceExists = true;
    try { await access(options.assertSourceUnavailable, constants.F_OK); }
    catch { sourceExists = false; }
    if (sourceExists) throw new Error(`Comparação Gemma 4 requer source indisponível, mas '${options.assertSourceUnavailable}' ainda existe.`);
  }
  const decoded = await readGenerationTraceBundle(options.trace);
  if (decoded.bundle.candidatePolicy.dtype !== "F32" || decoded.bundle.candidatePolicy.runtime !== "llm-inner paged Gemma4Text literal F32") {
    throw new Error("Trace não declara o candidato paginado Gemma4Text F32 esperado.");
  }
  const artifact = await openGemma4CompositeLiteralArtifact(options.artifact);
  try {
    if (fingerprintIR(artifact.program.textProgram) !== decoded.bundle.irFingerprint) throw new Error("Trace Gemma 4 não corresponde ao programa textual embutido no artefato literal.");
    const reference = decoded.reference;
    const candidate = await generateGemma4PagedTextLiteralF32(artifact, {
      inputIds: [reference.inputTokens], positionIds: [reference.promptPositionIds], maxNewTokens: reference.maxNewTokens,
      ...(reference.eosTokenId === undefined ? {} : { eosTokenId: reference.eosTokenId }),
    }, { maxReadBytes: options.maxReadBytes });
    return compareGenerationTrace(candidate, reference, {
      candidateRuntime: decoded.bundle.candidatePolicy.runtime,
      ...(options.topK === undefined ? {} : { topK: options.topK }),
      tolerance: { maxAbsoluteError: options.maxAbsoluteError ?? 0, maxRelativeError: options.maxRelativeError ?? 0 },
    });
  } finally {
    await artifact.close();
  }
}
