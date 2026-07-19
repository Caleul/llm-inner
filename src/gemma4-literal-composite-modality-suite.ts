import { sha256File } from "./trace.js";
import {
  compareGemma4LiteralCompositeTrace,
  type Gemma4LiteralCompositeDifferentialReport,
} from "./gemma4-literal-composite-differential.js";
import type { Gemma4CompositeTraceModality } from "./gemma4-composite-trace-profile.js";

const MODALITIES = ["image", "video", "audio"] as const;

export interface Gemma4LiteralCompositeModalityEvidence {
  modality: Gemma4CompositeTraceModality;
  trace: string;
  traceSha256: string;
  comparison: Gemma4LiteralCompositeDifferentialReport;
}

export interface Gemma4LiteralCompositeModalitySuiteReport {
  kind: "gemma4-embedded-literal-composite-modality-suite";
  schemaVersion: 1;
  sourceCheckpointAccessed: false;
  candidateFidelityAcknowledged: true;
  fidelityClaim: "candidate-modality-coverage-with-runtime-defined-reductions";
  model: string;
  revisionOrChecksum: string;
  referenceRuntime: string;
  candidateRuntime: string;
  modalities: Gemma4LiteralCompositeModalityEvidence[];
  summary: {
    modalityCount: 3;
    zeroTolerancePrefillPasses: 3;
    zeroToleranceGenerationPasses: 3;
    runtimeDefinedReductions: number;
    firstDivergence: null;
  };
}

/**
 * Runs one fail-closed source-removed differential boundary over every
 * registered Gemma 4 modality. The suite deliberately preserves the native
 * BMM acknowledgement: zero-tolerance tensor agreement does not publish the
 * missing Apple Accelerate scalar schedule or upgrade checkpoint fidelity.
 */
export async function compareGemma4LiteralCompositeModalitySuite(options: {
  artifact: string;
  traces: Record<Gemma4CompositeTraceModality, string>;
  maxReadBytes: number;
  maxTowerTensorBytes: number;
  maxAbsoluteError: 0;
  maxRelativeError: 0;
  topK: number;
  assertSourceUnavailable: string;
  allowUnverifiedFidelity: true;
}): Promise<Gemma4LiteralCompositeModalitySuiteReport> {
  if (options.maxAbsoluteError !== 0 || options.maxRelativeError !== 0) {
    throw new Error("Suite de modalidades Gemma 4 requer tolerância absoluta e relativa zero.");
  }
  if (options.allowUnverifiedFidelity !== true) {
    throw new Error("Suite de modalidades Gemma 4 requer reconhecimento explícito das reduções runtime-defined.");
  }
  const modalities: Gemma4LiteralCompositeModalityEvidence[] = [];
  for (const modality of MODALITIES) {
    const trace = options.traces[modality];
    const comparison = await compareGemma4LiteralCompositeTrace({
      artifact: options.artifact,
      trace,
      maxReadBytes: options.maxReadBytes,
      maxTowerTensorBytes: options.maxTowerTensorBytes,
      maxAbsoluteError: 0,
      maxRelativeError: 0,
      topK: options.topK,
      assertSourceUnavailable: options.assertSourceUnavailable,
      allowUnverifiedFidelity: true,
    });
    modalities.push({ modality, trace, traceSha256: await sha256File(trace), comparison });
  }
  return buildGemma4LiteralCompositeModalitySuiteReport(modalities);
}

/** Builds and validates the aggregate separately so malformed evidence is unit-testable. */
export function buildGemma4LiteralCompositeModalitySuiteReport(
  modalities: Gemma4LiteralCompositeModalityEvidence[],
): Gemma4LiteralCompositeModalitySuiteReport {
  if (modalities.length !== MODALITIES.length || modalities.some((entry, index) => entry.modality !== MODALITIES[index])) {
    throw new Error("Suite Gemma 4 requer image, video e audio exatamente uma vez e nessa ordem.");
  }
  const identities = modalities.map(({ comparison }) => comparison.generation.reference);
  const expected = identities[0]!;
  for (const [index, entry] of modalities.entries()) {
    const { comparison } = entry;
    if (comparison.modality !== entry.modality) {
      throw new Error(`Suite Gemma 4 recebeu trace ${entry.modality} identificado como ${comparison.modality}.`);
    }
    if (comparison.prefill.tolerance.maxAbsoluteError !== 0 || comparison.prefill.tolerance.maxRelativeError !== 0 ||
      comparison.generation.tolerance.maxAbsoluteError !== 0 || comparison.generation.tolerance.maxRelativeError !== 0) {
      throw new Error(`Suite Gemma 4 ${entry.modality} não foi comparada a tolerância zero.`);
    }
    if (comparison.prefill.fidelityClass !== "lossless-within-dtype" || comparison.prefill.operations.length !== 5 ||
      comparison.prefill.operations.some((operation) => operation.status !== "pass") ||
      comparison.prefill.firstDivergentOperation !== null || comparison.prefill.missingCandidateOperationIds.length !== 0) {
      throw new Error(`Suite Gemma 4 ${entry.modality} não passou todas as fronteiras de prefill.`);
    }
    if (comparison.generation.fidelityClass !== "lossless-within-dtype" ||
      comparison.generation.firstDivergence !== null || !comparison.generation.promptMatches ||
      comparison.generation.generatedTokenIds.length === 0 ||
      comparison.generation.generatedTokenIds.some((step) => step.status !== "pass") ||
      comparison.generation.kvCache.some((cache) => cache.status !== "pass") || comparison.generation.terminalLogits === null) {
      throw new Error(`Suite Gemma 4 ${entry.modality} não passou geração e cache.`);
    }
    if (comparison.reductionDomainEvaluation.runtimeDefinedReductions <= 0) {
      throw new Error(`Suite Gemma 4 ${entry.modality} não registrou sua fronteira BMM runtime-defined.`);
    }
    const identity = identities[index]!;
    if (identity.model !== expected.model || identity.revisionOrChecksum !== expected.revisionOrChecksum ||
      identity.runtime !== expected.runtime || comparison.generation.candidateRuntime !== modalities[0]!.comparison.generation.candidateRuntime) {
      throw new Error("Suite Gemma 4 mistura identidade, revisão ou runtime entre modalidades.");
    }
  }
  const runtimeDefinedReductions = modalities.reduce((total, entry) =>
    total + entry.comparison.reductionDomainEvaluation.runtimeDefinedReductions, 0);
  if (runtimeDefinedReductions !== 100) {
    throw new Error(`Suite Gemma 4 esperava cobrir as 100 BMM runtime-defined, encontrou ${runtimeDefinedReductions}.`);
  }
  return {
    kind: "gemma4-embedded-literal-composite-modality-suite",
    schemaVersion: 1,
    sourceCheckpointAccessed: false,
    candidateFidelityAcknowledged: true,
    fidelityClaim: "candidate-modality-coverage-with-runtime-defined-reductions",
    model: expected.model,
    revisionOrChecksum: expected.revisionOrChecksum,
    referenceRuntime: expected.runtime,
    candidateRuntime: modalities[0]!.comparison.generation.candidateRuntime,
    modalities: structuredClone(modalities),
    summary: {
      modalityCount: 3,
      zeroTolerancePrefillPasses: 3,
      zeroToleranceGenerationPasses: 3,
      runtimeDefinedReductions,
      firstDivergence: null,
    },
  };
}
