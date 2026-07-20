import assert from "node:assert/strict";
import test from "node:test";
import {
  buildGemma4LiteralCompositeModalitySuiteReport,
  type Gemma4LiteralCompositeModalityEvidence,
} from "../src/gemma4-literal-composite-modality-suite.js";

const runtime = "transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode";
const candidate = "llm-inner embedded-literal Gemma4 composite BF16-policy executor";

test("Gemma 4 modality suite requires zero-tolerance image, video, and audio generation evidence", () => {
  const evidence = [fixture("image", 32), fixture("video", 32), fixture("audio", 36)];
  const report = buildGemma4LiteralCompositeModalitySuiteReport(evidence);
  assert.deepEqual(report.summary, {
    modalityCount: 3,
    zeroTolerancePrefillPasses: 3,
    zeroToleranceGenerationPasses: 3,
    runtimeDefinedReductions: 100,
    firstDivergence: null,
  });
  assert.equal(report.fidelityClaim, "candidate-modality-coverage-with-runtime-defined-reductions");

  const missingVideo = [evidence[0]!, evidence[2]!];
  assert.throws(() => buildGemma4LiteralCompositeModalitySuiteReport(missingVideo), /image, video e audio/);
  const approximate = structuredClone(evidence);
  approximate[1]!.comparison.generation.fidelityClass = "approximate";
  assert.throws(() => buildGemma4LiteralCompositeModalitySuiteReport(approximate), /não passou geração e cache/);
  const inventedScheduleCoverage = structuredClone(evidence);
  inventedScheduleCoverage[2]!.comparison.reductionDomainEvaluation.runtimeDefinedReductions = 35;
  assert.throws(() => buildGemma4LiteralCompositeModalitySuiteReport(inventedScheduleCoverage), /100 BMM runtime-defined/);
  const malformedReplay = structuredClone(evidence);
  malformedReplay[0]!.comparison.runtimeReductionReplay = {
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    executionCount: 0,
    executions: [],
  };
  assert.throws(() => buildGemma4LiteralCompositeModalitySuiteReport(malformedReplay), /replay nativo incompleto/);
});

function fixture(modality: "image" | "video" | "audio", runtimeDefinedReductions: number): Gemma4LiteralCompositeModalityEvidence {
  const operation = (index: number) => ({ operationId: `${modality}-${index}`, output: `value-${index}`, status: "pass" as const });
  return {
    modality,
    trace: `/tmp/${modality}.json`,
    traceSha256: modality.repeat(64).slice(0, 64),
    comparison: {
      modality,
      prefill: {
        candidateRuntime: candidate,
        tolerance: { maxAbsoluteError: 0, maxRelativeError: 0 },
        operations: Array.from({ length: 5 }, (_, index) => operation(index)),
        missingCandidateOperationIds: [],
        firstDivergentOperation: null,
        fidelityClass: "lossless-within-dtype",
      },
      generation: {
        reference: {
          runtime,
          executionMode: "torch.inference_mode",
          attentionImplementation: "eager",
          model: "google/gemma-4-E4B",
          revisionOrChecksum: "411aa17b749aa952df1359d2dcea73917a544d9a",
          containerFormat: "safetensors",
          quantization: "none; dense BF16 storage",
          inputTokens: [1, 2],
          promptPositionIds: [0, 1],
          dtypePolicy: "native eager BF16",
          maxNewTokens: 1,
        },
        candidateRuntime: candidate,
        tolerance: { maxAbsoluteError: 0, maxRelativeError: 0 },
        promptMatches: true,
        generatedTokenIds: [{ index: 0, status: "pass", candidate: { tokenId: 184, positionId: 2 }, reference: { tokenId: 184, positionId: 2 } }],
        terminalLogits: { shape: [1], elementCount: 1, maxAbsoluteError: 0, maxRelativeError: 0, cosineSimilarity: 1, topKOverlap: 1, argmaxAgreement: true, nonFiniteMismatchCount: 0 },
        kvCache: [{ layer: 0, status: "pass" }],
        firstDivergence: null,
        fidelityClass: "lossless-within-dtype",
      },
      reductionDomainEvaluation: {
        activeAssignments: 1,
        reductions: 1,
        stages: 0,
        domains: 1,
        runtimeDefinedReductions,
      },
      runtimeReductionReplay: null,
    },
  };
}
