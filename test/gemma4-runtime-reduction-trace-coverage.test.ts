import assert from "node:assert/strict";
import test from "node:test";
import {
  buildGemma4RuntimeReductionTraceCoverage,
  validateGemma4RuntimeReductionTraceCoverage,
} from "../src/gemma4-runtime-reduction-trace-coverage.js";
import {
  executeGemma4RuntimeReduction,
  expectedGemma4RuntimeReductionAttestation,
  gemma4RuntimeReductionTensorEvidence,
  type Gemma4RuntimeReductionExecution,
  type Gemma4RuntimeReductionProvider,
  type Gemma4RuntimeReductionRequest,
} from "../src/gemma4-runtime-reduction-provider.js";
import type { Gemma4VisionProgram } from "../src/gemma4-vision.js";
import type { DifferentialOperationSample } from "../src/types.js";
import {
  GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
  loadGemma4RuntimeReductionAdapterProgram,
} from "../src/gemma4-authoritative-runtime.js";
import { Gemma4TorchRuntimeReductionProvider } from "../src/gemma4-torch-runtime-reduction-provider.js";

test("Gemma 4 native-reduction coverage binds both exact operands and native output in program order", () => {
  const program = fixtureProgram();
  const operations = fixtureOperations();
  const coverage = buildGemma4RuntimeReductionTraceCoverage(program, operations);

  assert.deepEqual(coverage.entries, [{
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul",
    output: "vision_layer_0_attention_scores",
    outputShape: [1, 1, 2, 2],
    orderedOperands: [
      { input: "vision_layer_0_q_rotated", producerOperationId: "vision_layer_0_q_rope", shape: [1, 1, 2, 2] },
      { input: "vision_layer_0_k_rotated", producerOperationId: "vision_layer_0_k_rope", shape: [1, 1, 2, 2] },
    ],
  }]);
  validateGemma4RuntimeReductionTraceCoverage(coverage, program, operations);
});

test("Gemma 4 native-reduction coverage fails closed for an earlier or tampered operand checkpoint", () => {
  const program = fixtureProgram();
  const missingPostRope = fixtureOperations().filter((sample) => sample.operationId !== "vision_layer_0_q_rope");
  missingPostRope.push(sample("vision_layer_0_q_norm", "vision_layer_0_q_normalized", [1, 1, 2, 2]));
  assert.throws(
    () => buildGemma4RuntimeReductionTraceCoverage(program, missingPostRope),
    /não contém o operando nativo vision_layer_0_q_rotated/,
  );

  const operations = fixtureOperations();
  const coverage = buildGemma4RuntimeReductionTraceCoverage(program, operations);
  coverage.entries[0]!.orderedOperands[0]!.shape[3] = 3;
  assert.throws(
    () => validateGemma4RuntimeReductionTraceCoverage(coverage, program, operations),
    /cobertura de operandos BMM incompleta ou divergente/,
  );
});

test("Gemma 4 runtime-reduction provider accepts only its pinned contract and exact output shape", () => {
  const request = {
    scope: "vision" as const,
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul" as const,
    program: fixtureProgram(),
    operands: [sampleTensor([1, 1, 2, 2]), sampleTensor([1, 1, 2, 2])] as const,
  };
  const provider: Gemma4RuntimeReductionProvider = {
    contractId: "torch-2.12.1-cpu-inference-matmul-v1",
    executions: [],
    execute(actual) {
      assert.equal(actual, request);
      return execution(actual, sampleTensor([1, 1, 2, 2], 3.5));
    },
  };
  assert.deepEqual(executeGemma4RuntimeReduction(provider, request, [1, 1, 2, 2]), sampleTensor([1, 1, 2, 2], 3.5));

  const wrongShape: Gemma4RuntimeReductionProvider = { ...provider, execute: (actual) => execution(actual, sampleTensor([1, 1, 1, 2])) };
  assert.throws(() => executeGemma4RuntimeReduction(wrongShape, request, [1, 1, 2, 2]), /shape divergente/);
  const nonFinite: Gemma4RuntimeReductionProvider = { ...provider, execute: (actual) => execution(actual, { shape: [1, 1, 2, 2], values: Float32Array.of(0, 0, 0, Infinity) }) };
  assert.throws(() => executeGemma4RuntimeReduction(nonFinite, request, [1, 1, 2, 2]), /tensor inválido/);
  const wrongContract = { ...provider, contractId: "wrong" } as unknown as Gemma4RuntimeReductionProvider;
  assert.throws(() => executeGemma4RuntimeReduction(wrongContract, request, [1, 1, 2, 2]), /não corresponde ao contrato/);

  const wrongAttestation: Gemma4RuntimeReductionProvider = {
    ...provider,
    execute(actual) {
      const result = execution(actual, sampleTensor([1, 1, 2, 2]));
      result.evidence.runtimeAttestation.platform = "Linux-x86_64" as "Darwin-arm64";
      return result;
    },
  };
  assert.throws(() => executeGemma4RuntimeReduction(wrongAttestation, request, [1, 1, 2, 2]), /evidência.*divergente/);

  const mismatchedProgram = fixtureProgram();
  mismatchedProgram.assignments[2]!.operation = "add";
  assert.throws(() => executeGemma4RuntimeReduction(provider, { ...request, program: mismatchedProgram }, [1, 1, 2, 2]), /não corresponde.*BMM/);
});

test("Gemma 4 runtime-reduction provider executes the integrity-bound embedded adapter", async () => {
  const request = {
    scope: "vision" as const,
    operationId: "vision_layer_0_attention_scores",
    operation: "attention-score-matmul" as const,
    program: fixtureProgram(),
    operands: [sampleTensor([1, 1, 2, 2]), sampleTensor([1, 1, 2, 2])] as const,
  };
  const provider = new Gemma4TorchRuntimeReductionProvider(
    "venv/bin/python",
    await loadGemma4RuntimeReductionAdapterProgram(),
  );
  assert.deepEqual(executeGemma4RuntimeReduction(provider, request, [1, 1, 2, 2]), sampleTensor([1, 1, 2, 2]));
  assert.equal(provider.executions.length, 1);
  assert.equal(provider.executions[0]!.adapterProgramSha256, GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256);
});

function fixtureProgram(): Gemma4VisionProgram {
  return {
    kind: "gemma4-vision-features",
    sourceFormat: "safetensors",
    tower: { attentionHeads: 1, headDim: 2 } as Gemma4VisionProgram["tower"],
    textHiddenSize: 2,
    rmsNormEpsilon: 1e-6,
    runtimeDtype: "BF16",
    assignments: [
      { id: "vision_layer_0_q_rope", operation: "multidimensional-rope", inputs: ["q"], output: "vision_layer_0_q_rotated" },
      { id: "vision_layer_0_k_rope", operation: "multidimensional-rope", inputs: ["k"], output: "vision_layer_0_k_rotated" },
      {
        id: "vision_layer_0_attention_scores",
        operation: "attention-score-matmul",
        inputs: ["vision_layer_0_q_rotated", "vision_layer_0_k_rotated"],
        output: "vision_layer_0_attention_scores",
        dtypePolicy: { inputDtype: "BF16", computeDtype: "pytorch-native-batched-matmul", accumulationDtype: "runtime-defined", outputDtype: "BF16" },
      },
    ],
    output: "image_features",
  };
}

function fixtureOperations(): DifferentialOperationSample[] {
  return [
    sample("vision_layer_0_q_rope", "vision_layer_0_q_rotated", [1, 1, 2, 2]),
    sample("vision_layer_0_k_rope", "vision_layer_0_k_rotated", [1, 1, 2, 2]),
    sample("vision_layer_0_attention_scores", "vision_layer_0_attention_scores", [1, 1, 2, 2]),
  ];
}

function sample(operationId: string, output: string, shape: number[]): DifferentialOperationSample {
  return { operationId, output, tensor: sampleTensor(shape) };
}

function sampleTensor(shape: number[], value = 0) {
  return { shape, values: new Float32Array(shape.reduce((total, dimension) => total * dimension, 1)).fill(value) };
}

function execution(request: Gemma4RuntimeReductionRequest, output: ReturnType<typeof sampleTensor>): Gemma4RuntimeReductionExecution {
  return {
    output,
    evidence: {
      schemaVersion: 1,
      contractId: "torch-2.12.1-cpu-inference-matmul-v1",
      scope: request.scope,
      operationId: request.operationId,
      operation: request.operation,
      sourceCheckpointAccessed: false,
      adapterProgramSha256: GEMMA4_RUNTIME_REDUCTION_ADAPTER_SHA256,
      runtimeAttestation: expectedGemma4RuntimeReductionAttestation(),
      orderedOperands: [
        gemma4RuntimeReductionTensorEvidence(request.operands[0]),
        gemma4RuntimeReductionTensorEvidence(request.operands[1]),
      ],
      output: gemma4RuntimeReductionTensorEvidence(output),
    },
  };
}
