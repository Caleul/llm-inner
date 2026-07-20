import assert from "node:assert/strict";
import test from "node:test";
import {
  buildGemma4RuntimeReductionTraceCoverage,
  validateGemma4RuntimeReductionTraceCoverage,
} from "../src/gemma4-runtime-reduction-trace-coverage.js";
import type { Gemma4VisionProgram } from "../src/gemma4-vision.js";
import type { DifferentialOperationSample } from "../src/types.js";

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

function fixtureProgram(): Gemma4VisionProgram {
  return {
    kind: "gemma4-vision-features",
    sourceFormat: "safetensors",
    tower: {} as Gemma4VisionProgram["tower"],
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
  return { operationId, output, tensor: { shape, values: new Float32Array(shape.reduce((total, dimension) => total * dimension, 1)) } };
}
