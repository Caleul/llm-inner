import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { buildModelIR } from "../src/architecture.js";
import { compareExecutionTrace, compareGenerationTrace } from "../src/differential.js";
import { activationF32, elementwiseF32, executeReferenceF32, executeReferenceF64, generateReferenceF32, generateReferenceF64, rmsNormF32, rotaryF32 } from "../src/executor.js";
import { selectGreedyToken } from "../src/generation.js";
import { GEMMA4_E4B_PYTORCH_BF16_TRIG_IMPLEMENTATION } from "../src/gemma4-text.js";
import { SafetensorsCatalogReader } from "../src/safetensors.js";
import { decodeMlxF32Payload } from "../src/bridge.js";
import { sleefCosF32, sleefSinF32, sleefTanhF32 } from "../src/sleef-f32.js";
import type { DenseF32Tensor, DenseTensor, ModelCatalog, TensorInfo } from "../src/types.js";
import { roundF32ToBF16 } from "../src/utils.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;
const f64Policy = { computeDtype: "F64", accumulationDtype: "F64", outputDtype: "F64" } as const;
const f32Policy = { computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" } as const;
const sleefTanh = {
  authority: "pytorch-source-and-installed-binary", runtime: "pytorch-eager-cpu-darwin-arm64",
  pytorchSourceCommit: "7269437d655783a26cba32aa88195b741ff496aa",
  sleefSourceCommit: "5a1d179df9cf652951b59010a2d2075372d67f68", kernel: "Sleef_tanhf4_u10advsimd",
} as const;

test("pinned PyTorch ARM SLEEF tanh transcript preserves authoritative F32 results", () => {
  assert.deepEqual([0.1, 1, -1, 8, 0.001, -3.7, 0, 8.7].map(sleefTanhF32), [
    0.0996679961681366, 0.7615941762924194, -0.7615941762924194, 0.9999997615814209,
    0.0009999996982514858, -0.998778223991394, 0, 1,
  ]);
  const input = { shape: [1, 4], values: Float32Array.of(-3.7, 0.001, 1, 8) };
  assert.deepEqual([...activationF32(input, "gelu", "tanh", sleefTanh).values], [
    -0.0002718120813369751, 0.0005003989790566266, 0.8411920070648193, 8,
  ]);
  assert.deepEqual([...elementwiseF32([input], "tanh_softcap", 30, sleefTanh, { afterDivide: "BF16", afterTanh: "BF16", afterMultiply: "BF16" }).values], [
    -3.6875, 0.00099945068359375, 1, 7.84375,
  ]);
});

test("pinned PyTorch ARM SLEEF trig and BF16 RoPE casts preserve authoritative results", () => {
  assert.deepEqual([0, 0.001, 0.1, 0.5, 1, 1.234, 10, 124].map((value) => [sleefSinF32(value), sleefCosF32(value)]), [
    [0, 1],
    [0.0009999999310821295, 0.9999995231628418],
    [0.0998334214091301, 0.9950041770935059],
    [0.4794255495071411, 0.8775825500488281],
    [0.8414709568023682, 0.5403023362159729],
    [0.943818211555481, 0.3304651379585266],
    [-0.5440211296081543, -0.83907151222229],
    [-0.995686948299408, -0.09277620166540146],
  ]);
  const operation = {
    id: "rope", op: "rotary_embedding" as const, input: "x", positionInput: "position_ids", output: "y",
    ropeType: "default", theta: 10_000, rotaryDim: 2, layout: "rotate_half" as const,
    trigImplementation: { ...GEMMA4_E4B_PYTORCH_BF16_TRIG_IMPLEMENTATION },
    rotaryCasts: { cosine: "BF16", sine: "BF16", directProduct: "BF16", rotatedProduct: "BF16", sum: "BF16" } as const,
    dtypePolicy: { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16" },
  };
  const result = rotaryF32({ shape: [1, 1, 1, 2], values: Float32Array.of(0.5, -1) }, [[1]], operation);
  assert.deepEqual([...result.values], [1.109375, -0.119140625]);
  assert.throws(() => sleefSinF32(125), /requires the unimplemented rempif range reducer/);
});

test("PyTorch CPU cascade RMS schedule reproduces the authoritative BF16 vector", () => {
  const input = { shape: [1, 16], values: Float32Array.from([0.5, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15]) };
  const operation = {
    id: "rms", op: "rms_norm" as const, input: "x", output: "y", epsilon: 1e-6, weightTransform: "none" as const,
    axis: -1, reductionSize: 16,
    dtypePolicy: { inputDtype: "BF16", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "BF16", reduction: {
      kind: "pytorch-cpu-f32-cascade-sum" as const, vectorLanes: 4 as const, ilpFactor: 4 as const, cascadeLevels: 4 as const,
      minimumLevelStep: 16 as const, registerFold: "ascending" as const, laneFold: "ascending" as const,
    } },
  };
  assert.deepEqual([...rmsNormF32(input, undefined, operation).values].map(roundF32ToBF16), [
    0.056884765625, -0.11376953125, 0.2275390625, -0.33984375,
    0.455078125, -0.56640625, 0.6796875, -0.796875,
    0.91015625, -1.0234375, 1.1328125, -1.25,
    1.359375, -1.4765625, 1.59375, -1.703125,
  ]);
});

function info(name: string, shape: number[]): TensorInfo {
  return { name, storageDtype: "F64", storageShape: [...shape], logicalShape: [...shape] };
}

function dense(shape: number[], values: number[]): DenseTensor {
  return { shape, values: Float64Array.from(values) };
}

async function tinyLlama() {
  const tensors = new Map<string, TensorInfo>();
  const weights = new Map<string, DenseTensor>();
  const add = (name: string, shape: number[], values: number[]) => {
    tensors.set(name, info(name, shape));
    weights.set(name, dense(shape, values));
  };
  add("model.embed_tokens.weight", [3, 2], [0, 0, 3, 4, 0, 0]);
  add("model.layers.0.input_layernorm.weight", [2], [1, 1]);
  for (const projection of ["q_proj", "k_proj", "v_proj", "o_proj"]) {
    add(`model.layers.0.self_attn.${projection}.weight`, [2, 2], [1, 0, 0, 1]);
  }
  add("model.layers.0.post_attention_layernorm.weight", [2], [1, 1]);
  add("model.layers.0.mlp.gate_proj.weight", [2, 2], [0, 0, 0, 0]);
  add("model.layers.0.mlp.up_proj.weight", [2, 2], [0, 0, 0, 0]);
  add("model.layers.0.mlp.down_proj.weight", [2, 2], [0, 0, 0, 0]);
  add("model.norm.weight", [2], [1, 1]);
  add("lm_head.weight", [3, 2], [1, 0, 0, 1, 1, 1]);
  const catalog: ModelCatalog = {
    source: "/fixtures/tiny-llama",
    format: "safetensors",
    config: {
      model_type: "llama", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1,
      num_attention_heads: 1, num_key_value_heads: 1, head_dim: 2, vocab_size: 3,
      rms_norm_eps: 1e-6, hidden_act: "silu",
    }, rawMetadata: {}, tensors,
  };
  const ir = await buildModelIR(catalog, preview);
  const operations = [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
  for (const operation of operations) {
    operation.dtypePolicy = { ...f64Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F64";
  }
  return { ir, weights };
}

/** Build a two-layer decoder where layer 1 consumes layer 0's post-RoPE KV. */
async function tinySharedKvLlama() {
  const { ir, weights } = await tinyLlama();
  const second = structuredClone(ir.layers[0]!);
  second.index = 1;
  for (const operation of second.operations) {
    operation.id = operation.id.replaceAll("layer_0", "layer_1");
    operation.output = operation.output.replaceAll("layer_0", "layer_1").replace("hidden_states_1", "hidden_states_2");
    operation.layer = 1;
    if ("input" in operation) operation.input = operation.input.replaceAll("layer_0", "layer_1").replace("hidden_states_0", "hidden_states_1");
    if ("inputs" in operation) operation.inputs = operation.inputs.map((input) => input.replaceAll("layer_0", "layer_1").replace("hidden_states_0", "hidden_states_1"));
    if ("query" in operation) operation.query = operation.query.replaceAll("layer_0", "layer_1");
    if ("key" in operation) operation.key = operation.key.replaceAll("layer_0", "layer_1");
    if ("value" in operation) operation.value = operation.value.replaceAll("layer_0", "layer_1");
    if ("weight" in operation) operation.weight.name = operation.weight.name.replace("model.layers.0", "model.layers.1");
    if ("bias" in operation && operation.bias) operation.bias.name = operation.bias.name.replace("model.layers.0", "model.layers.1");
  }
  second.operations = second.operations.filter((operation) => ![
    "layer_1_k_proj", "layer_1_v_proj", "layer_1_k_heads", "layer_1_v_heads", "layer_1_k_rope",
  ].includes(operation.id));
  const consumer = second.operations.find((operation) => operation.id === "layer_1_attention");
  if (!consumer || consumer.op !== "scaled_dot_product_attention") throw new Error("shared-KV fixture did not retain layer 1 attention");
  consumer.key = "layer_0_k_rot";
  consumer.value = "layer_0_v_heads";
  consumer.kvSharing = { enabled: true, producerLayer: 0, group: "full_attention" };
  ir.layers.push(second);
  ir.architecture.numLayers = 2;
  for (const operation of ir.epilogue) {
    if ("input" in operation) operation.input = operation.input.replace("hidden_states_1", "hidden_states_2");
  }
  for (const [name, tensor] of [...weights]) {
    if (name.startsWith("model.layers.0.")) weights.set(name.replace("model.layers.0", "model.layers.1"), dense([...tensor.shape], [...tensor.values]));
  }
  return { ir, weights };
}

test("executor F64 runs a generated dense Llama decoder through logits", async () => {
  const { ir, weights } = await tinyLlama();
  const result = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const inputNorm = Math.sqrt(12.5 + 1e-6);
  const afterAttentionFactor = 1 + 1 / inputNorm;
  const finalFactor = afterAttentionFactor / Math.sqrt(12.5 * afterAttentionFactor ** 2 + 1e-6);
  const expected = [3 * finalFactor, 4 * finalFactor, 7 * finalFactor];
  assert.deepEqual(result.logits.shape, [1, 1, 3]);
  for (const [index, value] of expected.entries()) assert.ok(Math.abs(result.logits.values[index]! - value) < 1e-12);
  assert.equal(result.values.get("layer_0_attention_context")?.shape.join("x"), "1x1x2");
});

test("shared-KV consumers reuse the producer cache across F64 decode without duplicate ownership", async () => {
  const { ir, weights } = await tinySharedKvLlama();
  const full = executeReferenceF64(ir, { inputIds: [[1, 2]], tensors: weights });
  const prefill = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const decoded = executeReferenceF64(ir, { inputIds: [[2]], positionIds: [[1]], pastKeyValues: prefill.pastKeyValues, tensors: weights });
  assert.deepEqual([...prefill.pastKeyValues.keys()], [0]);
  assert.deepEqual([...decoded.pastKeyValues.keys()], [0]);
  assert.equal(decoded.pastKeyValues.get(0)?.key.shape.join("x"), "1x1x2x2");
  for (let index = 0; index < decoded.logits.values.length; index += 1) {
    assert.ok(Math.abs(decoded.logits.values[index]! - full.logits.values[full.logits.values.length - decoded.logits.values.length + index]!) < 1e-12);
  }
  const generated = generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: 2 });
  assert.deepEqual(generated.stepPastKeyValues.map((cache) => [...cache.keys()]), [[0], [0]]);
  const consumer = ir.layers[1]!.operations.find((operation) => operation.op === "scaled_dot_product_attention")!;
  if (consumer.op !== "scaled_dot_product_attention") throw new Error("shared-KV fixture has no consumer");
  consumer.kvSharing = { enabled: true, producerLayer: 1 };
  assert.throws(() => executeReferenceF64(ir, { inputIds: [[1]], tensors: weights }), /producerLayer inteiro de uma camada anterior/);
});

test("shared-KV consumers preserve F32 incremental logits and producer-only cache", async () => {
  const { ir, weights } = await tinySharedKvLlama();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const f32Weights = new Map([...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]));
  const full = executeReferenceF32(ir, { inputIds: [[1, 2]], tensors: f32Weights });
  const prefill = executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights });
  const decoded = executeReferenceF32(ir, { inputIds: [[2]], positionIds: [[1]], pastKeyValues: prefill.pastKeyValues, tensors: f32Weights });
  assert.deepEqual([...decoded.pastKeyValues.keys()], [0]);
  for (let index = 0; index < decoded.logits.values.length; index += 1) {
    assert.equal(decoded.logits.values[index], full.logits.values[full.logits.values.length - decoded.logits.values.length + index]);
  }
});

test("operation differential report compares every stable IR output and records required evidence", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const allOperations = [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
  const report = compareExecutionTrace(ir, candidate, {
    runtime: "fixture-authoritative-runtime",
    model: "tiny-llama",
    revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors",
    quantization: "none",
    inputTokens: [[1]],
    dtypePolicy: "F64 scalar fixture",
    operations: allOperations.map((operation) => ({
      operationId: operation.id,
      output: operation.output,
      tensor: candidate.values.get(operation.output)!,
    })),
    pastKeyValues: [...candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value })),
  }, { candidateRuntime: "llm-inner F64 scalar", topK: 3 });
  assert.equal(report.fidelityClass, "lossless-within-dtype");
  assert.equal(report.firstDivergentOperation, null);
  assert.equal(report.operations.length, allOperations.length);
  assert.deepEqual(report.kvCache.map((comparison) => comparison.status), ["pass"]);
  assert.equal(report.logits?.argmaxAgreement, true);
  assert.equal(report.logits?.topKOverlap, 1);
  assert.equal(report.reference.revisionOrChecksum, "in-repository-fixture");
});

test("operation differential report fails closed for missing captures, shape drift, and numeric divergence", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const operations = [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
  const first = operations[0]!;
  const firstValue = candidate.values.get(first.output)!;
  const altered = { shape: [...firstValue.shape], values: Float64Array.from(firstValue.values) };
  altered.values[0] = altered.values[0]! + 1;
  const report = compareExecutionTrace(ir, candidate, {
    runtime: "fixture-authoritative-runtime",
    model: "tiny-llama",
    revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors",
    quantization: "none",
    inputTokens: [[1]],
    dtypePolicy: "F64 scalar fixture",
    operations: [{ operationId: first.id, output: first.output, tensor: altered }],
    pastKeyValues: [],
  }, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(report.fidelityClass, "incomplete");
  assert.equal(report.firstDivergentOperation, first.id);
  assert.equal(report.operations[0]?.status, "diverged");
  assert.equal(report.missingReferenceOperationIds.length, operations.length - 1);

  const shapeReport = compareExecutionTrace(ir, candidate, {
    runtime: "fixture-authoritative-runtime", model: "tiny-llama", revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F64 scalar fixture",
    operations: [{ operationId: first.id, output: first.output, tensor: { shape: [99], values: new Float64Array(99) } }],
    pastKeyValues: [],
  }, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(shapeReport.operations[0]?.status, "shape-mismatch");
});

test("a complete operation trace with a numerical mismatch is approximate, not incomplete", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const operations = [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
  const referenceOperations = operations.map((operation) => {
    const tensor = candidate.values.get(operation.output)!;
    return { operationId: operation.id, output: operation.output, tensor: { shape: [...tensor.shape], values: Float64Array.from(tensor.values) } };
  });
  referenceOperations[0]!.tensor.values[0] = referenceOperations[0]!.tensor.values[0]! + 1;
  const report = compareExecutionTrace(ir, candidate, {
    runtime: "fixture-authoritative-runtime", model: "tiny-llama", revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors", quantization: "none", inputTokens: [[1]], dtypePolicy: "F64 scalar fixture",
    operations: referenceOperations,
    pastKeyValues: [...candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value })),
  }, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(report.fidelityClass, "approximate");
  assert.equal(report.firstDivergentOperation, operations[0]!.id);
  assert.deepEqual(report.missingReferenceOperationIds, []);
});

test("operation differential report compares terminal softcapped logits to final_logit_softcap", async () => {
  const { ir, weights } = await tinyLlama();
  const softcap = 0.25;
  ir.epilogue.push({
    id: "final_logit_softcap",
    op: "elementwise",
    kind: "tanh_softcap",
    inputs: ["logits"],
    scalar: softcap,
    output: "softcapped_logits",
    dtypePolicy: { computeDtype: "F64", accumulationDtype: "F64", outputDtype: "F64" },
  });
  const candidate = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const operations = [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
  const report = compareExecutionTrace(ir, candidate, {
    runtime: "fixture-authoritative-runtime",
    model: "tiny-llama-with-softcap",
    revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors",
    quantization: "none",
    inputTokens: [[1]],
    dtypePolicy: "F64 scalar fixture",
    operations: operations.map((operation) => ({
      operationId: operation.id,
      output: operation.output,
      tensor: candidate.values.get(operation.output)!,
    })),
    pastKeyValues: [...candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value })),
  }, { candidateRuntime: "llm-inner F64 scalar", topK: 3 });

  const rawLogits = candidate.values.get("logits")!;
  assert.notDeepEqual([...candidate.logits.values], [...rawLogits.values]);
  assert.equal(report.fidelityClass, "lossless-within-dtype");
  assert.equal(report.logits?.maxAbsoluteError, 0);
  assert.equal(report.logits?.argmaxAgreement, true);
});

test("executor F64 fails closed for compiler-default implicit dtype policies", async () => {
  const { ir, weights } = await tinyLlama();
  ir.prelude[0]!.dtypePolicy = { computeDtype: "model-configured" };
  assert.throws(
    () => executeReferenceF64(ir, { inputIds: [[1]], tensors: weights }),
    /política F64 explícita/,
  );
});

test("executor F64 rejects missing cast boundaries instead of assuming F64", async () => {
  const { ir, weights } = await tinyLlama();
  ir.prelude[0]!.dtypePolicy = {};
  assert.throws(
    () => executeReferenceF64(ir, { inputIds: [[1]], tensors: weights }),
    /política F64 explícita.*computeDtype=ausente/,
  );
});

test("executor F64 rejects an attention softmax declared in another dtype", async () => {
  const { ir, weights } = await tinyLlama();
  const attention = ir.layers[0]!.operations.find((operation) => operation.op === "scaled_dot_product_attention");
  assert.equal(attention?.op, "scaled_dot_product_attention");
  if (attention?.op === "scaled_dot_product_attention") attention.softmaxComputeDtype = "F32";
  assert.throws(
    () => executeReferenceF64(ir, { inputIds: [[1]], tensors: weights }),
    /softmaxComputeDtype=F64/,
  );
});

test("executor F32 fails closed instead of treating non-default RoPE as rotate_half", async () => {
  const { ir, weights } = await tinyLlama();
  const f32Weights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const qRope = ir.layers[0]!.operations.find((operation) => operation.id === "layer_0_q_rope");
  assert.equal(qRope?.op, "rotary_embedding");
  if (qRope?.op !== "rotary_embedding") throw new Error("fixture sem Q RoPE");

  qRope.ropeType = "linear";
  assert.throws(() => executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights }), /ropeType=linear/);

  qRope.ropeType = "default";
  qRope.layout = "interleaved_pairs";
  assert.throws(() => executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights }), /layout RoPE rotate_half/);

  qRope.layout = "rotate_half";
  qRope.scaling = { rope_type: "default", factor: 2 };
  assert.throws(() => executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights }), /rope_scaling explícito/);
});

test("executor F64 applies a canonical additive attention mask before softmax", async () => {
  const { ir, weights } = await tinyLlama();
  const unmasked = executeReferenceF64(ir, { inputIds: [[1, 0]], tensors: weights });
  const attentionMask = dense([1, 1, 2, 2], [0, -Infinity, -Infinity, 0]);
  const masked = executeReferenceF64(ir, { inputIds: [[1, 0]], attentionMask, tensors: weights });
  const unmaskedContext = unmasked.values.get("layer_0_attention_context")!;
  const maskedContext = masked.values.get("layer_0_attention_context")!;
  assert.ok(unmaskedContext.values.slice(2, 4).every((value) => value !== 0));
  assert.deepEqual([...maskedContext.values.slice(2, 4)], [0, 0]);
});

test("executor F64 incremental KV cache matches the final full-prompt logits", async () => {
  const { ir, weights } = await tinyLlama();
  const full = executeReferenceF64(ir, { inputIds: [[1, 1]], tensors: weights });
  const prefill = executeReferenceF64(ir, { inputIds: [[1]], tensors: weights });
  const decoded = executeReferenceF64(ir, {
    inputIds: [[1]],
    positionIds: [[1]],
    pastKeyValues: prefill.pastKeyValues,
    tensors: weights,
  });
  assert.deepEqual([...decoded.logits.values], [...full.logits.values.slice(3, 6)]);
  assert.equal(decoded.pastKeyValues.get(0)?.key.shape.join("x"), "1x1x2x2");
  assert.equal(decoded.pastKeyValues.get(0)?.value.shape.join("x"), "1x1x2x2");
});

test("F64 greedy generation advances absolute positions and returns a cache for every emitted token", async () => {
  const { ir, weights } = await tinyLlama();
  const generated = generateReferenceF64(ir, {
    inputIds: [[1]],
    positionIds: [[7]],
    tensors: weights,
    maxNewTokens: 3,
  });
  assert.deepEqual(generated.generatedTokenIds, [2, 2, 2]);
  assert.deepEqual(generated.inputIds, [1, 2, 2, 2]);
  assert.deepEqual(generated.steps, [
    { tokenId: 2, positionId: 8 },
    { tokenId: 2, positionId: 9 },
    { tokenId: 2, positionId: 10 },
  ]);
  assert.equal(generated.selectionLogits.length, 3);
  assert.equal(generated.pastKeyValues.get(0)?.key.shape.join("x"), "1x1x4x2");
  const direct = executeReferenceF64(ir, { inputIds: [[1, 2, 2, 2]], positionIds: [[7, 8, 9, 10]], tensors: weights });
  assert.deepEqual([...generated.logits.values], [...direct.logits.values.slice(9, 12)]);
});

test("greedy selection uses the final sequence logits, rejects non-finite scores, and keeps the lowest tied token", () => {
  assert.equal(selectGreedyToken(dense([1, 2, 3], [99, 0, 0, 4, 7, 7])), 1);
  assert.throws(() => selectGreedyToken(dense([1, 1, 3], [0, Number.NaN, 2])), /finitos/);
  assert.throws(() => selectGreedyToken(dense([1, 0, 3], [])), /shape/);
});

test("generation differential report requires token, position, terminal-logit, and KV-cache agreement", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = generateReferenceF64(ir, { inputIds: [[1]], positionIds: [[7]], tensors: weights, maxNewTokens: 2 });
  const reference = {
    runtime: "fixture-authoritative-runtime",
    model: "tiny-llama",
    revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors" as const,
    quantization: "none",
    inputTokens: [1],
    promptPositionIds: [7],
    dtypePolicy: "F64 scalar fixture",
    maxNewTokens: 2,
    generatedTokenIds: [...candidate.generatedTokenIds],
    steps: candidate.steps.map((step) => ({ ...step })),
    selectionLogits: [...candidate.selectionLogits],
    stepPastKeyValues: candidate.stepPastKeyValues.map((snapshot) => [...snapshot].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value }))),
    logits: candidate.logits,
    pastKeyValues: [...candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value })),
  };
  const report = compareGenerationTrace(candidate, reference, { candidateRuntime: "llm-inner F64 scalar", topK: 3 });
  assert.equal(report.fidelityClass, "lossless-within-dtype");
  assert.equal(report.firstDivergence, null);
  assert.equal(report.promptMatches, true);
  assert.deepEqual(report.generatedTokenIds.map((step) => step.status), ["pass", "pass"]);
  assert.deepEqual(report.generatedTokenIds.map((step) => step.selectionLogits?.maxAbsoluteError), [0, 0]);
  assert.deepEqual(report.kvCache.map((cache) => cache.status), ["pass"]);
  assert.equal(report.terminalLogits?.argmaxAgreement, true);

  const selectionLogits = candidate.selectionLogits.map((tensor) => ({ shape: [...tensor.shape], values: Float64Array.from(tensor.values) }));
  selectionLogits[0]!.values[0] = selectionLogits[0]!.values[0]! + 1;
  const selectionDivergence = compareGenerationTrace(candidate, { ...reference, selectionLogits }, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(selectionDivergence.fidelityClass, "approximate");
  assert.equal(selectionDivergence.firstDivergence, "generation:selection-logits-0");

  const nonGreedyReference = { ...reference, generatedTokenIds: [0, ...reference.generatedTokenIds.slice(1)], steps: [{ ...reference.steps[0]!, tokenId: 0 }, ...reference.steps.slice(1)] };
  assert.throws(() => compareGenerationTrace(candidate, nonGreedyReference, { candidateRuntime: "llm-inner F64 scalar" }), /logits determinísticos exigem outro argmax/);

  const nonGreedyCandidate = { ...candidate, generatedTokenIds: [0, ...candidate.generatedTokenIds.slice(1)], inputIds: [1, 0, ...candidate.generatedTokenIds.slice(1)], steps: [{ ...candidate.steps[0]!, tokenId: 0 }, ...candidate.steps.slice(1)] };
  assert.throws(() => compareGenerationTrace(nonGreedyCandidate, reference, { candidateRuntime: "llm-inner F64 scalar" }), /logits determinísticos exigem outro argmax/);

  const stepPastKeyValues = reference.stepPastKeyValues.map((snapshot) => snapshot.map((cache) => ({
    ...cache,
    key: { shape: [...cache.key.shape], values: Float64Array.from(cache.key.values) },
    value: { shape: [...cache.value.shape], values: Float64Array.from(cache.value.values) },
  })));
  stepPastKeyValues[0]![0]!.key.values[0] = stepPastKeyValues[0]![0]!.key.values[0]! + 1;
  const cacheDivergence = compareGenerationTrace(candidate, { ...reference, stepPastKeyValues }, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(cacheDivergence.fidelityClass, "approximate");
  assert.equal(cacheDivergence.firstDivergence, "generation:step-0-kv-cache-layer-0");
  assert.equal(cacheDivergence.generatedTokenIds[0]?.kvCache?.[0]?.status, "diverged");
});

test("generation differential report marks token/position divergence approximate and missing evidence incomplete", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: 1 });
  const base = {
    runtime: "fixture-authoritative-runtime", model: "tiny-llama", revisionOrChecksum: "in-repository-fixture",
    containerFormat: "safetensors" as const, quantization: "none", inputTokens: [1], promptPositionIds: [0],
    dtypePolicy: "F64 scalar fixture", maxNewTokens: 1, generatedTokenIds: [2],
    steps: [{ tokenId: 2, positionId: 99 }], selectionLogits: [...candidate.selectionLogits],
    stepPastKeyValues: candidate.stepPastKeyValues.map((snapshot) => [...snapshot].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value }))), logits: candidate.logits,
    pastKeyValues: [...candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value })),
  };
  const divergent = compareGenerationTrace(candidate, base, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(divergent.fidelityClass, "approximate");
  assert.equal(divergent.firstDivergence, "generation:step-0");
  const incomplete = compareGenerationTrace(candidate, { ...base, steps: [], selectionLogits: [], stepPastKeyValues: [], generatedTokenIds: [], pastKeyValues: [] }, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(incomplete.fidelityClass, "incomplete");
  assert.equal(incomplete.generatedTokenIds[0]?.status, "missing-reference");
  assert.equal(incomplete.kvCache[0]?.status, "missing-reference");
});

test("generation differential report rejects malformed authoritative captures", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: 1 });
  assert.throws(() => compareGenerationTrace(candidate, {
    runtime: "fixture", model: "tiny", revisionOrChecksum: "fixture", containerFormat: "safetensors", quantization: "none",
    inputTokens: [1], promptPositionIds: [0], dtypePolicy: "F64", maxNewTokens: 1, eosTokenId: 2,
    generatedTokenIds: [2, 2], steps: [{ tokenId: 2, positionId: 1 }, { tokenId: 2, positionId: 2 }], selectionLogits: [...candidate.selectionLogits], stepPastKeyValues: candidate.stepPastKeyValues.map((snapshot) => [...snapshot].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value }))),
    logits: candidate.logits, pastKeyValues: [],
  }, { candidateRuntime: "llm-inner F64 scalar" }), /mais tokens do que o limite declarado/);
});

test("generation differential report never treats a different prompt or malformed candidate result as equivalent", async () => {
  const { ir, weights } = await tinyLlama();
  const candidate = generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: 1 });
  const reference = {
    runtime: "fixture", model: "tiny", revisionOrChecksum: "fixture", containerFormat: "safetensors" as const, quantization: "none",
    inputTokens: [0], promptPositionIds: [0], dtypePolicy: "F64", maxNewTokens: 1,
    generatedTokenIds: [...candidate.generatedTokenIds], steps: candidate.steps, selectionLogits: [...candidate.selectionLogits], stepPastKeyValues: candidate.stepPastKeyValues.map((snapshot) => [...snapshot].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value }))), logits: candidate.logits,
    pastKeyValues: [...candidate.pastKeyValues].map(([layer, cache]) => ({ layer, key: cache.key, value: cache.value })),
  };
  const report = compareGenerationTrace(candidate, reference, { candidateRuntime: "llm-inner F64 scalar" });
  assert.equal(report.promptMatches, false);
  assert.equal(report.firstDivergence, "generation:prompt");
  assert.equal(report.fidelityClass, "approximate");
  assert.throws(() => compareGenerationTrace({ ...candidate, generatedTokenIds: [0] }, { ...reference, inputTokens: [1] }, { candidateRuntime: "llm-inner F64 scalar" }), /Resultado candidato de geração/);
});

test("greedy generation evaluates EOS before stopping and rejects ambiguous generation inputs", async () => {
  const { ir, weights } = await tinyLlama();
  const eos = generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: 5, eosTokenId: 2 });
  assert.deepEqual(eos.generatedTokenIds, [2]);
  assert.equal(eos.pastKeyValues.get(0)?.key.shape.join("x"), "1x1x2x2");
  assert.throws(() => generateReferenceF64(ir, { inputIds: [[1], [1]], tensors: weights, maxNewTokens: 1 }), /exatamente um prompt/);
  assert.throws(() => generateReferenceF64(ir, { inputIds: [[1]], positionIds: [[0.5]], tensors: weights, maxNewTokens: 1 }), /posições absolutas/);
  assert.throws(() => generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: -1 }), /maxNewTokens/);
  assert.throws(() => generateReferenceF64(ir, { inputIds: [[1]], tensors: weights, maxNewTokens: 1, eosTokenId: -1 }), /eosTokenId/);
});

test("executor F64 rejects incomplete and incompatible KV caches", async () => {
  const { ir, weights } = await tinyLlama();
  assert.throws(
    () => executeReferenceF64(ir, { inputIds: [[1]], pastKeyValues: new Map(), tensors: weights }),
    /não contém a camada 0/,
  );
  assert.throws(
    () => executeReferenceF64(ir, {
      inputIds: [[1]],
      pastKeyValues: new Map([[0, { key: dense([1, 1, 1, 3], [0, 0, 0]), value: dense([1, 1, 1, 3], [0, 0, 0]) }]]),
      tensors: weights,
    }),
    /shape do cache KV é incompatível/,
  );
});

test("executors reject malformed or non-canonical additive attention masks", async () => {
  const { ir, weights } = await tinyLlama();
  const f32Weights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const masked = executeReferenceF32(ir, {
    inputIds: [[1, 0]],
    attentionMask: { shape: [1, 1, 2, 2], values: Float32Array.from([0, -Infinity, -Infinity, 0]) },
    tensors: f32Weights,
  });
  assert.deepEqual([...masked.values.get("layer_0_attention_context")!.values.slice(2, 4)], [0, 0]);
  assert.throws(
    () => executeReferenceF32(ir, { inputIds: [[1, 0]], attentionMask: { shape: [1, 2], values: new Float32Array(2) }, tensors: f32Weights }),
    /attentionMask deve ter shape/,
  );
  assert.throws(
    () => executeReferenceF32(ir, { inputIds: [[1, 0]], attentionMask: { shape: [1, 1, 2, 2], values: Float32Array.from([0, 0, NaN, 0]) }, tensors: f32Weights }),
    /valores finitos ou -Infinity/,
  );
  assert.throws(
    () => executeReferenceF32(ir, { inputIds: [[1, 0]], attentionMask: { shape: [1, 1, 2, 2], values: Float32Array.from([-Infinity, -Infinity, 0, -Infinity]) }, tensors: f32Weights }),
    /excluiu todas as chaves/,
  );
});

test("executor F32 incremental KV cache preserves declared F32 boundaries", async () => {
  const { ir, weights } = await tinyLlama();
  const f32Weights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const full = executeReferenceF32(ir, { inputIds: [[1, 1]], tensors: f32Weights });
  const prefill = executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights });
  const decoded = executeReferenceF32(ir, {
    inputIds: [[1]],
    positionIds: [[1]],
    pastKeyValues: prefill.pastKeyValues,
    tensors: f32Weights,
  });
  assert.deepEqual([...decoded.logits.values], [...full.logits.values.slice(3, 6)]);
  assert.ok(decoded.pastKeyValues.get(0)?.key.values instanceof Float32Array);
});

test("F32 greedy generation keeps F32 logits and canonical cache ownership", async () => {
  const { ir, weights } = await tinyLlama();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const f32Weights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  const generated = generateReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights, maxNewTokens: 2 });
  assert.deepEqual(generated.generatedTokenIds, [2, 2]);
  assert.ok(generated.logits.values instanceof Float32Array);
  assert.ok(generated.pastKeyValues.get(0)?.key.values instanceof Float32Array);
  assert.equal(generated.pastKeyValues.get(0)?.key.shape.join("x"), "1x1x3x2");
});

test("reader range-loads an on-disk F64 Safetensors fixture into executor logits", async () => {
  const { ir, weights } = await tinyLlama();
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-f64-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({
      model_type: "llama", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1,
      num_attention_heads: 1, num_key_value_heads: 1, head_dim: 2, vocab_size: 3,
      rms_norm_eps: 1e-6, hidden_act: "silu",
    }));
    await writeF64Safetensors(path.join(directory, "model.safetensors"), weights);

    const reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      const loaded = new Map<string, DenseTensor>();
      for (const tensor of catalog.tensors.values()) loaded.set(tensor.name, await reader.readDenseF64(tensor));
      const result = executeReferenceF64(ir, { inputIds: [[1]], tensors: loaded });
      assert.deepEqual([...result.logits.values], [...(executeReferenceF64(ir, { inputIds: [[1]], tensors: weights })).logits.values]);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("reader rejects F32 range loads instead of silently widening their semantics", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-f32-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    const header = Buffer.from(JSON.stringify({ "x.weight": { dtype: "F32", shape: [1], data_offsets: [0, 4] } }), "utf8");
    const payload = Buffer.alloc(4);
    payload.writeFloatLE(1, 0);
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(header.length));
    await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, header, payload]));
    const reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      await assert.rejects(() => reader.readDenseF64(catalog.tensors.get("x.weight")!), /storageDtype=F64/);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("reader range-loads F32 Safetensors into the explicit F32 executor", async () => {
  const { ir, weights } = await tinyLlama();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const f32Weights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-f32-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    await writeF32Safetensors(path.join(directory, "model.safetensors"), f32Weights);
    const reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      const loaded = new Map<string, DenseF32Tensor>();
      for (const tensor of catalog.tensors.values()) loaded.set(tensor.name, await reader.readDenseF32(tensor));
      const result = executeReferenceF32(ir, { inputIds: [[1]], tensors: loaded });
      const direct = executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights });
      assert.ok(result.logits.values instanceof Float32Array);
      assert.deepEqual([...result.logits.values], [...direct.logits.values]);
      assert.deepEqual([...loaded.get("model.embed_tokens.weight")!.values], [0, 0, 3, 4, 0, 0]);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("reader range-loads F16 and BF16 Safetensors losslessly into F32 values", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-f16-bf16-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    await writeMixed16Safetensors(path.join(directory, "model.safetensors"));
    const reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      const f16 = await reader.readDenseF16AsF32(catalog.tensors.get("f16.weight")!);
      const bf16 = await reader.readDenseBF16AsF32(catalog.tensors.get("bf16.weight")!);
      assert.deepEqual(f16.shape, [6]);
      assert.deepEqual([...f16.values.slice(0, 4)], [1, -2, 0.00006103515625, 65504]);
      assert.equal(f16.values[4], Infinity);
      assert.ok(Number.isNaN(f16.values[5]!));
      assert.deepEqual([...bf16.values.slice(0, 4)], [1, -2.5, 0.5, 3.3895313892515355e38]);
      assert.equal(bf16.values[4], -Infinity);
      assert.ok(Number.isNaN(bf16.values[5]!));
      await assert.rejects(() => reader.readDenseF16AsF32(catalog.tensors.get("bf16.weight")!), /storageDtype=F16/);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("F32 executor runs a complete decoder from F16 and BF16 Safetensors storage", async () => {
  const { ir, weights } = await tinyLlama();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const expectedWeights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  const expected = executeReferenceF32(ir, { inputIds: [[1]], tensors: expectedWeights });

  for (const storageDtype of ["F16", "BF16"] as const) {
    const directory = await mkdtemp(path.join(tmpdir(), `llm-inner-${storageDtype.toLowerCase()}-decoder-`));
    try {
      await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
      await write16Safetensors(path.join(directory, "model.safetensors"), weights, storageDtype);
      const reader = new SafetensorsCatalogReader(directory);
      try {
        const catalog = await reader.inspect();
        const loaded = new Map<string, DenseF32Tensor>();
        for (const tensor of catalog.tensors.values()) loaded.set(tensor.name, await reader.readDenseAsF32(tensor));
        const result = executeReferenceF32(ir, { inputIds: [[1]], tensors: loaded });
        assert.deepEqual([...result.logits.values], [...expected.logits.values], `${storageDtype} logits must preserve the decoded storage values`);
      } finally {
        await reader.close();
      }
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }
});

test("F32 dense dispatch rejects integer and quantized storage rather than guessing a conversion", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-f32-dispatch-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    const header = Buffer.from(JSON.stringify({ "x.weight": { dtype: "I8", shape: [1], data_offsets: [0, 1] } }), "utf8");
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(header.length));
    await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, header, Buffer.from([1])]));
    const reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      await assert.rejects(() => reader.readDenseAsF32(catalog.tensors.get("x.weight")!), /storageDtype=I8/);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("F32 executor rejects an F64 policy instead of silently changing cast boundaries", async () => {
  const { ir, weights } = await tinyLlama();
  const f32Weights = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  assert.throws(() => executeReferenceF32(ir, { inputIds: [[1]], tensors: f32Weights }), /política F32 explícita/);
});

test("MLX dequantized F32 payload preserves declared provenance and rejects malformed backend bytes", () => {
  const quantization = {
    family: "mlx" as const, mode: "affine", bits: 4, groupSize: 4,
    scaleTensor: "linear.scales", biasTensor: "linear.biases",
  };
  const bytes = Buffer.alloc(8);
  bytes.writeFloatLE(1.25, 0);
  bytes.writeFloatLE(-2.5, 4);
  const tensor = decodeMlxF32Payload({ shape: [1, 2], f32leBase64: bytes.toString("base64") }, [1, 2], quantization);
  assert.deepEqual([...tensor.values], [1.25, -2.5]);
  assert.deepEqual(tensor.sourceQuantization, quantization);
  assert.throws(
    () => decodeMlxF32Payload({ shape: [1, 2], f32leBase64: Buffer.alloc(4).toString("base64") }, [1, 2], quantization),
    /bytes F32/,
  );
  assert.throws(
    () => decodeMlxF32Payload({ shape: [2], f32leBase64: bytes.toString("base64") }, [1, 2], quantization),
    /shape/,
  );
});

test("F32 executor accepts MLX materialized weights only with matching quantization provenance", async () => {
  const { ir, weights } = await tinyLlama();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { ...f32Policy };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
  const lmHead = ir.epilogue.find((operation) => operation.id === "lm_head");
  assert.equal(lmHead?.op, "linear");
  if (lmHead?.op !== "linear") throw new Error("fixture sem lm_head");
  const quantization = { family: "mlx" as const, mode: "affine", bits: 4, groupSize: 4, scaleTensor: "lm_head.scales" };
  lmHead.weight.quantization = quantization;
  const materialized = new Map<string, DenseF32Tensor>(
    [...weights].map(([name, tensor]) => [name, { shape: [...tensor.shape], values: Float32Array.from(tensor.values) }]),
  );
  const head = materialized.get("lm_head.weight")!;
  head.sourceQuantization = { ...quantization };
  assert.doesNotThrow(() => executeReferenceF32(ir, { inputIds: [[1]], tensors: materialized }));
  delete head.sourceQuantization;
  assert.throws(
    () => executeReferenceF32(ir, { inputIds: [[1]], tensors: materialized }),
    /exige proveniência idêntica/,
  );
});

test("Safetensors inspection rejects inverted, out-of-payload, and overlapping tensor ranges", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-invalid-ranges-"));
  const model = path.join(directory, "model.safetensors");
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    const cases: Array<{ name: string; header: object; expected: RegExp }> = [
      {
        name: "inverted",
        header: { "x.weight": { dtype: "F32", shape: [1], data_offsets: [4, 0] } },
        expected: /data_offsets invertidos/,
      },
      {
        name: "out-of-payload",
        header: { "x.weight": { dtype: "F32", shape: [1], data_offsets: [0, 8] } },
        expected: /ultrapassa o payload/,
      },
      {
        name: "overlap",
        header: {
          "x.weight": { dtype: "F32", shape: [1], data_offsets: [0, 4] },
          "y.weight": { dtype: "F32", shape: [1], data_offsets: [2, 6] },
        },
        expected: /sobrepõe outro intervalo/,
      },
    ];
    for (const fixture of cases) {
      await writeRawSafetensors(model, fixture.header, Buffer.alloc(6));
      const reader = new SafetensorsCatalogReader(directory);
      try {
        await assert.rejects(() => reader.inspect(), fixture.expected, fixture.name);
      } finally {
        await reader.close();
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Safetensors inspection rejects hostile shard paths and oversized headers before opening them", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-invalid-shard-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama" }));
    await writeFile(
      path.join(directory, "model.safetensors.index.json"),
      JSON.stringify({ weight_map: { "x.weight": "../outside.safetensors" } }),
    );
    let reader = new SafetensorsCatalogReader(directory);
    try {
      await assert.rejects(() => reader.inspect(), /Nome de shard Safetensors inválido/);
    } finally {
      await reader.close();
    }

    await rm(path.join(directory, "model.safetensors.index.json"));
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(100 * 1024 * 1024 + 1));
    await writeFile(path.join(directory, "model.safetensors"), prefix);
    reader = new SafetensorsCatalogReader(directory);
    try {
      await assert.rejects(() => reader.inspect(), /limite Safetensors/);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("MLX quantization requires U32 packing and a row-compatible scales matrix", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-mlx-layout-"));
  try {
    const config = {
      model_type: "llama",
      quantization: { bits: 4, group_size: 4, mode: "affine" },
    };
    await writeFile(path.join(directory, "config.json"), JSON.stringify(config));

    await writeRawSafetensors(path.join(directory, "model.safetensors"), {
      "linear.weight": { dtype: "U32", shape: [2, 1], data_offsets: [0, 8] },
    }, Buffer.alloc(8));
    let reader = new SafetensorsCatalogReader(directory);
    try {
      await assert.rejects(() => reader.inspect(), /linear\.scales está ausente/);
    } finally {
      await reader.close();
    }

    await writeRawSafetensors(path.join(directory, "model.safetensors"), {
      "linear.weight": { dtype: "U32", shape: [2, 1], data_offsets: [0, 8] },
      "linear.scales": { dtype: "F16", shape: [1, 2], data_offsets: [8, 12] },
    }, Buffer.alloc(12));
    reader = new SafetensorsCatalogReader(directory);
    try {
      await assert.rejects(() => reader.inspect(), /scales possui 1 linhas, mas o peso possui 2/);
    } finally {
      await reader.close();
    }

    await writeRawSafetensors(path.join(directory, "model.safetensors"), {
      "linear.weight": { dtype: "F32", shape: [2, 4], data_offsets: [0, 32] },
      "linear.scales": { dtype: "F16", shape: [2, 1], data_offsets: [32, 36] },
    }, Buffer.alloc(36));
    reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      const weight = catalog.tensors.get("linear.weight")!;
      assert.equal(catalog.format, "safetensors");
      assert.equal(weight.quantization, undefined);
      assert.deepEqual(weight.logicalShape, [2, 4]);
    } finally {
      await reader.close();
    }

    await writeFile(path.join(directory, "config.json"), JSON.stringify({
      model_type: "llama", quantization: { bits: 4, group_size: 4 },
    }));
    await writeRawSafetensors(path.join(directory, "model.safetensors"), {
      "linear.weight": { dtype: "U32", shape: [2, 1], data_offsets: [0, 8] },
      "linear.scales": { dtype: "F16", shape: [2, 1], data_offsets: [8, 12] },
    }, Buffer.alloc(12));
    reader = new SafetensorsCatalogReader(directory);
    try {
      await assert.rejects(() => reader.inspect(), /mode explícito/);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("native MLX affine reader reconstructs little-endian U32 codes with per-group F32 parameters", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "llm-inner-mlx-affine-"));
  try {
    await writeFile(path.join(directory, "config.json"), JSON.stringify({ model_type: "llama", quantization: { bits: 4, group_size: 4, mode: "affine" } }));
    const weights = Buffer.alloc(32);
    for (let row = 0; row < 2; row += 1) {
      for (let column = 0; column < 32; column += 1) {
        const code = row === 0 ? column & 15 : 15 - (column & 15);
        const wordOffset = (row * 4 + Math.floor(column / 8)) * 4;
        weights.writeUInt32LE((weights.readUInt32LE(wordOffset) | (code << ((column % 8) * 4))) >>> 0, wordOffset);
      }
    }
    const scales = Buffer.alloc(64);
    const biases = Buffer.alloc(64);
    for (let index = 0; index < 16; index += 1) {
      scales.writeFloatLE(index + 1, index * 4);
      biases.writeFloatLE(100 + index, index * 4);
    }
    await writeRawSafetensors(path.join(directory, "model.safetensors"), {
      "linear.weight": { dtype: "U32", shape: [2, 4], data_offsets: [0, 32] },
      "linear.scales": { dtype: "F32", shape: [2, 8], data_offsets: [32, 96] },
      "linear.biases": { dtype: "F32", shape: [2, 8], data_offsets: [96, 160] },
    }, Buffer.concat([weights, scales, biases]));
    const reader = new SafetensorsCatalogReader(directory);
    try {
      const catalog = await reader.inspect();
      const weight = catalog.tensors.get("linear.weight")!;
      const values = await reader.readMlxAffineAsF32(weight, catalog.tensors.get("linear.scales")!, catalog.tensors.get("linear.biases")!);
      assert.deepEqual(values.shape, [2, 32]);
      assert.deepEqual(values.sourceQuantization, { family: "mlx", mode: "affine", bits: 4, groupSize: 4, scaleTensor: "linear.scales", biasTensor: "linear.biases" });
      assert.equal(values.values[0], 100);
      assert.equal(values.values[3], 103);
      assert.equal(values.values[4], 109);
      assert.equal(values.values[31], 227);
      assert.equal(values.values[32], 243);
      assert.equal(values.values[63], 115);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

async function writeRawSafetensors(file: string, header: object, payload: Buffer): Promise<void> {
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.length));
  await writeFile(file, Buffer.concat([prefix, headerBytes, payload]));
}

async function writeF64Safetensors(file: string, tensors: ReadonlyMap<string, DenseTensor>): Promise<void> {
  let offset = 0;
  const header: Record<string, { dtype: "F64"; shape: number[]; data_offsets: [number, number] }> = {};
  for (const [name, tensor] of tensors) {
    const length = tensor.values.length * Float64Array.BYTES_PER_ELEMENT;
    header[name] = { dtype: "F64", shape: [...tensor.shape], data_offsets: [offset, offset + length] };
    offset += length;
  }
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.length));
  const payload = Buffer.alloc(offset);
  let byteOffset = 0;
  for (const tensor of tensors.values()) {
    for (let index = 0; index < tensor.values.length; index += 1) payload.writeDoubleLE(tensor.values[index]!, byteOffset + index * Float64Array.BYTES_PER_ELEMENT);
    byteOffset += tensor.values.length * Float64Array.BYTES_PER_ELEMENT;
  }
  await writeFile(file, Buffer.concat([prefix, headerBytes, payload]));
}

async function writeF32Safetensors(file: string, tensors: ReadonlyMap<string, DenseF32Tensor>): Promise<void> {
  let offset = 0;
  const header: Record<string, { dtype: "F32"; shape: number[]; data_offsets: [number, number] }> = {};
  for (const [name, tensor] of tensors) {
    const length = tensor.values.length * Float32Array.BYTES_PER_ELEMENT;
    header[name] = { dtype: "F32", shape: [...tensor.shape], data_offsets: [offset, offset + length] };
    offset += length;
  }
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.length));
  const payload = Buffer.alloc(offset);
  let byteOffset = 0;
  for (const tensor of tensors.values()) {
    for (let index = 0; index < tensor.values.length; index += 1) payload.writeFloatLE(tensor.values[index]!, byteOffset + index * Float32Array.BYTES_PER_ELEMENT);
    byteOffset += tensor.values.length * Float32Array.BYTES_PER_ELEMENT;
  }
  await writeFile(file, Buffer.concat([prefix, headerBytes, payload]));
}

async function writeMixed16Safetensors(file: string): Promise<void> {
  const header = {
    "f16.weight": { dtype: "F16", shape: [6], data_offsets: [0, 12] },
    "bf16.weight": { dtype: "BF16", shape: [6], data_offsets: [12, 24] },
  };
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.length));
  const payload = Buffer.alloc(24);
  for (const [index, value] of [0x3c00, 0xc000, 0x0400, 0x7bff, 0x7c00, 0x7e00].entries()) {
    payload.writeUInt16LE(value, index * 2);
  }
  for (const [index, value] of [0x3f80, 0xc020, 0x3f00, 0x7f7f, 0xff80, 0x7fc1].entries()) {
    payload.writeUInt16LE(value, 12 + index * 2);
  }
  await writeFile(file, Buffer.concat([prefix, headerBytes, payload]));
}

async function write16Safetensors(
  file: string,
  tensors: ReadonlyMap<string, DenseTensor>,
  storageDtype: "F16" | "BF16",
): Promise<void> {
  let offset = 0;
  const header: Record<string, { dtype: "F16" | "BF16"; shape: number[]; data_offsets: [number, number] }> = {};
  for (const [name, tensor] of tensors) {
    const length = tensor.values.length * Uint16Array.BYTES_PER_ELEMENT;
    header[name] = { dtype: storageDtype, shape: [...tensor.shape], data_offsets: [offset, offset + length] };
    offset += length;
  }
  const headerBytes = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(headerBytes.length));
  const payload = Buffer.alloc(offset);
  let byteOffset = 0;
  for (const tensor of tensors.values()) {
    for (let index = 0; index < tensor.values.length; index += 1) {
      payload.writeUInt16LE(storageDtype === "F16" ? encodeFixtureF16(tensor.values[index]!) : encodeBF16(tensor.values[index]!), byteOffset + index * 2);
    }
    byteOffset += tensor.values.length * Uint16Array.BYTES_PER_ELEMENT;
  }
  await writeFile(file, Buffer.concat([prefix, headerBytes, payload]));
}

function encodeFixtureF16(value: number): number {
  const known = new Map<number, number>([[0, 0x0000], [1, 0x3c00], [3, 0x4200], [4, 0x4400]]);
  const encoded = known.get(value);
  if (encoded === undefined) throw new Error(`Fixture F16 não tem codificação para ${value}.`);
  return encoded;
}

function encodeBF16(value: number): number {
  const bytes = new ArrayBuffer(4);
  const view = new DataView(bytes);
  view.setFloat32(0, value, true);
  return view.getUint16(2, true);
}
