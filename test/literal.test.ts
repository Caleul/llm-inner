import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import test from "node:test";
import { buildModelIR } from "../src/architecture.js";
import { compileModel } from "../src/compiler.js";
import { executeLiteralF32, buildDenseF32LiteralProgram, generateLiteralF32, validateLiteralCalculationProgram } from "../src/literal.js";
import type { LiteralCalculationProgram } from "../src/literal.js";
import { executeReferenceF32, generateReferenceF32 } from "../src/executor.js";
import { materializeReferenceF32Constants } from "../src/materialize.js";
import { SafetensorsCatalogReader } from "../src/safetensors.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("dense F32 Safetensors becomes a source-independent literal program that replays after checkpoint removal", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-literal-"));
  const model = path.join(root, "model");
  try {
    await writeTinyF32Llama(model);
    const artifact = path.join(root, "tiny.literal.json");
    await compileModel({ source: model, output: artifact, preview, literal: true });
    const program = JSON.parse(await readFile(artifact, "utf8")) as LiteralCalculationProgram;

    const serialized = JSON.stringify(program);
    assert.equal(serialized.includes(model), false);
    assert.equal(serialized.includes(".safetensors"), false);
    assert.equal(program.constants.length, 12);
    assert.equal(program.constants.every((constant) => constant.encoding === "base64" && constant.byteOrder === "little-endian"), true);
    assert.equal(program.stateTransitions.length, 1);
    assert.deepEqual(program.stateTransitions[0], {
      id: "layer_0_attention_kv_cache", layer: 0, operation: "append-post-rope",
      keyInput: "layer_0_k_rot", valueInput: "layer_0_v_heads", cacheOutput: "past_key_values.0",
    });

    await rm(model, { recursive: true, force: true });
    const replay = executeLiteralF32(program, { inputIds: [[1]] });
    assert.deepEqual(replay.logits.shape, [1, 1, 3]);
    assert.deepEqual([...replay.logits.values], [0.8485280871391296, 1.1313707828521729, 1.9798989295959473]);
    assert.deepEqual(replay.pastKeyValues.get(0)?.key.shape, [1, 1, 1, 2]);
    const generation = generateLiteralF32(program, { inputIds: [[1]], maxNewTokens: 2 });
    assert.deepEqual(generation.generatedTokenIds, [2, 2]);
    assert.equal(generation.stepPastKeyValues.length, 2);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("literal program validation rejects an assignment that reads an undeclared predecessor", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-literal-invalid-"));
  const model = path.join(root, "model");
  try {
    await writeTinyF32Llama(model);
    const reader = new SafetensorsCatalogReader(model);
    const catalog = await reader.inspect();
    const program = await buildDenseF32LiteralProgram(await buildModelIR(catalog, preview), catalog, reader);
    await reader.close();
    const corrupted = structuredClone(program);
    const firstLayerLinear = corrupted.assignments.layers[0]!.operations.find((operation) => operation.id === "layer_0_q_proj");
    if (!firstLayerLinear || firstLayerLinear.op !== "linear") throw new Error("fixture did not produce q projection");
    firstLayerLinear.input = "hidden_state_from_nowhere";
    assert.throws(() => validateLiteralCalculationProgram(corrupted), /não foi declarada antes do uso/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("literal F16 and BF16 programs retain stored bytes, declare IEEE decoders, and replay after source removal", async () => {
  for (const dtype of ["F16", "BF16"] as const) {
    const root = await mkdtemp(path.join(tmpdir(), `llm-inner-literal-${dtype.toLowerCase()}-`));
    const model = path.join(root, "model");
    const pairedF32Model = path.join(root, "paired-f32-model");
    try {
      await writeTinyDenseLlama(model, dtype, true);
      await writeTinyDenseLlama(pairedF32Model, "F32", true);
      const artifact = path.join(root, "tiny.literal.json");
      const pairedArtifact = path.join(root, "paired.literal.json");
      await compileModel({ source: model, output: artifact, preview, literal: true });
      await compileModel({ source: pairedF32Model, output: pairedArtifact, preview, literal: true });
      const program = JSON.parse(await readFile(artifact, "utf8")) as LiteralCalculationProgram;
      const pairedF32Program = JSON.parse(await readFile(pairedArtifact, "utf8")) as LiteralCalculationProgram;

      assert.equal(program.constants.every((constant) => constant.storageDtype === dtype), true);
      assert.equal(program.constants.every((constant) => Buffer.from(constant.payloadBase64, "base64").length === constant.storageShape.reduce((size, dimension) => size * dimension, 1) * 2), true);
      assert.equal(program.storageDecoders.every((decoder) => decoder.operation === (dtype === "F16" ? "ieee-f16-to-f32" : "ieee-bf16-to-f32")), true);

      await rm(model, { recursive: true, force: true });
      await rm(pairedF32Model, { recursive: true, force: true });
      const replay = executeLiteralF32(program, { inputIds: [[1]] });
      const generation = generateLiteralF32(program, { inputIds: [[1]], maxNewTokens: 2 });
      const pairedReplay = executeLiteralF32(pairedF32Program, { inputIds: [[1]] });
      const pairedGeneration = generateLiteralF32(pairedF32Program, { inputIds: [[1]], maxNewTokens: 2 });
      const embeddingOutput = program.assignments.prelude[0]?.output;
      assert.ok(embeddingOutput);
      assert.deepEqual([...replay.values.get(embeddingOutput)!.values], [...pairedReplay.values.get(embeddingOutput)!.values]);
      assert.deepEqual([...replay.logits.values], [...pairedReplay.logits.values]);
      assert.deepEqual(generation.generatedTokenIds, pairedGeneration.generatedTokenIds);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  }
});

test("literal validation rejects an altered decoder instead of implicitly widening storage", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-literal-decoder-"));
  const model = path.join(root, "model");
  try {
    await writeTinyDenseLlama(model, "BF16");
    const reader = new SafetensorsCatalogReader(model);
    const catalog = await reader.inspect();
    const ir = await buildModelIR(catalog, preview);
    const program = await buildDenseF32LiteralProgram(ir, catalog, reader);
    await reader.close();
    const corrupted = structuredClone(program);
    corrupted.storageDecoders[0]!.operation = "ieee-f16-to-f32";
    assert.throws(() => validateLiteralCalculationProgram(corrupted), /decoder de storage literal não corresponde/);
    const missing = structuredClone(program) as unknown as { storageDecoders?: LiteralCalculationProgram["storageDecoders"] };
    delete missing.storageDecoders;
    assert.throws(() => validateLiteralCalculationProgram(missing as unknown as LiteralCalculationProgram), /não declara decoders de storage/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("literal MLX affine U32 embeds codes and BF16 parameters, then replays after source removal", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-literal-mlx-affine-"));
  const model = path.join(root, "mlx-affine");
  try {
    await writeTinyMlxAffineLlama(model);
    const reader = new SafetensorsCatalogReader(model);
    const catalog = await reader.inspect();
    const ir = await buildModelIR(catalog, preview);
    forceF32Policy(ir);
    const expectedTensors = await materializeReferenceF32Constants(ir, catalog, reader);
    const expected = executeReferenceF32(ir, { inputIds: [[1]], tensors: expectedTensors });
    const expectedGeneration = generateReferenceF32(ir, { inputIds: [[1]], tensors: expectedTensors, maxNewTokens: 2 });
    await reader.close();

    const artifact = path.join(root, "tiny.mlx.literal.json");
    await compileModel({ source: model, output: artifact, preview, literal: true });
    const program = JSON.parse(await readFile(artifact, "utf8")) as LiteralCalculationProgram;
    assert.equal(program.sourceFormat, "mlx-safetensors");
    const affine = program.storageDecoders.filter((decoder) => decoder.operation === "mlx-affine-u32-to-f32");
    assert.equal(affine.length, 9);
    assert.equal(affine.every((decoder) => decoder.bits === 3 && decoder.groupSize === 4 && decoder.parameterDtype === "BF16"), true);
    assert.equal(program.constants.filter((constant) => constant.storageDtype === "U32").length, 9);
    assert.equal(JSON.stringify(program).includes(model), false);
    assert.equal(JSON.stringify(program).includes(".safetensors"), false);

    const corrupted = structuredClone(program);
    const firstAffine = corrupted.storageDecoders.find((decoder) => decoder.operation === "mlx-affine-u32-to-f32");
    if (!firstAffine || firstAffine.operation !== "mlx-affine-u32-to-f32") throw new Error("fixture did not emit MLX affine decoder");
    firstAffine.packing = "row-major-contiguous-lsb-first-u32x" as never;
    assert.throws(() => validateLiteralCalculationProgram(corrupted), /decoder MLX affine literal não corresponde/);

    await rm(model, { recursive: true, force: true });
    const replay = executeLiteralF32(program, { inputIds: [[1]] });
    const generation = generateLiteralF32(program, { inputIds: [[1]], maxNewTokens: 2 });
    assert.deepEqual([...replay.logits.values], [...expected.logits.values]);
    assert.deepEqual(generation.generatedTokenIds, expectedGeneration.generatedTokenIds);
    assert.deepEqual([...generation.logits.values], [...expectedGeneration.logits.values]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

async function writeTinyF32Llama(directory: string): Promise<void> {
  await writeTinyDenseLlama(directory, "F32");
}

async function writeTinyDenseLlama(directory: string, dtype: "F32" | "F16" | "BF16", fractional = false): Promise<void> {
  await mkdir(directory, { recursive: true });
  const weights: Array<[string, number[], number[]]> = [
    ["model.embed_tokens.weight", [3, 2], [0, 0, ...(fractional ? [3.5, 4.5] : [3, 4]), 0, 0]],
    ["model.layers.0.input_layernorm.weight", [2], [1, 1]],
    ["model.layers.0.self_attn.q_proj.weight", [2, 2], [1, 0, 0, 1]],
    ["model.layers.0.self_attn.k_proj.weight", [2, 2], [1, 0, 0, 1]],
    ["model.layers.0.self_attn.v_proj.weight", [2, 2], [1, 0, 0, 1]],
    ["model.layers.0.self_attn.o_proj.weight", [2, 2], [1, 0, 0, 1]],
    ["model.layers.0.post_attention_layernorm.weight", [2], [1, 1]],
    ["model.layers.0.mlp.gate_proj.weight", [2, 2], [0, 0, 0, 0]],
    ["model.layers.0.mlp.up_proj.weight", [2, 2], [0, 0, 0, 0]],
    ["model.layers.0.mlp.down_proj.weight", [2, 2], [0, 0, 0, 0]],
    ["model.norm.weight", [2], [1, 1]],
    ["lm_head.weight", [3, 2], [1, 0, 0, 1, 1, 1]],
  ];
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = weights.map(([name, shape, values]) => {
    const payload = encodeDensePayload(dtype, values);
    header[name] = { dtype, shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encodedHeader = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(encodedHeader.length));
  await writeFile(path.join(directory, "config.json"), JSON.stringify({
    model_type: "llama", hidden_size: 2, intermediate_size: 2, num_hidden_layers: 1,
    num_attention_heads: 1, num_key_value_heads: 1, head_dim: 2, vocab_size: 3,
    rms_norm_eps: 1e-6, hidden_act: "silu",
  }));
  await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, encodedHeader, ...payloads]));
}

function encodeDensePayload(dtype: "F32" | "F16" | "BF16", values: readonly number[]): Buffer {
  const payload = Buffer.alloc(values.length * (dtype === "F32" ? 4 : 2));
  values.forEach((value, index) => {
    if (dtype === "F32") payload.writeFloatLE(value, index * 4);
    else if (dtype === "F16") payload.writeUInt16LE(encodeF16(value), index * 2);
    else payload.writeUInt16LE(float32Bits(value) >>> 16, index * 2);
  });
  return payload;
}

function encodeF16(value: number): number {
  const bits = float32Bits(value);
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const fraction = bits & 0x7fffff;
  if (exponent === 0xff) return sign | (fraction === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const mantissa = fraction | 0x800000;
    const shift = 14 - halfExponent;
    const rounded = (mantissa + (1 << (shift - 1)) - ((mantissa >>> shift) & 1)) >>> shift;
    return sign | rounded;
  }
  const rounded = fraction + 0x0fff + ((fraction >>> 13) & 1);
  if ((rounded & 0x800000) !== 0) return sign | ((halfExponent + 1) << 10);
  return sign | (halfExponent << 10) | (rounded >>> 13);
}

function float32Bits(value: number): number {
  const view = new DataView(new ArrayBuffer(4));
  view.setFloat32(0, value, true);
  return view.getUint32(0, true);
}

/** Uses 3-bit codes so each row contains codes that straddle U32 boundaries. */
async function writeTinyMlxAffineLlama(directory: string): Promise<void> {
  await mkdir(directory, { recursive: true });
  const width = 32;
  const groupSize = 4;
  const bits = 3;
  const groups = width / groupSize;
  const quantizedNames = [
    "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight", "lm_head.weight",
  ];
  const tensors: Array<[string, "F32" | "BF16" | "U32", number[], number[]]> = [];
  for (const [tensorIndex, name] of quantizedNames.entries()) {
    const codes = Array.from({ length: width * width }, (_, index) => (index * 5 + tensorIndex * 3 + Math.floor(index / width)) & 7);
    const scales = Array.from({ length: width * groups }, (_, index) => 0.03125 * (1 + ((index + tensorIndex) % 4)));
    const biases = Array.from({ length: width * groups }, (_, index) => -0.25 + 0.0625 * ((index + tensorIndex) % 5));
    const module = name.slice(0, -".weight".length);
    tensors.push(
      [name, "U32", [width, width * bits / 32], packMlxCodes(codes, width, width, bits)],
      [`${module}.scales`, "BF16", [width, groups], scales],
      [`${module}.biases`, "BF16", [width, groups], biases],
    );
  }
  for (const name of ["model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight", "model.norm.weight"]) {
    tensors.push([name, "F32", [width], new Array<number>(width).fill(1)]);
  }
  await writeFile(path.join(directory, "config.json"), JSON.stringify({
    model_type: "llama", hidden_size: width, intermediate_size: width, num_hidden_layers: 1,
    num_attention_heads: 1, num_key_value_heads: 1, head_dim: width, vocab_size: width,
    rms_norm_eps: 1e-6, hidden_act: "silu", quantization: { bits, group_size: groupSize, mode: "affine" },
  }));
  await writeSafetensorsFixture(path.join(directory, "model.safetensors"), tensors);
}

function packMlxCodes(codes: readonly number[], rows: number, columns: number, bits: number): number[] {
  assert.equal(codes.length, rows * columns);
  assert.equal(columns * bits % 32, 0);
  const wordsPerRow = columns * bits / 32;
  const packed = new Array<number>(rows * wordsPerRow).fill(0);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const bitOffset = column * bits;
      const wordIndex = row * wordsPerRow + Math.floor(bitOffset / 32);
      const shift = bitOffset % 32;
      const code = codes[row * columns + column]!;
      packed[wordIndex] = (packed[wordIndex]! | (code << shift)) >>> 0;
      if (shift + bits > 32) packed[wordIndex + 1] = (packed[wordIndex + 1]! | (code >>> (32 - shift))) >>> 0;
    }
  }
  return packed;
}

async function writeSafetensorsFixture(
  file: string,
  tensors: Array<[string, "F32" | "BF16" | "U32", number[], number[]]>,
): Promise<void> {
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = tensors.map(([name, dtype, shape, values]) => {
    const payload = Buffer.alloc(values.length * (dtype === "BF16" ? 2 : 4));
    values.forEach((value, index) => {
      if (dtype === "F32") payload.writeFloatLE(value, index * 4);
      else if (dtype === "BF16") payload.writeUInt16LE(float32Bits(value) >>> 16, index * 2);
      else payload.writeUInt32LE(value, index * 4);
    });
    header[name] = { dtype, shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encodedHeader = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(encodedHeader.length));
  await writeFile(file, Buffer.concat([prefix, encodedHeader, ...payloads]));
}

function forceF32Policy(ir: Awaited<ReturnType<typeof buildModelIR>>): void {
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    operation.dtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
}
