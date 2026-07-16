import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import * as path from "node:path";
import test from "node:test";
import { buildModelIR } from "../src/architecture.js";
import { compileModel } from "../src/compiler.js";
import { executeLiteralF32, buildDenseF32LiteralProgram, generateLiteralF32, validateLiteralCalculationProgram } from "../src/literal.js";
import type { LiteralCalculationProgram } from "../src/literal.js";
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
