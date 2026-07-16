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

test("literal export fails closed instead of silently widening BF16 storage into its F32 replay contract", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "llm-inner-literal-bf16-"));
  const model = path.join(root, "model");
  try {
    await writeTinyF32Llama(model);
    const reader = new SafetensorsCatalogReader(model);
    const catalog = await reader.inspect();
    const ir = await buildModelIR(catalog, preview);
    catalog.tensors.get("model.embed_tokens.weight")!.storageDtype = "BF16";
    await assert.rejects(
      () => buildDenseF32LiteralProgram(ir, catalog, reader),
      /aceita somente tensor Safetensors F32 denso não quantizado/,
    );
    await reader.close();
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

async function writeTinyF32Llama(directory: string): Promise<void> {
  await mkdir(directory, { recursive: true });
  const weights: Array<[string, number[], number[]]> = [
    ["model.embed_tokens.weight", [3, 2], [0, 0, 3, 4, 0, 0]],
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
    const payload = Buffer.alloc(values.length * 4);
    values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
    header[name] = { dtype: "F32", shape, data_offsets: [offset, offset + payload.length] };
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
