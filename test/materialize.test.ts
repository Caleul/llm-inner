import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { buildModelIR } from "../src/architecture.js";
import { materializeReferenceF32Constants } from "../src/materialize.js";
import { SafetensorsCatalogReader } from "../src/safetensors.js";
import type { DenseF32Tensor, ModelCatalog, ModelIR, TensorInfo } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

function tensor(name: string, shape: number[], quantization?: TensorInfo["quantization"]): TensorInfo {
  return { name, storageDtype: "F32", storageShape: [...shape], logicalShape: [...shape], ...(quantization ? { quantization } : {}) };
}

function catalog(): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  for (const [name, shape] of [
    ["model.embed_tokens.weight", [3, 2]],
    ["model.layers.0.input_layernorm.weight", [2]],
    ["model.layers.0.self_attn.q_proj.weight", [2, 2]],
    ["model.layers.0.self_attn.k_proj.weight", [2, 2]],
    ["model.layers.0.self_attn.v_proj.weight", [2, 2]],
    ["model.layers.0.self_attn.o_proj.weight", [2, 2]],
    ["model.layers.0.post_attention_layernorm.weight", [2]],
    ["model.layers.0.mlp.gate_proj.weight", [4, 2]],
    ["model.layers.0.mlp.up_proj.weight", [4, 2]],
    ["model.layers.0.mlp.down_proj.weight", [2, 4]],
    ["model.norm.weight", [2]],
    ["lm_head.weight", [3, 2]],
  ] as Array<[string, number[]]>) tensors.set(name, tensor(name, shape));
  return {
    source: "/tmp/fixture", format: "safetensors", rawMetadata: {}, tensors,
    config: { model_type: "llama", hidden_size: 2, intermediate_size: 4, num_hidden_layers: 1, num_attention_heads: 1, num_key_value_heads: 1, head_dim: 2, vocab_size: 3, rms_norm_eps: 1e-6 },
  };
}

test("materializer reads every unique dense F32 IR constant once and rejects stale references", async () => {
  const ir = await buildModelIR(catalog(), preview);
  const calls: string[] = [];
  const reader = {
    async readDenseAsF32(info: TensorInfo): Promise<DenseF32Tensor> {
      calls.push(info.name);
      return { shape: [...info.logicalShape], values: new Float32Array(info.logicalShape.reduce((a, b) => a * b, 1)) };
    },
  };
  const constants = await materializeReferenceF32Constants(ir, catalog(), reader);
  assert.equal(constants.size, 12);
  assert.equal(new Set(calls).size, 12);

  const stale = structuredClone(ir) as ModelIR;
  const embedding = stale.prelude[0];
  assert.equal(embedding?.op, "embedding");
  if (embedding?.op === "embedding") embedding.weight.shape = [99, 2];
  await assert.rejects(() => materializeReferenceF32Constants(stale, catalog(), reader), /diverge do catálogo/);
});

test("materializer requires the declared MLX bridge and preserves its provenance", async () => {
  const source = catalog();
  const quantization = { family: "mlx" as const, mode: "affine", bits: 4, groupSize: 2, scaleTensor: "model.embed_tokens.scales" };
  const embedding = source.tensors.get("model.embed_tokens.weight")!;
  embedding.storageDtype = "U32";
  embedding.storageShape = [3, 1];
  embedding.quantization = quantization;
  source.format = "mlx-safetensors";
  const ir = await buildModelIR(source, preview);
  const reader = { async readDenseAsF32(info: TensorInfo): Promise<DenseF32Tensor> { return { shape: [...info.logicalShape], values: new Float32Array(info.logicalShape.reduce((a, b) => a * b, 1)) }; } };
  await assert.rejects(() => materializeReferenceF32Constants(ir, source, reader), /requer TensorBridge/);
  const constants = await materializeReferenceF32Constants(ir, source, reader, {
    async readMlxDequantizedF32(_catalog, name) {
      assert.equal(name, "model.embed_tokens.weight");
      return { shape: [3, 2], values: new Float32Array(6), sourceQuantization: { ...quantization } };
    },
  });
  assert.deepEqual(constants.get("model.embed_tokens.weight")?.sourceQuantization, quantization);
});

test("temporary Safetensors source materializes binary32 values through the catalog range reader", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "llm-inner-f32-"));
  try {
    const header = Buffer.from(JSON.stringify({ weight: { dtype: "F32", shape: [2, 2], data_offsets: [0, 16] } }), "utf8");
    const prefix = Buffer.alloc(8);
    prefix.writeBigUInt64LE(BigInt(header.length));
    const values = Buffer.alloc(16);
    [1.25, -2.5, 3.75, 4.5].forEach((value, index) => values.writeFloatLE(value, index * 4));
    await writeFile(path.join(directory, "model.safetensors"), Buffer.concat([prefix, header, values]));
    await writeFile(path.join(directory, "config.json"), "{}\n");
    const reader = new SafetensorsCatalogReader(directory);
    try {
      const source = await reader.inspect();
      const ir: ModelIR = {
        schemaVersion: 2, source: { path: directory, format: "safetensors" }, architecture: { modelType: "llama", hiddenSize: 2, numLayers: 0, numAttentionHeads: 1, numKeyValueHeads: 1, headDim: 2 }, config: {}, preview,
        inputs: [], prelude: [{ id: "token_embedding", op: "embedding", tokenInput: "input_ids", output: "x", weight: { name: "weight", shape: [2, 2], storageDtype: "F32" }, dtypePolicy: {} }], layers: [], epilogue: [], fidelity: { exactByConstruction: false, assumptions: [], unsupported: [], warnings: [] },
      };
      const constants = await materializeReferenceF32Constants(ir, source, reader);
      assert.deepEqual([...constants.get("weight")!.values], [1.25, -2.5, 3.75, 4.5]);
    } finally {
      await reader.close();
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
