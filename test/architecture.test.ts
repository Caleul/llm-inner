import test from "node:test";
import assert from "node:assert/strict";
import { buildModelIR } from "../src/architecture.js";
import type { JsonObject, ModelCatalog, TensorInfo } from "../src/types.js";

function tensor(name: string, shape: number[]): TensorInfo {
  return {
    name,
    storageDtype: "F32",
    storageShape: [...shape],
    logicalShape: [...shape],
  };
}

function catalog(modelType: string, extras: JsonObject = {}): ModelCatalog {
  const tensors = new Map<string, TensorInfo>();
  const add = (name: string, shape: number[]) => tensors.set(name, tensor(name, shape));
  add("model.embed_tokens.weight", [32, 4]);
  add("model.layers.0.input_layernorm.weight", [4]);
  add("model.layers.0.self_attn.q_proj.weight", [4, 4]);
  add("model.layers.0.self_attn.q_proj.bias", [4]);
  add("model.layers.0.self_attn.k_proj.weight", [4, 4]);
  add("model.layers.0.self_attn.v_proj.weight", [4, 4]);
  add("model.layers.0.self_attn.o_proj.weight", [4, 4]);
  add("model.layers.0.post_attention_layernorm.weight", [4]);
  add("model.layers.0.mlp.gate_proj.weight", [8, 4]);
  add("model.layers.0.mlp.up_proj.weight", [8, 4]);
  add("model.layers.0.mlp.down_proj.weight", [4, 8]);
  add("model.norm.weight", [4]);
  add("lm_head.weight", [32, 4]);
  return {
    source: "/tmp/model",
    format: "safetensors",
    config: {
      model_type: modelType,
      hidden_size: 4,
      intermediate_size: 8,
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 4,
      vocab_size: 32,
      rms_norm_eps: 1e-6,
      ...extras,
    },
    rawMetadata: {},
    tensors,
  };
}

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;

test("Llama interpreta post_attention_layernorm como norma pré-MLP", async () => {
  const ir = await buildModelIR(catalog("llama"), preview);
  const ids = ir.layers[0]!.operations.map((operation) => operation.id);
  assert.equal(ids.includes("layer_0_post_attention_norm"), false);
  assert.ok(ids.indexOf("layer_0_pre_ffn_norm") > ids.indexOf("layer_0_attention_residual"));
  const q = ir.layers[0]!.operations.find((operation) => operation.id === "layer_0_q_proj");
  assert.equal(q?.op, "linear");
  if (q?.op === "linear") assert.equal(q.bias?.name, "model.layers.0.self_attn.q_proj.bias");
});

test("Gemma 2 preserva quatro normas do bloco", async () => {
  const source = catalog("gemma2", {
    query_pre_attn_scalar: 4,
    attn_logit_softcapping: 50,
  });
  source.tensors.set(
    "model.layers.0.pre_feedforward_layernorm.weight",
    tensor("model.layers.0.pre_feedforward_layernorm.weight", [4]),
  );
  source.tensors.set(
    "model.layers.0.post_feedforward_layernorm.weight",
    tensor("model.layers.0.post_feedforward_layernorm.weight", [4]),
  );
  const ir = await buildModelIR(source, preview);
  const ids = ir.layers[0]!.operations.map((operation) => operation.id);
  assert.ok(ids.indexOf("layer_0_post_attention_norm") < ids.indexOf("layer_0_attention_residual"));
  assert.ok(ids.indexOf("layer_0_pre_ffn_norm") > ids.indexOf("layer_0_attention_residual"));
  assert.ok(ids.indexOf("layer_0_post_ffn_norm") < ids.indexOf("layer_0_mlp_residual"));
});

test("Gemma 4 falha fechado até existir adaptador específico", async () => {
  await assert.rejects(
    () => buildModelIR(catalog("gemma4_text"), preview),
    /não possui adaptador exato/,
  );
});
