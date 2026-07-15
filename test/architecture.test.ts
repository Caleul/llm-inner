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

test("Qwen 3 aplica Q/K/V RMSNorm por cabeça após reshape", async () => {
  const source = catalog("qwen3", { hidden_act: "silu" });
  source.tensors.set(
    "model.layers.0.self_attn.q_norm.weight",
    tensor("model.layers.0.self_attn.q_norm.weight", [4]),
  );
  source.tensors.set(
    "model.layers.0.self_attn.k_norm.weight",
    tensor("model.layers.0.self_attn.k_norm.weight", [4]),
  );
  source.tensors.set(
    "model.layers.0.self_attn.v_norm.weight",
    tensor("model.layers.0.self_attn.v_norm.weight", [4]),
  );
  const operations = (await buildModelIR(source, preview)).layers[0]!.operations;
  const indexOf = (id: string) => operations.findIndex((operation) => operation.id === id);

  assert.ok(indexOf("layer_0_q_heads") < indexOf("layer_0_q_norm"));
  assert.ok(indexOf("layer_0_q_norm") < indexOf("layer_0_q_rope"));
  assert.ok(indexOf("layer_0_k_heads") < indexOf("layer_0_k_norm"));
  assert.ok(indexOf("layer_0_k_norm") < indexOf("layer_0_k_rope"));
  assert.ok(indexOf("layer_0_v_heads") < indexOf("layer_0_v_norm"));
  assert.ok(indexOf("layer_0_v_norm") < indexOf("layer_0_attention"));
  const qNorm = operations.find((operation) => operation.id === "layer_0_q_norm");
  assert.equal(qNorm?.op, "rms_norm");
  if (qNorm?.op === "rms_norm") assert.equal(qNorm.input, "layer_0_q_heads");
  const attention = operations.find((operation) => operation.id === "layer_0_attention");
  assert.equal(attention?.op, "scaled_dot_product_attention");
  if (attention?.op === "scaled_dot_product_attention") assert.equal(attention.value, "layer_0_v_norm");
});

test("Q/K RMSNorm com dimensão diferente de head_dim falha fechado", async () => {
  const source = catalog("qwen3", { hidden_act: "silu" });
  source.tensors.set(
    "model.layers.0.self_attn.q_norm.weight",
    tensor("model.layers.0.self_attn.q_norm.weight", [2]),
  );
  source.tensors.set(
    "model.layers.0.self_attn.k_norm.weight",
    tensor("model.layers.0.self_attn.k_norm.weight", [2]),
  );
  await assert.rejects(
    () => buildModelIR(source, preview),
    /head_dim=4/,
  );
});

test("projeção K com dimensão incompatível falha antes de gerar IR", async () => {
  const source = catalog("llama");
  source.tensors.set(
    "model.layers.0.self_attn.k_proj.weight",
    tensor("model.layers.0.self_attn.k_proj.weight", [3, 4]),
  );
  await assert.rejects(() => buildModelIR(source, preview), /k_proj da camada 0 deve ter shape 4x4/);
});

test("MLP gated com dimensão intermediária incompatível falha fechado", async () => {
  const source = catalog("llama");
  source.tensors.set(
    "model.layers.0.mlp.down_proj.weight",
    tensor("model.layers.0.mlp.down_proj.weight", [4, 7]),
  );
  await assert.rejects(() => buildModelIR(source, preview), /down_proj da camada 0 deve ter shape 4x8/);
});

test("embedding e norma final precisam respeitar hidden_size e vocab_size", async () => {
  const badEmbedding = catalog("llama");
  badEmbedding.tensors.set("model.embed_tokens.weight", tensor("model.embed_tokens.weight", [31, 4]));
  await assert.rejects(() => buildModelIR(badEmbedding, preview), /vocab=31.*vocab_size=32/);

  const badNorm = catalog("llama");
  badNorm.tensors.set("model.norm.weight", tensor("model.norm.weight", [3]));
  await assert.rejects(() => buildModelIR(badNorm, preview), /norma final deve ter shape 4/);
});
