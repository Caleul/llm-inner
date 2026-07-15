import test from "node:test";
import assert from "node:assert/strict";
import { buildModelIR } from "../src/architecture.js";
import { executeReferenceF64 } from "../src/executor.js";
import type { DenseTensor, ModelCatalog, TensorInfo } from "../src/types.js";

const preview = { outputRows: 1, inputTerms: 1, includeWeights: false } as const;
const f64Policy = { computeDtype: "F64", accumulationDtype: "F64", outputDtype: "F64" } as const;

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

test("executor F64 fails closed for compiler-default implicit dtype policies", async () => {
  const { ir, weights } = await tinyLlama();
  ir.prelude[0]!.dtypePolicy = { computeDtype: "model-configured" };
  assert.throws(
    () => executeReferenceF64(ir, { inputIds: [[1]], tensors: weights }),
    /política F64 explícita/,
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
