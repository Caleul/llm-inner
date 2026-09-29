import { readFile, writeFile, stat } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { compileFixedTwoTokenModel, evaluateFixedTwoTokenModel } from "../dist/src/fixed-f16-two-token-model.js";
import { compileFixedLogitAudit } from "../dist/src/fixed-f16-logit-audit.js";
import { SafetensorsCatalogReader } from "../dist/src/safetensors.js";

const fixture = JSON.parse(await readFile("test/fixtures/tiny-random-llama-two-token-attention-stages.json", "utf8"));
const reader = new SafetensorsCatalogReader("artifacts/tiny-random-llama");
const startCompile = process.hrtime.bigint();
const model = await compileFixedTwoTokenModel(reader, fixture["0"].cos[0], fixture["0"].sin[0]);
const compilationMs = Number(process.hrtime.bigint() - startCompile) / 1e6;
await reader.close();
for (let i = 0; i < 3; i++) evaluateFixedTwoTokenModel(model, [1, 2]);
const samples = [];
let output;
for (let i = 0; i < 20; i++) {
  const start = process.hrtime.bigint();
  output = evaluateFixedTwoTokenModel(model, [1, 2]);
  samples.push(Number(process.hrtime.bigint() - start) / 1e6);
}
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const python = spawnSync("venv/bin/python", ["helpers/benchmark_fixed_llama_forward.py"], { encoding: "utf8" });
if (python.status !== 0) throw new Error(python.stderr);
const reference = JSON.parse(python.stdout);
if (output.tokens[1] !== reference.next_token || output.tokens[1] !== reference.generated_token) {
  throw new Error("Próximo token difere do forward ou generate PyTorch.");
}
const counts = {};
for (const layer of model.layers) for (const node of layer.attention.nodes) counts[node.op] = (counts[node.op] ?? 0) + 1;
const projectionTerms = (program) => program.rows.reduce((sum, row) => sum + row.terms.length, 0);
const mlpTermsPerToken = model.layers.reduce((sum, layer) =>
  sum + [layer.mlp.gate, layer.mlp.up, layer.mlp.down].reduce((part, projection) => part + projectionTerms(projection), 0), 0);
const logitAudit = compileFixedLogitAudit(model, 1, output.tokens[1]);
const result = {
  checkpoint_bytes: (await stat("artifacts/tiny-random-llama/model.safetensors")).size,
  compiled_json_bytes: Buffer.byteLength(JSON.stringify(model)),
  one_logit_audit_json_bytes: Buffer.byteLength(JSON.stringify(logitAudit)),
  one_logit_audit_nodes: logitAudit.nodes.length,
  attention_unique_nodes: model.layers.map((layer) => layer.attention.nodes.length),
  attention_node_ops: counts,
  mlp_projection_terms: model.layers.map((layer) => [layer.mlp.gate, layer.mlp.up, layer.mlp.down].map(projectionTerms)),
  lm_head_terms: model.vocabSize * 16,
  counted_arithmetic_per_two_token_forward: {
    attention_graph_binary_nodes: Object.entries(counts).filter(([op]) => op.startsWith("mul-") || op.startsWith("add-")).reduce((sum, [, count]) => sum + count, 0),
    mlp_projection_multiply_add_terms: mlpTermsPerToken * 2,
    lm_head_multiply_add_terms: model.vocabSize * 16 * 2,
    lm_head_pairwise_folds: model.vocabSize * 3 * 2,
    note: "Counts exclude RMSNorm, SiLU, softmax internals, and F16 conversions; multiply-add terms name a product plus accumulation.",
  },
  compile_ms: compilationMs,
  javascript: { runtime: "Node.js scalar F16", samples: samples.length, median_ms: median(samples), min_ms: Math.min(...samples), next_token: output.tokens[1] },
  reference,
};
await writeFile("FIXED-FORWARD-METRICS.json", JSON.stringify(result, null, 2) + "\n");
process.stdout.write(JSON.stringify(result, null, 2) + "\n");
