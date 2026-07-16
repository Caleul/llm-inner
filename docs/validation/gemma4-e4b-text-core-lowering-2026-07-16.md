# Gemma 4 E4B text-core lowering check — 2026-07-16

This is a structural lowering check against the immutable BF16 package audited
in `gemma4-e4b-source-audit-2026-07-16.md`. It is not a runtime differential,
literal export, source-removed replay, or Gemma 4 checkpoint claim.

The text config was supplied directly to the explicit `gemma4_text` adapter
while the catalog retained the registered `model.language_model` tensor names:

```bash
node --input-type=module -e 'import { SafetensorsCatalogReader } from "./dist/src/safetensors.js"; import { buildModelIR } from "./dist/src/architecture.js"; const reader = new SafetensorsCatalogReader("./gemma-4-E4B-dense"); try { const catalog = await reader.inspect(); const ir = await buildModelIR({ ...catalog, config: catalog.config.text_config }, { outputRows: 1, inputTerms: 1, includeWeights: false }); const shared = ir.layers.filter((layer) => layer.operations.some((operation) => operation.op === "scaled_dot_product_attention" && operation.kvSharing)).length; console.log(JSON.stringify({ modelType: ir.architecture.modelType, layers: ir.layers.length, prelude: ir.prelude.map((operation) => operation.id), sharedKvConsumers: shared, finalOutput: ir.epilogue.at(-1)?.output })); } finally { await reader.close(); }'
```

Observed output:

```json
{"modelType":"gemma4_text","layers":42,"prelude":["token_embedding","ple_token_identity","ple_context_projection","ple_context_scale","ple_context_reshape","ple_context_norm","ple_combine","ple_combine_scale"],"sharedKvConsumers":18,"finalOutput":"softcapped_logits"}
```

The composite `gemma4` package remains intentionally rejected. Image/audio
placeholder replacement and tower/projection execution must be lowered before
this textual subgraph can become the selected package's literal program.
