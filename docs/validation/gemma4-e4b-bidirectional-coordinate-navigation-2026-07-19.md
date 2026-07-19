# Gemma 4 E4B: bidirectional coordinate navigation — 2026-07-19

## Candidate boundary

Schema v35 makes every instantiated forward assignment navigable in both
directions without operation-, layer-, shape- or dtype-specific dispatch.
`outputCoordinate.write` identifies the unique scalar destination written by
the assignment. `consumerCoordinates` groups every downstream read by consumer
operation. Concrete scalar views and fail-closed BMM audits expose the rendered
destination as `renderedOutputCoordinate`.

The notation parser also lowers the registered layout-only form
`row_major_alias(x)[...]` to the exact source read `x[...]`. This moved reshape
dependencies from opaque whole-value reads to addressable tensor elements
without changing any formula or mathematical dimension.

## Real artifact and storage integrity

Source: `google/gemma-4-E4B`, immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, dense unquantized BF16/F32
Safetensors. The artifact was regenerated with:

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Result:

- artifact schema `35`, formula-language schema `20`, calculation-graph schema `3`;
- 21,389,760,728 bytes, SHA-256
  `9f7afe7ca94ba239a7c891aadb782a480d68ff0f5c4c77ca63640904f4f2fcfe`;
- 2,130 embedded constants and 15,992,314,836 learned payload bytes;
- 2,708 assignments, 3,491 predecessors and 3,361 producer edges;
- 2,708 structured output writes and three output-shape assertions;
- 4,287 predecessor element reads, 55 shape reads and three whole-value reads;
- 4,200 downstream reads: 4,142 element, 55 shape and three whole-value.

The source-present audit command was:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --verify-gemma4-payloads \
  --output /private/tmp/llm-inner-loop93-v35-source-audit.json
```

All 2,130 constants matched the six source files byte-for-byte. Source and
literal storage both produced SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The audit report SHA-256 is
`41d19857c293d201d3dcfa06430161f1fad0de223bd8114be99a40325ad7a086`.

## Source-removed navigation

`./gemma-4-E4B-dense` was physically renamed under an EXIT/INT/TERM restore
trap. With that path absent, the reader reopened only the literal JSON and
produced:

- summary: `/private/tmp/llm-inner-loop93-v35-summary-final.json`, SHA-256
  `82f462be6321d6054aac492fb23fd1176b21a486d5c9848ed8e1f93a386c9e51`;
- concrete `layer_0_q_proj[0,0,0]` window: output write
  `layer_0_q_linear[0,0,0]`, predecessor reads
  `layer_0_attn_norm[0,0,0..1]`, and consumer read
  `layer_0_q_linear[batch,sequence,head*256+head_feature]`; report SHA-256
  `3a946a49ab6a016039dcf1c5ef0342a8cf93612124e4f72ccbcbe79d4b1ced29`;
- fail-closed vision-score BMM window: structured output
  `vision_layer_0_attention_scores[0,0,0,0]`, exact Q/K predecessor reads and
  the downstream softmax read; report SHA-256
  `32c7c62dd7687cdf67970c41f82fa802f76fef1e4d9296f271f344051ddd0d7e`;
- end-to-end view: 2,708 forward operations, 22 generation-control operations,
  2,076 reachable constants, 54 declared runtime-unreachable constants and
  exactly 100 fail-closed BMMs; report SHA-256
  `bac1311c43db2f61de04a02d9659d9dd2ecffd77a42b54713337e69342ae76d9`.

## Three-modality differential and preserved limit

With the source path still absent, the registered image, video and audio suite
passed 3/3 prefills and 3/3 greedy generations at zero absolute and relative
tolerance against
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`.
Every modality generated token `184`, had no first divergence and matched all
24 KV caches. Runtime-defined BMM counts remained 32 image, 32 video and 36
audio. Report SHA-256:
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

This is candidate navigation and source-removal evidence, not checkpoint
acceptance. Apple Accelerate SGEMM still has no authoritative scalar product
rounding and accumulation schedule for the 100 compatible BMM assignments.
Those assignments remain `fail-closed-runtime-reduction`; no
`.agent-loop/checkpoints/gemma4-dense-lossless/` marker was created.
