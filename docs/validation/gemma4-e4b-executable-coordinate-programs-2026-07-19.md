# Gemma 4 E4B: executable coordinate programs — 2026-07-19

## Completed boundary

Schema v35 made every forward edge navigable in both directions, but its
`coordinates` and shape `axis` fields were still human-readable strings. A
reader had to parse expressions such as `head*256+head_feature`,
`floor(video_frame/frames)`, `0..patches-1` and
`STABLE_TRUE_PREFIX_RANK(input_ids==258880,batch,sequence)` before it could
follow an element.

Schema v36 embeds the parsed program beside every coordinate access:

- `tensor-element.coordinatePrograms` contains one AST per tensor axis;
- `tensor-shape.axisProgram` contains the executable shape-axis AST;
- `calculationGraph.coordinateLanguage` defines signed safe-integer bindings,
  arithmetic, inclusive ranges, stable-prefix selection, evaluation order and
  fail-closed errors; and
- formula-language schema 21 points coordinate authority to that embedded
  language and forbids reparsing the human `expression` field for execution.

The parser is shared by every Gemma 4 scope and call site. It does not dispatch
by operation, layer, dtype or shape. Unknown helpers, missing symbols/tensors,
ragged integer tensors, unsafe arithmetic, zero divisors and inverted ranges
fail closed. Tests independently execute nested indexing, floor division,
modulo, inclusive ranges and multimodal stable-prefix rank, and reject an
unregistered host helper.

## Real artifact and source-present storage audit

The real dense, unquantized package is `google/gemma-4-E4B` at immutable
revision `411aa17b749aa952df1359d2dcea73917a544d9a`. The artifact was regenerated
with:

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

- artifact schema 36, formula-language schema 21 and calculation-graph schema 4;
- 21,391,443,862 bytes, SHA-256
  `4b277bdcb7272923f0ebd94a9e99eb7a0b6685d157c2f4b42d8cb2cf06709483`;
- 2,130 constants and 15,992,314,836 learned payload bytes;
- 2,708 assignments, 3,491 predecessors and 3,361 producer edges;
- 36,495 coordinate programs, including seven inclusive ranges and six
  stable-prefix-rank programs; and
- 1,722 reduction domains, including exactly 100 runtime-defined BMMs.

The source-present audit compared every tensor by name, dtype, shape and bytes:

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --verify-gemma4-payloads \
  --output /tmp/gemma4-v36-source-audit.json
```

All 2,130 payloads matched. Source and literal storage both produced SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The audit report SHA-256 is
`e024884c0737c16f6737c0b1a3aaeef51a1e0eb81d1455984156ae452bcadb0a`.

## Source-removed validation

`./gemma-4-E4B-dense` was physically renamed under an EXIT/INT/TERM restore
trap. While that path was absent, the reader used only the literal JSON to:

- verify all 2,130 embedded payload commitments covering 15,992,314,836 bytes;
- list all 100 runtime-defined reductions and validate all 36,495 coordinate
  programs; report SHA-256
  `53ed6697d748a9439e196cefa7578d494cbcbe26163409757de3c1961909821f`;
- render four decoded BF16 learned scalars for a window of
  `layer_0_q_proj[0,0,0]`, with executable constant output coordinates; report
  SHA-256 `186a03e9586d1b6914065c82c172b9255e1a2e2ac108bacfdc6fb02ddfb5759d`;
- render all 64 addressable products of
  `composite_image_features/vision_layer_0_attention_scores[0,0,0,0]` while
  retaining `fail-closed-runtime-reduction`; report SHA-256
  `d70d49c4f70aca204a5830c83bd326391e0179a61f4288bb9c38f934fedb1c40`;
  and
- build the one-step end-to-end closure with 2,708 forward operations, 13
  generation operations, 2,076 reachable constants, 54 declared
  runtime-unreachable constants and exactly 100 fail-closed operations; report
  SHA-256 `bf9c754f28276cdeb1b62acf064c0525e3b41f21fbd701d9bcecfb9da0253ed1`.

The same source-removed shell reran the registered image, video and audio
authoritative suite against
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`.
All three prefills and all three greedy generations passed at zero absolute and
relative tolerance, with no first divergence and 24/24 caches per modality.
The report preserved 32 image, 32 video and 36 audio runtime-defined BMMs and
has SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.
The checkpoint directory was restored after every command completed.

Fresh repository validation after documentation and artifact regeneration:

- `npm run typecheck`: pass;
- `npm test`: 251 tests, 251 pass, 0 fail; and
- `git diff --check`: pass.

## Preserved fidelity limit

This boundary removes host parsing from coordinate navigation; it does not
invent Apple Accelerate SGEMM product rounding or accumulation order. The same
100 BMM assignments remain `fail-closed-runtime-reduction`, so this session
does not certify the complete dense-lossless objective.
