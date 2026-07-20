# Gemma 4 E4B: closed coordinate bindings — 2026-07-19

## Completed boundary

Schema v36 made coordinate navigation executable, but its symbol node could
still read any scalar name supplied by the caller. Live inspection of the real
graph found free extent aliases and axis-name mismatches in complete operation
classes: video frame flattening, masked scatter, per-layer reshape/embedding,
vision head reshape/softmax/attention/pooling, audio convolution reshape and
attention, and text per-layer reshape/select.

Schema v37 closes that dependency boundary generically:

- coordinate-language schema 2 adds `tensor-axis(tensor, axis)` for explicit
  reads such as `pixel_values_videos.shape[1]`;
- calculation-graph schema 5 validates every output and predecessor coordinate
  AST against the owning output axes, reduction domains, preceding scalar or
  `STRUCT` locals, and declared call-site tensors;
- formula-language schema 22 states that free host extent aliases are invalid;
- repeated fixed dimensions are lowered from the Gemma 4 contract, while
  dynamic extents are read from the declared tensor shape; and
- the same parser, evaluator and closure validator apply to every operation,
  shape and call site. Missing axes, undeclared tensors, invalid dimensions and
  free symbols fail closed.

This boundary does not add a per-layer binding table. It removes the undeclared
bindings from the formulas that created them and makes graph construction reject
their recurrence.

## Real artifact and source-present storage audit

Source: dense unquantized `google/gemma-4-E4B`, immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a`.

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

- artifact schema 37, formula-language schema 22, calculation-graph schema 5
  and coordinate-language schema 2;
- 21,391,466,420 bytes, SHA-256
  `2d2129ab31f56f0a4733c9970990b9cdb944f6a8ee3dfbc9fae1c0d382619008`;
- 2,130 constants/decoders and 15,992,314,836 embedded learned bytes;
- 2,708 assignments, 3,491 predecessors and 3,361 producer edges;
- 36,505 coordinate programs, with 14 nested tensor-axis nodes, seven inclusive
  ranges and six stable-prefix-rank nodes; and
- 1,462 reductions, 256 stages and 1,722 domains, including exactly 100
  runtime-defined BMMs.

The source-present audit used:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --verify-gemma4-payloads \
  --output /tmp/gemma4-v37-source-audit.json
```

All 2,130 payloads matched by name, dtype, shape and bytes. Source and literal
storage both produced SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The audit report SHA-256 is
`ef2302d841f6ce390c8bd61fcf4b4299fbec0c0f0b343fe2bf27c6512f9a7678`,
and `forbiddenSourcePathPresent` is false.

## Source-removed replay and literal substitution

`./gemma-4-E4B-dense` was physically renamed under an EXIT/INT/TERM restore
trap. While that exact path was absent, the reader used only the 21 GB JSON.

Payload verification covered all 2,130 constants and 15,992,314,836 bytes,
reported `sourceCheckpointAccessed=false`, retained all 100 runtime reductions
and produced SHA-256
`5bd2c1eff2c2a6b0cd22a86fc06c6a2a36c00caedf6b54aeb406b0306f42660d`.

The scalar audit command was:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --operation layer_0_q_proj --output-coordinate 0,0,0 \
  --input-start 0 --input-count 4 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-v37-source-removed-q-view.json
```

It rendered four addressable q-projection terms and substituted each learned
operand with storage bits, decoded F32 bits and a numeric literal. The first is
BF16 `0xbbcb`, decoded as F32 bits `0xbbcb0000` and literal
`-0.006195068359375`; its product formula is
`exact_product(layer_0_attn_norm[0,0,0] * -0.006195068359375)`. The report has
SHA-256 `bd19178bdd97e776b0ee76c87eb7b0a395cb4f8f382000e44ed841d6d12224b7`
and records `sourceCheckpointAccessed=false`.

The one-token end-to-end view contains 2,708 forward operations, 13 generation
operations, 42 cache transitions, 2,076 reachable constants and 54 declared
runtime-unreachable constants. It records `sourceCheckpointAccessed=false`,
keeps exactly 100 operations `fail-closed-runtime-reduction`, and has SHA-256
`f35008c6fcf3354cdb62419e150430ead803194ce556a8e87109be15d7fea41a`.

Still without the source, the registered authoritative image, video and audio
suite passed all three prefills and all three greedy generations at zero
absolute and relative tolerance. `firstDivergence` is null; the result retains
32 image, 32 video and 36 audio runtime-defined BMMs. Its byte-stable SHA-256 is
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.
The source directory was restored and both hidden temporary paths were absent
after validation.

## Repository validation and preserved fidelity limit

```text
npm run typecheck  # exit 0
npm test           # 251/251, exit 0
git diff --check   # exit 0
```

Tests execute tensor-axis parsing/evaluation, accept a closed dynamic extent,
reject a free `frames` alias, reject an undeclared tensor and reject an invalid
shape axis. Building the complete fixture graph independently checks closure for
all operation classes.

The 100 Apple Accelerate SGEMM BMM assignments remain
`fail-closed-runtime-reduction`: this boundary closes coordinate bindings but
does not claim an unobserved native scalar product and accumulation schedule.
