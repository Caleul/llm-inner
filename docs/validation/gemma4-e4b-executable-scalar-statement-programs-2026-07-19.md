# Gemma 4 E4B: executable scalar statement programs — 2026-07-19

## Completed boundary

Schema v38 made every local scalar producer, consumer and coordinate
navigable, but arithmetic still lived only in `scalarAssignments` strings.
Schema v39 embeds one closed syntax tree per statement:

- scalar-calculation schema 5 adds `statementPrograms` with exact ordinal,
  source, targets and expression;
- formula-language schema 24 declares the tagged node union and evaluation
  rules;
- calculation-graph schema 7 binds identifiers structurally at every image,
  video and audio call site; and
- artifact schema 39 rejects a missing, altered or unparsable statement tree.

The parser dispatches only on formula syntax. It does not contain operation,
layer, assignment, shape or dtype tables. It recognizes numeric/string/BOOL
literals, identifiers, arrays, unary and binary operators, conditionals,
calls, tensor indexing, members, inclusive ranges, named arguments, filtered
domains, ordered loops and structured composite invocation. Unknown tokens,
trailing text, malformed targets, ambiguous loops and invalid invocation
queries fail closed.

Structural call-site binding is material: an instantiated identifier such as
`composite_image_features/vision_layer_0_q_rotated` remains one tensor name in
the AST rather than becoming a division expression. Parsing every operation
class also exposed unbalanced parentheses in the audio relative-position
formula. That formula now constructs its shared BF16 angle explicitly and
feeds the same expression to the selected SLEEF sine/cosine branch.

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

- artifact schema 39, formula-language schema 24, calculation-graph schema 7
  and scalar-calculation schema 5;
- 21,406,708,755 bytes, SHA-256
  `942a33a61a7380bf8e0f12a916fc47dc183c7b678706e78e73e3b066b19a35d5`;
- 2,130 constants/decoders and 15,992,314,836 embedded learned bytes;
- 2,708 instantiated assignments and 4,144 scalar statements/programs;
- 118,780 expression nodes, including 1,718 reduction/dot/FMA calls, two
  ordered loops and 64 filtered domains; and
- 1,722 executable reduction domains plus exactly 100 runtime-defined BMMs.

The source-present audit command was:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --verify-gemma4-payloads \
  --output /tmp/gemma4-v39-source-audit.json
```

All 2,130 payloads matched by name, dtype, shape and bytes. Source and literal
storage both produced SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The audit report SHA-256 is
`a1a6df78b6216b84ef3db42e453b5428047c9dffa7f3f8217513b571ed6f3909`,
and `forbiddenSourcePathPresent` is false.

## Source-removed replay and literal substitution

`./gemma-4-E4B-dense` was physically renamed under an EXIT/INT/TERM restore
trap. While the declared path was absent, the schema-39 reader used only the
literal JSON to verify every payload, list runtime reductions and open the
one-step end-to-end forward/greedy closure:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads --list-runtime-reductions \
  --end-to-end-calculation --generation-max-new-tokens 1 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-v39-source-removed.json
```

The 107,108,026-byte report has SHA-256
`a407b662057a356fca1868e25d0dd45261fae5ce8a779875719ee10d7cfdc7f0` and
records `sourceCheckpointAccessed=false`. It contains 4,144 statement programs
and 118,780 expression nodes alongside the existing 4,144 statement-dataflow
entries.

A source-removed four-term window for `layer_0_q_proj[0,0,0]` decoded BF16
words `0xbbcb`, `0x3af2`, `0xbd4c`, `0x3d08` to exact literals
`-0.006195068359375`, `0.0018463134765625`, `-0.0498046875`, and
`0.033203125`. Each product embeds its literal and contains no `weight[...]`
placeholder. The report SHA-256 is
`ba2eef0901024d11abce425f39ec362f39c0877f542429a542695d1874ae8759`.

## Authoritative differential and preserved fidelity limit

With the source still absent, the registered image, video and audio suite
compared schema 39 at zero absolute and relative tolerance against
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`.
All three prefills and all three greedy generations passed
`lossless-within-dtype`; every modality generated token 184 at position 2,
matched 24 KV caches and had `firstDivergence=null`. The report retains 32
image, 32 video and 36 audio runtime-defined BMMs and has SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

The source directory was restored and the temporary hidden path was absent
after the trap completed.

Fresh repository validation after implementation, documentation and artifact
regeneration:

```text
npm run typecheck  # exit 0
npm test           # 253/253, exit 0
git diff --check   # exit 0
```

The 100 Apple Accelerate SGEMM BMM assignments remain
`fail-closed-runtime-reduction`. Executable statement syntax closes the
artifact-side parsing gap; it does not invent unpublished product rounding,
lane, tile, FMA or accumulation semantics.
