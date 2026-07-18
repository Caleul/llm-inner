# Gemma 4 E4B exact-product scalar audit

This is candidate artifact evidence from loop 70. It does not accept the
dense-lossless checkpoint: 100 instantiated native batched-matmul operations
remain fail-closed and the accepted audio forward/generation evidence remains
approximate.

## Independent starting review

The loop started from recovery commit
`dca9ff541ebbf5fcc878448d88e62d28c4575410`, whose unhanded-off schema-v13
work had replaced dense decoder equation strings with executable
`exact-safe-integer-expression-v1` and `u32-bit-expression-v1` ASTs. Before
changing it, loop 70 inspected the diff from `78aa25d`, ran `npm run
typecheck`, and passed 228/228 tests. It then opened the real schema-v13
artifact (`21,384,464,112` bytes), validated all 2,130 decoder declarations and
confirmed the embedded decoder-language contract. The schema-v13 claim is
therefore independently reviewed here, but it is superseded by schema v14
rather than promoted to the product checkpoint.

## Class-wide defect and boundary closed

The real scalar view for
`composite_image_features/vision_layer_0_q[0,0,0]` exposed a contradiction:

- the operation declared `arm-neon-bf16-dot-fma`, where each mathematical
  product remains exact until the F32 accumulator update;
- each displayed term was nevertheless materialized as
  `F32(input * literal_weight)`;
- the reduction then stopped substituting learned values and used
  `weight[0,i]` plus opaque `REGISTER_TREE`, `VECTOR_TAIL` and `SCALAR_TAIL`
  labels.

Schema v14 fixes the complete compatible linear class rather than one
assignment ID:

- `formulaLanguage` schema 2 declares `exact_product(a*b)` and permits it only
  under a fused operation-declared schedule;
- definition-level formulas use `exact_product` for every compatible FMA/dot
  reduction;
- text, vision and audio scalar views share one reduction transcript builder;
- the ARM transcript declares register/lane mapping, every F32 update, the
  0+4/1+5/2+6/3+7 register tree, pairwise or ascending horizontal fold,
  independent vector-tail accumulation, and ascending scalar tail;
- every complete audit defines `product[i]` in order with the exact decoded
  learned literal; no rendered audit may contain `weight[...]`, `bias[...]` or
  `decode(...)`;
- a fail-closed validator rejects symbolic learned references, decoded
  literals absent from the transcript, and complete reductions with missing or
  out-of-order terms.

A class scan of the real artifact found 594 linear/clipped-linear definitions
and 707 instantiated linear assignments. All 594 definitions use a fused
schedule and all 594 carry `exact_product`; zero retained the prematurely
rounded formula.

## Real artifact identity

Generation command:

```bash
npm run build && node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Result:

- schema: 14;
- artifact bytes: `21,384,477,385`;
- artifact SHA-256:
  `7a6cc6d3389bb5a37fb5c37dc3c42c0de0a9877339aa199a20523372e055d61f`;
- constants/storage decoders: `2,130` / `2,130`;
- embedded learned payload bytes: `15,992,314,836`.

The immutable package remains `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`. The source-present audit command
was:

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop70-source-audit.json \
  --verify-gemma4-payloads
```

It recomputed six immutable files totaling `16,024,773,810` bytes. Every
learned payload matched by name, dtype, shape and byte content; source and
literal concatenated storage SHA-256 were both
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

## Source-removed proof

The checkpoint directory was physically moved under an EXIT trap. With
`./gemma-4-E4B-dense` absent, `inspect:gemma4-literal --verify-payloads
--assert-source-unavailable` verified all 2,130 constants and all
`15,992,314,836` embedded bytes, again producing storage SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

Three complete, non-windowed scalar audits then read only the artifact:

| operation | schedule | terms | learned literals | scalar assignments |
| --- | --- | ---: | ---: | ---: |
| `layer_0_q_proj[0,0,0]` | `arm-neon-bf16-dot-fma` | 2,560 | 2,560 | 2,578 |
| `composite_image_features/vision_layer_0_q[0,0,0]` | `arm-neon-bf16-dot-fma` | 768 | 772 | 786 |
| `composite_audio_features/audio_layer_0_relative_k_projection[0,0,0]` | `arm-neon-bf16-dot-fma` | 1,024 | 1,024 | 1,042 |

All three reported `sourceCheckpointAccessed=false`, `complete=true` and
`omittedTerms=0`. Every term used `exact_product` with a concrete decoded
literal. The complete transcripts contained no `weight[...]`, `decode(...)`,
`REGISTER_TREE`, `VECTOR_TAIL` or `SCALAR_TAIL` placeholder.

The source-removed end-to-end view for two greedy steps retained 2,709 forward
assignments, 20 generation assignments, 42 cache transitions and complete
storage partitioning: 2,076 reachable learned constants plus 54 explicitly
runtime-unreachable shared-KV locals equals all 2,130 embedded constants. Its
report SHA-256 was
`0a6e512dbfc3c984cd241dd413a8d03249674c87ebde2a6c45092109b0b01ade`.

## Validation and limits

Final repository validation passed `npm run typecheck` and `npm test` with
229 tests and zero failures. New tests distinguish rounded and exact products,
exercise a 43-term ARM reduction across the main 32-term region, one 8-term
vector tail and three scalar-tail terms, and reject a reintroduced symbolic
weight reference.

This change does not invent Apple Accelerate schedules. Vision score/value BMM
and audio content/position/value BMM remain `runtime-defined`, giving 100
fail-closed instantiated forward operations. Audio authoritative comparison
remains approximate. No
`.agent-loop/checkpoints/gemma4-dense-lossless/` marker was created.
