# Gemma 4 E4B explicit learned non-linear reductions

This is candidate schema-v15 artifact evidence from loop 71. It does not
accept the dense-lossless checkpoint: 100 instantiated native batched-matmul
operations remain fail-closed and the audio operation differential remains
approximate even though its terminal feature vector is exact.

## Independent starting review

Loop 71 started from commit
`171f4ad2cbb83d0928982b96190f0b259a0a64e3` and independently inspected loop
70's exact-product diff. Before changing it, `npm run typecheck` passed and
`npm test` passed 229/229 tests. That accepts the schema-v14 claim that the
complete compatible linear class substitutes exact products and explicit ARM
main/vector/scalar-tail reductions. It does not accept the Gemma 4 checkpoint.

## Class-wide defect and boundary closed

The real artifact exposed three contradictions outside the already-corrected
linear class:

- native RMSNorm scalar views collapsed the declared PyTorch four-level F32
  cascade to `PYTORCH_CPU_F32_CASCADE_SUM(...)`;
- the two BF16 channel LayerNorms declared PyTorch's vector Welford schedule
  but rendered a different ordered two-pass mean/variance formula; and
- the two Conv2d plus twelve depthwise-convolution views declared four
  interleaved F32 lanes but rendered one ordered accumulator.

Schema v15 and formula-language schema 3 correct these whole compatible
classes:

- a shared normalization module serializes every cascade state transition:
  square coordinates, 16-coordinate units, four cascade levels, conditional
  level merges and clears, register fold, lane fold and final sum;
- the Welford transcript serializes low/high lane updates, reciprocals,
  low/high merge, count-weighted ascending lane fold, variance, sqrt,
  reciprocal, `bias=F32(-inv_std*mean)`, the source second pass
  `F32(F32(x*inv_std)+bias)`, gamma multiplication and output cast;
- Conv2d and depthwise views now emit every concrete learned product and run
  the serialized `interleaved-f32-lanes` schedule, including padding terms as
  `F32(F32(0)*literal)` rather than dropping the learned value; and
- the formula language embeds normative generic cascade and Welford programs.
  Rendered views fail closed on opaque normalization helpers, missing cascade
  state, incomplete Welford state, symbolic learned references or incomplete
  product terms.

## Real artifact identity and source-present integrity

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

- schema: 15;
- artifact bytes: `21,384,481,663`;
- artifact SHA-256:
  `0b8bfc1cc50511df7ad9afc6d8926efb9ac37d936093c7fe5087e4acaefda27c`;
- constants/storage decoders: `2,130` / `2,130`;
- embedded learned payload bytes: `15,992,314,836`.

The source-present audit used immutable `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`. It recomputed six package files
totaling `16,024,773,810` bytes and compared every constant by name, dtype,
shape and bytes. Source and literal storage SHA-256 were both
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

## Source-removed class proof

The checkpoint directory was physically moved under an EXIT trap. With
`./gemma-4-E4B-dense` absent, embedded-payload verification passed for all
2,130 constants and all `15,992,314,836` bytes with the same storage SHA-256.

One source-removed scan opened the artifact once and rendered a zero-coordinate
scalar view for every instantiated learned normalization/convolution operation:

| operation class | schedule | instantiated views |
| --- | --- | ---: |
| text `rms_norm` | `pytorch-cpu-f32-cascade-sum` | 301 |
| composite/vision/audio `rms-norm` | `pytorch-cpu-f32-cascade-sum` | 336 |
| audio `conv2d-stride2` | `interleaved-f32-lanes` | 2 |
| audio `layer-norm-channels` | `pytorch-cpu-bf16-welford` | 2 |
| audio `causal-depthwise-convolution` | `interleaved-f32-lanes` | 12 |

All 653 views passed the fail-closed scalar validator and decoded 1,801 learned
scalars from embedded storage. All 14 learned product reductions were complete
with zero omitted terms. No transcript contained `decode(...)`, `weight[...]`,
`bias[...]`, `PYTORCH_CPU_F32_CASCADE_SUM`, `mean_channels(...)` or
`variance_channels(...)`.

Representative source-removed views were also persisted under
`/private/tmp/llm-inner-loop71-*.json` during validation:

| operation | learned literals | scalar assignments | proof |
| --- | ---: | ---: | --- |
| `layer_0_input_norm[0,0,0]` | 1 | 15 | complete text cascade |
| `vision_layer_0_input_norm[0,0,0]` | 1 | 16 | complete vision cascade |
| `audio_layer_0_attn_pre_norm[0,0,0]` | 1 | 16 | complete audio cascade |
| `audio_subsample_0_conv[0,0,0,0]` | 9 | 15 | complete four-lane products |
| `audio_subsample_0_norm[0,0,0,0]` | 1 | 18 | complete BF16 Welford |
| `audio_layer_0_conv_depthwise[0,2,1]` | 5 | 11 | complete four-lane products |

The source-removed end-to-end view still contains 2,709 forward assignments,
20 generation assignments and 42 cache transitions. Storage partitioning is
complete: 2,076 reachable constants plus 54 explicitly runtime-unreachable
shared-KV locals equals all 2,130 embedded constants. The report SHA-256 is
`354d364a506d0282845ec35d8e3ede98162b4d4684bde588ec86c897b45a89e9`.

## Differential regression and limits

The schema-only/scalar-view changes do not alter the candidate executor. A
source-removed rerun against the preserved authoritative captures established:

- vision image: 294/294 operations passed at zero tolerance,
  `lossless-within-dtype`; terminal max absolute/relative error `0`, cosine
  `1`, top-k overlap `1`, argmax agreement true;
- audio: 374/444 operations passed at zero tolerance and the first divergence
  remains `audio_layer_0_attention_position_scores`, with max absolute error
  `0.000030517578125`, max relative error `3.5272472797868976e-7`, cosine
  `0.9999999999999906`, top-k overlap `1` and argmax agreement true; the
  terminal `audio_features [1,2560]` still has zero error and cosine `1`.

Final repository validation passed `npm run typecheck` and `npm test` with 230
tests and zero failures. This work does not invent Apple Accelerate BMM trees:
the 100 vision/audio native score/value operations remain
`fail-closed-runtime-reduction`. No
`.agent-loop/checkpoints/gemma4-dense-lossless/` marker was created.
