# Gemma 4 E4B authoritative execution contract

This is candidate schema-v16 artifact evidence from loop 72. It does not
accept the dense-lossless checkpoint: five native batched-matmul operation
classes, instantiated as 100 vision/audio operations, remain fail-closed and
the authoritative full-composite audio differential remains approximate.

## Independent starting review

Loop 72 started from commit `a078d63b6dacfc0981542e94afa9eccb00c0906c`
and independently inspected loop 71's diff. Before changing it,
`npm run typecheck` and `npm test` passed with 230/230 tests. This accepts the
candidate schema-v15 normalization/convolution transcripts as current
repository evidence; it does not accept the Gemma 4 checkpoint.

## General native-BMM investigation

The remaining runtime-defined operations are one compatible semantic class,
not individual layer defects: vision attention score/value and audio content
score, position score and value BMM. PyTorch 2.12.1 dispatches the relevant
F32 audio path through Apple Accelerate SGEMM, whose scalar reduction tree is
not published.

A class-wide empirical check evaluated all 1,344 nonzero authoritative audio
score reductions against ordered separate-product, ordered FMA, interleaved,
contiguous-block and pairwise scalar schedules. The best candidate, ordered
FMA, reproduced only 739/1,344 reductions and still had maximum absolute error
`0.0000457763671875`. No one schedule was therefore assigned to another layer
or operation. The five classes remain `unpublished-fail-closed`.

## Context defect and boundary closed

The previous direct audio helper used `torch.no_grad`, while the full composite
helper used `torch.inference_mode`. With identical `[1,1,128]` input values,
three direct captures were bitwise stable but their terminal `audio_features`
differed from the full-composite forward in 2,233/2,560 elements, maximum
absolute difference `0.08203125`. Switching only the direct helper to
`torch.inference_mode` preserved its prior result, so a direct tower invocation
cannot be used as acceptance evidence for the full composite context.

Schema v16 makes that distinction artifact data:

- `canonicalCompositeRuntime` fixes
  `transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`;
- direct audio and vision runtimes are explicitly diagnostic subprogram
  contexts;
- `executionMode` and eager attention are mandatory; and
- all five Apple Accelerate BMM classes are recorded with an unpublished,
  fail-closed scalar schedule.

The three capture helpers now use or declare `torch.inference_mode`. Capture,
trace parsing, literal comparison, in-memory validation and the streaming
reader all reject an older `CPU-eager` runtime or an altered artifact contract.
The inspection CLI exposes the contract to downstream readers.

## Real schema-v16 artifact and source-removed integrity

The real artifact was regenerated from immutable `google/gemma-4-E4B`
revision `411aa17b749aa952df1359d2dcea73917a544d9a`:

- path: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,384,482,408`;
- SHA-256: `f406ad69657f7d7ce58044212c70d38b3ceb04a3d34a9ca67883d006444dd2d3`;
- constants/storage decoders: `2,130` / `2,130`;
- embedded learned payload bytes: `15,992,314,836`;
- instantiated calculation assignments: `2,709` with 3,363 explicit
  predecessor edges.

With `gemma-4-E4B-dense` physically absent, the schema-v16 reader opened the
artifact and recomputed every embedded payload commitment. All 2,130 constants
and all `15,992,314,836` payload bytes passed; the literal-storage SHA-256 was
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`,
and the report recorded `sourceCheckpointAccessed=false`.

## Source-removed differential evidence

A fresh direct-audio inference-mode trace has SHA-256
`1d0b8c85cc755ff38db896f61c6083acea181a5f94d263827feee63b2e72bd48`.
Source-removed replay of schema v16 compared 444 operations at zero tolerance:
374 passed exactly, the first divergence remained
`audio_layer_0_attention_position_scores` with maximum absolute error
`0.000030517578125`, and terminal `audio_features [1,2560]` was exact. All 48
source-anchored score pipeline stages and all 12 F32-to-BF16 attention-context
casts were exact. The report SHA-256 is
`af91030816ac95865e52b761e96230c9b976978397ca26f2a2b254e918426997`.

A fresh image-tower inference-mode trace has SHA-256
`889bc0db09e271732b67cfda845f06641f24078fda2eae7127f2cecf18a4c87e`.
Its source-removed schema-v16 replay passed all 294/294 operations at zero
tolerance, including terminal `image_features [1,2560]`, and was classified
`lossless-within-dtype`. The report SHA-256 is
`37a882d7adaf5801be2b2a62f882ece495939727051ccae88a5ead5aca037f01`.

A fresh full-composite audio trace has SHA-256
`698f1b847e5e695b221fb0038c6d7802163f02a967637c6e8b726aff4d7c8952`.
Strict replay rejected the program before execution because its native fidelity
is unverified. With the explicit diagnostic-only
`--allow-unverified-fidelity` acknowledgement and the source still absent,
prefill and generation were both approximate: `composite_audio_features`
diverged by `0.08203125`, logits by `2`, top-10 overlap was `0.9`, all 24 step
and terminal caches diverged, while greedy token `184` at position `2` still
agreed. This token agreement is not fidelity acceptance. The report SHA-256 is
`dd4a2fd2f5928b373f2679b8f7a92ba40dc1cc91bdea168096a829ce98ebc7a1`.

Final repository validation passed `npm run typecheck` and `npm test` with 231
tests and zero failures. No
`.agent-loop/checkpoints/gemma4-dense-lossless/` marker was created.
