# Gemma 4 E4B: authoritative native-BMM operand traces — 2026-07-20

## Completed boundary

The schema-v43 artifact can derive every product address entering its 100
Apple Accelerate BMM reductions from the serialized scalar AST. The previous
authoritative vision trace, however, captured Q/K immediately after
normalization while the native score BMM consumes Q/K after multidimensional
RoPE. A reduction probe could therefore compare the native output only by
reconstructing a mathematically later operand outside the trace.

Vision/audio checkpoint traces now use schema v2. Every trace contains:

- a UUID v4 `captureId`;
- SHA-256 identities for `config.json` and `model.safetensors`;
- the native output of each `runtime-defined` BMM;
- both ordered operand tensors at the exact kernel boundary; and
- `runtimeReductionCoverage`, derived from the Gemma 4 program rather than an
  operation-ID allowlist.

The vision helper captures `q_rotated` and repeated `k_rotated` inside the
pinned `eager_attention_forward`, immediately before `torch.matmul`. The
coverage builder follows each BMM input to its declared producer and requires
the producer ID, output name and positive shape to match the trace. The
source-removed comparator rebuilds the coverage against the artifact's
embedded vision/audio program and verifies both checkpoint hashes against
`sourceIdentity` before executing a tower.

Missing post-RoPE operands, substitution of an earlier Q/K checkpoint,
duplicate operation IDs, output drift, shape mutation, malformed capture UUID,
and mismatched checkpoint identity fail closed. The same operation-, input-
and shape-driven contract covers all vision and audio layers.

## Real E4B captures

Both captures used:

- model `google/gemma-4-E4B`;
- immutable revision `411aa17b749aa952df1359d2dcea73917a544d9a`;
- dense unquantized BF16 Safetensors;
- `transformers-5.5.0/torch-2.12.1` CPU eager inference mode; and
- artifact `artifacts/gemma4-e4b-dense.literal.json`, schema 43.

The video trace used two frames, nine patches per frame and width 768. It
contains 326 operation checkpoints and 32 complete native-reduction coverage
entries: score and value BMM for every one of 16 layers. Its size is
132,984,944 bytes and SHA-256 is
`4c5e7c4e82ebd878113e0b5e4940ce923fb0d81278d305d5b00c35ab4c4ab4a9`.

The one-frame audio trace contains 444 operation checkpoints and 36 complete
coverage entries: AC, BD and value BMM for every one of 12 layers. Its size is
18,417,067 bytes and SHA-256 is
`322676b07272914b6c83128322c3acc745246fc0c08eda13e6caa372cb6a80ad`.

## Source-removed validation

For each comparison, `./gemma-4-E4B-dense` was moved to
`/private/tmp/gemma-4-E4B-dense.source-hidden-codex-v44` under an
`EXIT`/`INT`/`TERM` restore trap. The comparator required the declared path to
be absent. Both reports record `sourceCheckpointAccessed=false`, and the source
directory was restored after each command.

The video replay compared all 326 checkpoints at zero absolute and relative
tolerance. All 326 passed, `firstDivergentOperation=null`, and fidelity is
`lossless-within-dtype`. The report embeds the 32 accepted coverage entries and
source identities; its 539,908-byte SHA-256 is
`f9654d8a31232a272a8589adcdcb53ef6b76f9327b38d0e8e437804b8b4eed2f`.

The audio replay compared all 444 checkpoints at zero tolerance: 385 pass and
the first divergence is `audio_layer_0_attention_position_scores`. Its 48/48
source-anchored post-BMM attention stages and 12/12 context casts remain
lossless within dtype. The overall classification correctly remains
`approximate`. The report embeds the 36 accepted coverage entries and source
identities; its 274,712-byte SHA-256 is
`da00799b1e0023b8293a738a743fbfdc19ae8002f61cfd955205e775892e007b`.

## Preserved fidelity boundary

This change closes authoritative operand identity, not Apple Accelerate's
unpublished scalar schedule. Probing the exact two-frame vision operands showed
that many candidate F32 reductions collapse to the same BF16 outputs, so the
capture does not uniquely identify product rounding, lanes, tiles or fold
order. The artifact still declares 100 `fail-closed-runtime-reduction`
assignments and `exactReplayClaim=forbidden`.
