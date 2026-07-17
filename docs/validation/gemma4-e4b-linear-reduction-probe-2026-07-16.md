# Gemma 4 E4B literal linear reduction probe — 2026-07-16

This is a bounded diagnostic over the existing immutable dense E4B literal
artifact and the complete native operation trace. It does not reopen the
checkpoint and it does not register a mathematical policy automatically.

## Reproduction

```bash
npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop7-operation-trace.json \
  --trace /tmp/gemma4-e4b-loop7-repeat-operation-trace.json \
  --operation-id layer_0_gate_proj \
  --output /tmp/gemma4-e4b-gate-proj-reduction-probe.json \
  --max-read-mib 16 --lane-counts 2,4,8,16,32,64,128,256

npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop7-operation-trace.json \
  --trace /tmp/gemma4-e4b-loop7-repeat-operation-trace.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-up-proj-reduction-probe.json \
  --max-read-mib 16 --lane-counts 2,4,8,16,32,64,128,256 \
  --lane-reduction-orders ascending,descending,balanced-pairwise
```

The probe requires two distinct native captures with different per-capture
UUIDs and the same source checksums, runtime/reference identity, program
fingerprint, named linear producer and named result. It rejects even a single
bitwise difference before it reads a bounded literal matrix range. It then
obtains the linear input and expected result only from those traced assignments
and compares every output coordinate bitwise after the operation's declared
BF16 result cast. It reports no source checkpoint access and never mutates the
artifact or adapter.

## Result

For `layer_0_gate_proj`, the previously declared
`interleaved-f32-lanes-32-ascending` matches all 10,240 coordinates, but it is
not unique under the broadened probe: 16-lane balanced and the FMA-equivalent
forms also match this one BF16 input. It remains a trace-compatible candidate,
not a native-kernel identification. For `layer_0_up_proj`, no tested profile
is exact. The probe separately models a separately rounded F32 product
and a fused-multiply-add lane update, then folds lanes ascending, descending,
or by an explicit balanced pairwise tree. For the BF16 E4B operands, the FMA
and separately rounded candidates agree. The closest result is
`interleaved-f32-lanes-32-balanced-pairwise`, which misses one BF16 coordinate
(element `8354`) by `1.1920928955078125e-7`; it is still evidence against
registering that schedule.

The literal E4B replay therefore remains approximate beginning at
`layer_0_up_proj`. This probe makes the reduction-search boundary reproducible
and fail-closed, but it is not source-removed generation evidence and does not
justify a Gemma 4 checkpoint marker.

## Independent repeatability review

On 2026-07-16, two separate captures of the pinned source package (the same
prompt `[2]` at position `0`) with capture IDs
`7280aa6d-08af-419b-9438-11fb78a1f6d1` and
`a1dc0598-253d-4814-b3a8-6999554fef6c` agreed bitwise on all 1,229 declared
assignments. The source-removed literal operation replay was also rerun against
one of those captures: it remains `approximate`, with 71 exact assignments and
one exact producer-owned KV cache; the first divergence is still
`layer_0_up_proj`.

Under the two-trace acceptance contract, `layer_0_gate_proj` has four matching
candidates: `interleaved-f32-lanes-16-balanced-pairwise`,
`interleaved-fma-lanes-16-balanced-pairwise`,
`interleaved-f32-lanes-32-ascending`, and
`interleaved-fma-lanes-32-ascending`. `layer_0_up_proj` has no matching
candidate among ordered F32/F64, 2, 4, 8,
16, 32, 64, 128 and 256 lanes, three explicit fold orders, and separate
F32-product/FMA lane semantics. Its closest profile misses one BF16 coordinate.
This deterministic mismatch remains a fail-closed semantic boundary, not a
schedule that may be registered from proximity.

## Fresh independent review — 2026-07-17

Two fresh captures from the local immutable E4B package (input token `[2]`,
position `0`) were created in this review and agreed bitwise before the probe
ran. The source-removed paged operation comparison against one capture reports
`sourceCheckpointAccessed=false`, 71 exact assignments, one exact producer KV
cache, and `layer_0_up_proj` as the first divergence. It remains
`approximate`; this document records a candidate validation result, not a
Gemma 4 checkpoint acceptance.
