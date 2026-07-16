# Gemma 4 E4B literal linear reduction probe — 2026-07-16

This is a bounded diagnostic over the existing immutable dense E4B literal
artifact and the complete native operation trace. It does not reopen the
checkpoint and it does not register a mathematical policy automatically.

## Reproduction

```bash
npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop7-operation-trace.json \
  --operation-id layer_0_gate_proj \
  --output /tmp/gemma4-e4b-gate-proj-reduction-probe.json \
  --max-read-mib 16 --lane-counts 2,4,8,16,32,64,128,256

npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop7-operation-trace.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-up-proj-reduction-probe.json \
  --max-read-mib 16 --lane-counts 2,4,8,16,32,64,128,256
```

The probe validates the trace's IR fingerprint before it reads a bounded
literal matrix range. It obtains the linear input only from the traced named
producer, obtains the expected result only from the same native trace, and
compares every output coordinate bitwise after the operation's declared BF16
result cast. It reports no source checkpoint access and never mutates the
artifact or adapter.

## Result

For `layer_0_gate_proj`, the only exact tested profile is the already declared
`interleaved-f32-lanes-32`; this independently accepts loop 7's candidate
schedule on all 10,240 coordinates. For `layer_0_up_proj`, no tested profile
is exact. The closest tested profile is four interleaved F32 lanes, but it
still mismatches 3 coordinates with maximum absolute error
`4.76837158203125e-7`; that is evidence against registering it as a native
schedule. Ordered F64 mismatches 4 coordinates (maximum absolute error
`3.814697265625e-6`).

The literal E4B replay therefore remains approximate beginning at
`layer_0_up_proj`. This probe makes the reduction-search boundary reproducible
and fail-closed, but it is not source-removed generation evidence and does not
justify a Gemma 4 checkpoint marker.
