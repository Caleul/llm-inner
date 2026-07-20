# Gemma 4 E4B: attested source-removed native-reduction replay — 2026-07-20

## Completed boundary

Schema v44 could route every remaining vision/audio BMM through the pinned
PyTorch CPU runtime, but the helper response asserted only a contract ID. A
different Torch build, platform or BLAS backend could therefore present the
same ID, and differential reports did not bind each invocation to its exact
operands and output.

Schema v45 closes that proof boundary:

- the artifact integrity contract fixes `torch-2.12.1`, build commit
  `7269437d655783a26cba32aa88195b741ff496aa`, `torch.inference_mode`, CPU,
  `Darwin-arm64`, Apple Accelerate SGEMM and `BLAS_INFO=accelerate`;
- the Python adapter verifies every runtime property before reading an
  operation request;
- the TypeScript boundary accepts only an operation ID and operation class
  matching one unique serialized `runtime-defined` assignment;
- each response echoes the operation identity and carries the runtime
  attestation;
- each execution receipt commits both ordered operands and the output by
  shape, byte count and SHA-256 over IEEE-F32 little-endian bytes; and
- composite differential comparison rejects a receipt count different from
  the active graph's runtime-defined reduction count.

This is operation-, shape-, runtime- and source-contract driven. It has no
layer-specific allowlist and covers all five Gemma 4 native-BMM classes.

## Real artifact and source-present audit

The real dense, unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 45.

- artifact bytes: `21,409,278,550`;
- artifact SHA-256:
  `46f99fa94d5cd06e73ad72ce2e1a6152377e72cb577810dc98238a9b3c85a157`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes; and
- integrity root:
  `7c30fc8d70a6c563ed30edb2d65246963ed7234949549b9b0a1ea3ae78ac97b3`.

The source-audit report is 1,012 bytes with SHA-256
`1577abf2322875e4852e2073852538f41066995db0e9e3e0ba981c02dacd0093`.

## Source physically unavailable

`./gemma-4-E4B-dense` was moved to the explicit temporary path
`/private/tmp/gemma-4-E4B-dense.source-hidden-v45` under an
`EXIT`/`INT`/`TERM` restoration trap. While the declared source path was
absent, the reader rehashed all 2,130 embedded payloads and recorded
`sourceCheckpointAccessed=false`. The source-removed integrity report is
43,030,644 bytes with SHA-256
`39cd6e7e5938bd8a6e4f28b1c2199a34c967c29852c483e82c185913d1102974`.

In the same source-hidden window, the composite suite compared the literal
program with the pinned authoritative Transformers 5.5.0 / Torch 2.12.1 eager
runtime at zero absolute and relative tolerance:

| modality | prefill | generation | native reductions | attested receipts |
| --- | --- | --- | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | 32 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | 32 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | 36 | 36 |

All three prefill comparisons and all three greedy generations passed;
`firstDivergence=null`. Every receipt carries the same verified Torch build,
platform and Accelerate attestation, and commits its actual operand/output
bytes. The 357,665-byte suite report has SHA-256
`2d262302c62a788a3cb82b5343b44922045adbe33ce5ac876fb29509769f85be`.
The trap restored the source directory and removed the temporary path.

## Preserved fidelity boundary

The artifact now reproduces all 100 opaque BMM values through the exact pinned
authoritative runtime without checkpoint access and records what executed.
Apple Accelerate still does not publish its scalar product-rounding, lane,
tile and fold schedule. Consequently the mathematical scalar view remains
`fail-closed-runtime-reduction`, the gate remains
`blocked-on-runtime-reduction`, and `exactReplayClaim=forbidden`. Runtime
replay evidence is not misrepresented as a published scalar reduction tree.
