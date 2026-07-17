# Gemma 4 E4B ARM vector-reduction boundary — 2026-07-17

This validation adds a literal, finite representation of PyTorch's ARM BF16
vector-dot candidate without treating source availability as proof that the
installed wheel dispatched to that path. The schedule declares every F32 FMA
lane, the eight-register tree, and the final horizontal fold. It is a
source-independent calculation schedule; it never invokes PyTorch or an ARM
kernel while replaying an artifact.

The candidate follows the pinned PyTorch source at commit
[`7269437d655783a26cba32aa88195b741ff496aa`](https://github.com/pytorch/pytorch/tree/7269437d655783a26cba32aa88195b741ff496aa):
`BFloat16` transposed GEMV reaches `bf16_dot_with_fp32_arith`, whose
`VectorizedN<float, 8>` register tree first adds registers `0+4` through
`3+7`, then `0+2` and `1+3`. The executable IR form supports both finite
widths exposed by the build abstractions: 32 lanes (8 registers × 4 lanes) and
64 lanes (8 × 8), with either an ascending or pairwise final horizontal fold.
Malformed widths, lane mappings, or folds are rejected by the literal writer,
probe, and paged replay boundary.

## Fresh immutable evidence

Two passive-hook CPU captures were freshly produced from public
`google/gemma-4-E4B` revision `411aa17b749aa952df1359d2dcea73917a544d9a`.
Their source identity is `config.json` SHA-256
`f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20` and dense
`model.safetensors` SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
Both captured the named canonical boundary
`layer_0_pre_ffn_norm -> layer_0_up_proj`, with BF16 input/output/parameter and
the row-major layouts recorded in the preceding loop.

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-loop29-arm-tree-a.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-loop29-arm-tree-b.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu
```

The literal candidate ran with the entire checkpoint directory renamed and an
exit trap restoring it. It read only
`artifacts/gemma4-e4b-dense.literal.json` and the two traces:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop29-arm-tree-a.json \
  --trace /tmp/gemma4-loop29-arm-tree-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-loop29-full-profile-source-removed.json \
  --max-read-mib 16 --lane-counts 2,4,8,16,32,64,128 \
  --lane-reduction-orders ascending,descending,balanced-pairwise \
  --tiled-lane-counts 2,4,8,16,32 --tiled-terms-per-lane 2,4,8 \
  --blocked-terms-per-block 2,4,8,16,32 \
  --blocked-tiled-lane-counts 2,4,8,16 \
  --blocked-tiled-terms-per-lane 2,4,8
```

The report has `sourceCheckpointAccessed: false`, `traceCount: 2`, 231 tested
profiles, and an empty `exactProfileIds`. The four ARM-vector candidates were
not close: the 32-lane variants mismatch 10,218 output coordinates (maximum
absolute error 10.1875), and the 64-lane variants mismatch 10,229 (maximum
10.75). Therefore the source-level ARM vector path is not silently installed
as the E4B artifact's semantic contract.

The strongest remaining profile is unchanged:
`interleaved-f32-lanes-32-balanced-pairwise` (also its FMA variant) misses one
BF16 output coordinate, `8354`, by `1.1920928955078125e-7`. This proves the
new operation family is executable and auditable, while preserving the actual
blocker: the installed PyTorch eager-BF16 CPU dispatch is not yet expressed by
an exact source-independent literal reduction schedule.
