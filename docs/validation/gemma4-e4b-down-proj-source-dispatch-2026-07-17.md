# Gemma 4 E4B layer-0 down projection source-dispatch review — 2026-07-17

This review closes the former `layer_0_down_proj` reduction ambiguity for the
pinned CPU runtime. It does **not** accept the dense-lossless checkpoint: the
literal replay is still approximate from a later operation.

## Established native path

The public PyTorch source at build commit
`7269437d655783a26cba32aa88195b741ff496aa` routes this BF16, bias-free,
transposed `Linear` shape through `addmm_impl_cpu_`, then the BF16 transposed
GEMV fast path (`bf16_gemv_trans`), which calls
`bf16_dot_with_fp32_arith` for each output row. The installed
`libtorch_cpu.dylib` has the same commit in `torch.__config__.show()`.

The local ARM64 disassembly of that exact function is decisive for this wheel:
the 32-term loop widens BF16 values with `shll`, accumulates eight four-lane
F32 registers with `fmla`, applies the `0+4`, `1+5`, `2+6`, `3+7`, then
`0+2`, `1+3`, then `0+1` register tree, and horizontally folds with two
`faddp` instructions. It contains no `bfdot` instruction. The machine reports
`hw.optional.arm.FEAT_BF16: 1`, but hardware capability alone is not treated
as proof of a compiled BFDOT branch.

Accordingly the literal assignment now declares the existing explicit
`arm-neon-bf16-dot-fma` schedule with 32 lanes, eight four-lane registers, and
`horizontalFold: "pairwise"`. This is bound only to the complete registered
E4B topology and `layer_0_down_proj`; replay does not infer it from shape.

## Falsified adjacent branch

The probe now models PyTorch's distinct BFDOT branch explicitly as
`arm-neon-bf16-bfdot-fma`: four active registers, each lane consuming two
contiguous BF16 products. It is not an alias for a modulo-lane reduction.

With the checkpoint directory absent, the new profile was replayed against two
independent eight-token down-projection captures:

```bash
npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop35-down-long-a.json \
  --trace /tmp/gemma4-loop35-down-long-b.json \
  --operation-id layer_0_down_proj \
  --output /tmp/gemma4-loop36-down-bfdot-source-removed-selection.json \
  --max-read-mib 16 --lane-counts 32 --min-distinct-inputs 1 \
  --profile-id arm-neon-bf16-bfdot-fma-32-ascending \
  --profile-id arm-neon-bf16-bfdot-fma-32-pairwise
```

The report has `sourceCheckpointAccessed: false`, no exact profile, and seven
mismatched BF16 elements for either horizontal fold (first element `499`,
maximum absolute error `0.0009765625`). This independently rejects the BFDOT
semantic branch rather than selecting it from the CPU's advertised feature.

## Regenerated source-removed replay

The full 21,326,357,515-byte literal artifact was regenerated atomically from
the immutable dense package and then used while `gemma-4-E4B-dense` was
renamed away. A fresh authoritative assignment trace was captured after the
IR fingerprint change and compared with zero tolerance:

```bash
npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop36-current-operation-checkpoints.json \
  --input-tokens 2 --position-ids 0 --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

node dist/src/gemma4-paged-text-operation-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop36-current-operation-checkpoints.json \
  --report /tmp/gemma4-loop36-source-removed-operation-report.json \
  --max-read-mib 16 --top-k 10 --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

The report records `sourceCheckpointAccessed: false`, 121 exact assignments
(up from 73 before this binding), and first divergence at `layer_1_o_proj`
with maximum absolute error `0.00000762939453125`. `layer_0_down_proj` is
therefore no longer the first divergent operation. The source-removed greedy
comparison still emits token `184` at position `1`, but remains `approximate`:
selection-logit maximum absolute error is `0.1875` and KV cache values diverge.
No checkpoint marker is created.
