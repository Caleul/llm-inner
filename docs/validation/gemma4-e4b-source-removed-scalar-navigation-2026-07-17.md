# Gemma 4 E4B source-removed scalar navigation — 2026-07-17

This is candidate evidence for the literal scalar-substitution product
boundary. It does **not** accept the dense-lossless checkpoint: the current
authoritative forward/generation differential remains approximate.

## Implemented boundary

`inspect:gemma4-literal` now exposes the dependency-ordered Gemma4Text program
and can render one indexed scalar assignment directly from the embedded JSON
payload. The operation index records predecessors, consumers, ordinal and
previous/next IDs. Scalar views cover every registered Gemma4Text operation
shape and decode learned values for embeddings, per-layer embeddings, linear
projections, RMSNorm and learned tensor scales.

Each learned scalar records its tensor, row-major coordinate, storage bits,
storage dtype, decoder ID/operation, exact widened F32 value and round-tripping
numeric literal. Linear views also carry complete reduction bounds, the
serialized schedule, per-term substituted formulas, output casts and an
explicit completeness marker. A requested term window is diagnostic and
reports omitted terms; omitting the window expands the complete reduction.

The architectural review removed the unaccepted assignment-specific
`layer_1_o_proj` reduction binding left by interrupted loop 38. The generic
capture support for declared attention output projections remains available,
but artifact generation no longer promotes that single assignment without a
general source/operation contract.

## Real artifact evidence with source unavailable

The command below ran after renaming `gemma-4-E4B-dense` away and restored it
only through the shell exit trap:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-operations \
  --operation layer_0_q_proj --output-coordinate 0,0,0 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop39-complete-q-proj-navigation.json
```

Observed result:

- artifact bytes: `21,326,357,515`;
- embedded constants: `2,130`;
- ordered Gemma4Text operations: `1,229`, from `token_embedding` through
  `final_logit_softcap`;
- `sourceCheckpointAccessed: false`;
- complete reduction bounds: `[0, 2560)`;
- substituted learned scalars: `2,560`;
- `complete: true`, `omittedTerms: 0`;
- no term formula contained `weight[`;
- report bytes: `3,779,940`;
- report SHA-256:
  `0939ffedaaddc86997a1ddbf8147de845bc10c5233e0af2c179cea07d5f802d3`.

The first weight was decoded from BF16 bits `0xbbcb` to
`-0.006195068359375`; the last was decoded from `0xbd5f` to
`-0.054443359375`. Their formulas were respectively:

```text
product[0] = F32(layer_0_attn_norm[0,0,0] * -0.006195068359375)
product[2559] = F32(layer_0_attn_norm[0,0,2559] * -0.054443359375)
```

The repository test suite additionally exercises complete and windowed linear
views, embedding token substitution, RMSNorm weights, learned tensor scales,
proportional RoPE, compact attention coordinates and GELU formulas after
source tensors are cleared.

## Remaining checkpoint boundary

The scalar navigator currently covers the ordered Gemma4Text path; vision and
audio tower assignment types do not yet have the same indexed scalar view.
More importantly, navigation does not change native numeric fidelity. The
latest accepted authoritative evidence still reports approximate E4B
source-removed logits/KV replay, so `.agent-loop/checkpoints/gemma4-dense-lossless/`
must not be created.
