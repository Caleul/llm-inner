# Gemma 4 E4B 32-lane ARM reduction candidate — 2026-07-17

The preceding ARM schedule implementation was incomplete: it performed the
first two `VectorizedN<float, 8>` register-tree stages but omitted the final
merge of the two remaining four-lane vectors. That loses half of the active
register state. The literal executor now performs all three stages before the
declared horizontal fold, and a non-degenerate unit test exercises a value in
the formerly dropped vector.

Fresh passive-hook captures used public `google/gemma-4-E4B`, revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, dense BF16
`model.safetensors` SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`, and
the pinned PyTorch 2.12.1 / Transformers 5.5.0 eager CPU runtime. Every
capture bound the row-major BF16
`layer_0_pre_ffn_norm -> layer_0_up_proj` assignment and was repeated before
the source-removed comparison accepted it.

The candidate comparison used only the literal artifact and six fresh traces:
two each for declared inputs `[2]`, `[17]`, and `[2,17]` at positions `[0]`,
`[0]`, and `[0,1]`. The checkpoint directory was renamed for the complete
probe and restored by an exit trap:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop31-emitted-token2-a.json \
  --trace /tmp/gemma4-loop31-emitted-token2-b.json \
  --trace /tmp/gemma4-loop31-emitted-token17-a.json \
  --trace /tmp/gemma4-loop31-emitted-token17-b.json \
  --trace /tmp/gemma4-loop31-emitted-tokens2-17-a.json \
  --trace /tmp/gemma4-loop31-emitted-tokens2-17-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-loop31-emitted-arm-32-source-removed.json \
  --max-read-mib 16 --lane-counts 32 --min-distinct-inputs 3 \
  --profile-id arm-neon-bf16-dot-fma-32-ascending \
  --profile-id arm-neon-bf16-dot-fma-32-pairwise
```

The report records `sourceCheckpointAccessed: false`, `traceCount: 6`, and
three distinct declared input groups. The complete 32-lane pairwise ARM tree
is exact for every observed BF16 coordinate: zero mismatches, zero maximum
absolute error, and no uncovered output feature. The ascending fold has one
mismatch at feature `1786` with absolute error `0.00006103515625`, so the
fold remains semantically material and only the pairwise schedule is emitted
for this trace-bound E4B `layer_0_up_proj` assignment.

This is a candidate operation-level acceptance claim, not a Gemma 4
checkpoint claim. Other E4B operations, end-to-end text generation, and
multimodal execution still require independent differential evidence.
