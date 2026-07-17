# Gemma 4 E4B layer-0 MLP ARM reduction candidates — 2026-07-17

This campaign revalidated the predecessor's `layer_0_up_proj` claim against
the regenerated literal artifact and established the same explicit schedule
for the adjacent `layer_0_gate_proj` assignment. It is operation-level
candidate evidence only; it does not establish text generation, multimodal
execution, or the Gemma 4 dense-lossless checkpoint.

## Immutable source and runtime

- Source: `google/gemma-4-E4B` revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Dense BF16 `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- Candidate artifact: `artifacts/gemma4-e4b-dense.literal.json`, SHA-256
  `7a778d31269187f71a85458cf97b6b91f8c28eacd3fca78e43463bf833db7f0c`.
- Authoritative runtime: PyTorch `2.12.1`, Transformers `5.5.0`, eager CPU
  BF16; traces bind the native kernel environment, row-major layouts and all
  source-file checksums.

The regenerated artifact declares `arm-neon-bf16-dot-fma` for exactly
`layer_0_gate_proj` and `layer_0_up_proj` on the complete registered E4B
topology: 32 lanes, eight four-lane registers, `i mod 32` placement, the three
register-tree merges, and a pairwise four-lane horizontal fold. No other
projection receives this schedule from topology or matrix shape.

## Reproducible candidate comparison

Six passive-hook traces were captured for each assignment: independent repeats
of token `[2]` at position `[0]`, token `[17]` at `[0]`, and `[2,17]` at
`[0,1]`. The checkpoint directory was renamed for both probes and restored by
an exit trap:

```bash
for target in gate up; do
  node dist/src/gemma4-linear-reduction-probe-cli.js \
    --artifact ./artifacts/gemma4-e4b-dense.literal.json \
    --trace /tmp/gemma4-loop32-current-${target}-token2-a.json \
    --trace /tmp/gemma4-loop32-current-${target}-token2-b.json \
    --trace /tmp/gemma4-loop32-current-${target}-token17-a.json \
    --trace /tmp/gemma4-loop32-current-${target}-token17-b.json \
    --trace /tmp/gemma4-loop32-current-${target}-tokens2-17-a.json \
    --trace /tmp/gemma4-loop32-current-${target}-tokens2-17-b.json \
    --operation-id layer_0_${target}_proj \
    --output /tmp/gemma4-loop32-current-${target}-source-removed.json \
    --max-read-mib 16 --lane-counts 32 --min-distinct-inputs 3 \
    --profile-id arm-neon-bf16-dot-fma-32-pairwise
done
```

Both reports state `sourceCheckpointAccessed: false`, `traceCount: 6`, and
select only `arm-neon-bf16-dot-fma-32-pairwise`. Each has zero mismatched BF16
coordinates and zero maximum absolute error. The `up` report is an independent
review of the prior candidate because its six traces carry the regenerated IR
fingerprint and its comparison consumes the new artifact.

The evidence is intentionally bounded to these two named E4B CPU assignments.
The remaining linear and normalization schedules, text end-to-end comparison,
KV cache/generation comparison, vision/audio paged replay, and the mandatory
checkpoint remain unproven.
