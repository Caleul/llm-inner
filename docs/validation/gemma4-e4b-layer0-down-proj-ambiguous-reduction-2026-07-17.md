# Gemma 4 E4B layer-0 MLP down projection: ambiguous reduction evidence — 2026-07-17

This source-removed campaign tested the remaining layer-0 MLP projection under
the same immutable dense E4B source and eager CPU BF16 runtime as the bounded
gate/up candidates. It deliberately does **not** install a reduction policy:
the observed output is compatible with multiple complete schedules.

## Bound inputs and source removal

- Source: `google/gemma-4-E4B`, revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Dense BF16 `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- Literal artifact: `artifacts/gemma4-e4b-dense.literal.json`, SHA-256
  `7a778d31269187f71a85458cf97b6b91f8c28eacd3fca78e43463bf833db7f0c`.
- Authoritative capture: PyTorch `2.12.1`, Transformers `5.5.0`, eager CPU
  BF16, passive native `down_proj` hooks which reject any logits or KV-cache
  change.

Two independent captures were made for each declared input `[2]` at `[0]`,
`[17]` at `[0]`, and `[2,17]` at `[0,1]`. An initial source-removed sweep of
49 declared profiles left six exact. The checkpoint directory was renamed
again for the following source-removed verdict, which replays precisely those
six complete candidates from the literal artifact:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop33-down-token2-a.json \
  --trace /tmp/gemma4-loop33-down-token2-b.json \
  --trace /tmp/gemma4-loop33-down-token17-a.json \
  --trace /tmp/gemma4-loop33-down-token17-b.json \
  --trace /tmp/gemma4-loop33-down-tokens2-17-a.json \
  --trace /tmp/gemma4-loop33-down-tokens2-17-b.json \
  --operation-id layer_0_down_proj \
  --output /tmp/gemma4-loop33-down-source-removed-selection.json \
  --max-read-mib 16 --lane-counts 32 --min-distinct-inputs 3 \
  --profile-id arm-neon-bf16-dot-fma-32-ascending \
  --profile-id arm-neon-bf16-dot-fma-32-pairwise \
  --profile-id interleaved-f32-lanes-32-ascending \
  --profile-id interleaved-fma-lanes-32-ascending \
  --profile-id interleaved-f32-lanes-32-balanced-pairwise \
  --profile-id interleaved-fma-lanes-32-balanced-pairwise
```

The report records `sourceCheckpointAccessed: false`, `traceCount: 6`, and
three stable input groups. Its `candidateSelection.status` is `ambiguous`;
there is no `profileId` to bind into `Gemma4Text`.

## Exact but non-identifying profiles

All six profiles had zero mismatched BF16 elements and zero maximum absolute
error across every trace:

- `arm-neon-bf16-dot-fma-32-ascending`
- `arm-neon-bf16-dot-fma-32-pairwise`
- `interleaved-f32-lanes-32-ascending`
- `interleaved-fma-lanes-32-ascending`
- `interleaved-f32-lanes-32-balanced-pairwise`
- `interleaved-fma-lanes-32-balanced-pairwise`

Because these schedules differ in declared register tree, product boundary, or
horizontal fold, their common BF16 observations do not establish which native
calculation occurred. `layer_0_down_proj` therefore retains the existing
ordered-scalar F64 literal policy. This is evidence about one pinned CPU
operation only; it does not establish text replay, generation, multimodal
execution, or the Gemma 4 dense-lossless checkpoint.
