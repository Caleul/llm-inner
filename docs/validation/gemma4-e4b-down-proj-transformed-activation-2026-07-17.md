# Gemma 4 E4B down-projection transformed-activation boundary — 2026-07-17

This campaign extended the ambiguous `layer_0_down_proj` CPU-BF16 reduction
evidence without allowing a synthetic activation to masquerade as a model
forward result. It adds an integrity-bound diagnostic capture contract and
shows that the currently plausible six schedules remain non-identifying.

## Capture contract

`capture:gemma4-linear-reduction` normally records
`reductionProbeInput: { "kind": "model-forward" }`. Its helper first runs an
unmodified reference forward, then repeats it with passive pre/post hooks and
requires bitwise-identical logits and every KV cache tensor.

For a kernel diagnostic only, the helper may then invoke the same registered
BF16 `Linear` on the captured activation after either of these explicit input
transforms:

- `{ "kind": "bf16-power-of-two-scale", "factor": 16 }`;
- `{ "kind": "bf16-scalar-scale", "factorBf16Bits": 16320 }`, where
  `16320` is the exact BF16 bit pattern `0x3fc0` (`1.5`).

The scalar is reconstructed in PyTorch from its uint16 BF16 payload rather
than from a JSON decimal. The trace parser accepts only finite nonzero BF16
payloads. The reduction probe includes this declaration in the trace identity,
input-group key, and runtime contract; model-forward and transformed traces
cannot be combined. Legacy traces normalize to `model-forward` only.

The diagnostic result is kernel evidence, not a source-removed model replay,
operation-differential result, or generation claim.

## Immutable source and environment

- Source: `google/gemma-4-E4B`, revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Dense BF16 `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- Literal artifact: `artifacts/gemma4-e4b-dense.literal.json`, SHA-256
  `7a778d31269187f71a85458cf97b6b91f8c28eacd3fca78e43463bf833db7f0c`.
- Native runtime: PyTorch `2.12.1`, Transformers `5.5.0`, eager CPU BF16 on
  macOS ARM64. The traces bind the observed Torch build hash, 10 intra-op
  threads, 14 inter-op threads, disabled deterministic algorithms, and the
  no-MKLDNN build.

## Source-removed evidence

Two independent captures were made for the declared prompt
`[1,17,42,101,512,2048,8191,65535]` at positions `[0,1,2,3,4,5,6,7]`.
The eight-row untransformed extension was also paired with the prior six
captures, producing four independent model-forward groups and 12 rows total.

The transformed capture command was:

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop35-down-bf16-1p5-a.json \
  --input-tokens 1,17,42,101,512,2048,8191,65535 \
  --position-ids 0,1,2,3,4,5,6,7 \
  --operation-id layer_0_down_proj \
  --activation-bf16-scale-bits 0x3fc0 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu
```

It was repeated independently as `...-b.json`. During each candidate replay,
the complete `gemma-4-E4B-dense` directory was renamed, restored only by an
exit trap, and asserted unavailable:

```bash
npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop35-down-bf16-1p5-a.json \
  --trace /tmp/gemma4-loop35-down-bf16-1p5-b.json \
  --operation-id layer_0_down_proj \
  --output /tmp/gemma4-loop35-down-bf16-1p5-source-removed-selection.json \
  --max-read-mib 16 --lane-counts 32 --min-distinct-inputs 1 \
  --profile-id arm-neon-bf16-dot-fma-32-ascending \
  --profile-id arm-neon-bf16-dot-fma-32-pairwise \
  --profile-id interleaved-f32-lanes-32-ascending \
  --profile-id interleaved-fma-lanes-32-ascending \
  --profile-id interleaved-f32-lanes-32-balanced-pairwise \
  --profile-id interleaved-fma-lanes-32-balanced-pairwise
```

All three reports declare `sourceCheckpointAccessed: false` and
`candidateSelection: { "status": "ambiguous" }`:

| Input contract | Traces | Exact profiles | Mismatched BF16 elements |
| --- | ---: | ---: | ---: |
| Four model-forward groups, including the new 8-row prompt | 8 | 6 | 0 for every profile |
| BF16 power-of-two factor 16 | 2 | 6 | 0 for every profile |
| BF16 scalar `0x3fc0` / 1.5 | 2 | 6 | 0 for every profile |

The exact profiles are unchanged: the two ARM register-fold variants and four
interleaved 32-lane F32/FMA variants. Since they make materially different
declared accumulation choices, the evidence does not authorize selecting one.
`layer_0_down_proj` retains its ordered-scalar F64 literal declaration, and
the current source-removed text replay remains approximate from that first
divergence. The Gemma 4 dense-lossless checkpoint remains absent.
