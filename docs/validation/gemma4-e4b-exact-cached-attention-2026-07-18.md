# Gemma 4 E4B exact cached-attention candidate

## Scope

This loop closed the first nontrivial cached-generation divergence in the real
dense Gemma 4 E4B text program. The artifact now declares and executes the
PyTorch eager CPU BF16 attention implementation selected for every compatible
Gemma4Text attention assignment. This is one operation-, dtype-, topology- and
source-dispatched contract, not an assignment allowlist.

The source identity is immutable:

- model: `google/gemma-4-E4B`;
- revision: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`.

The regenerated literal artifact is
`artifacts/gemma4-e4b-dense.literal.json`, 21,327,343,663 bytes, SHA-256
`32f7a52b0200cb4988edb8f05146af8d9f0af0450f10dd4ded1130ec8bd23868`.
It contains 2,130 embedded constants. Source-removed payload verification read
and decoded 15,992,314,836 embedded bytes and produced literal-storage SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

## Declared attention calculation

All 42 compatible text attention assignments carry the same pinned contract:

1. query-key scores use PyTorch's ARM non-BFDOT
   `bf16_dot_with_fp32_arith` register tree and cast each score to BF16;
2. topology and additive masking occur at the declared BF16 boundary;
3. softmax promotes scores to F32, uses the four-lane PyTorch max/sum fold and
   `Sleef_expf4_u10advsimd`, then casts each probability to BF16;
4. probability-value products use the same ARM BF16/F32 register tree and
   cast the context to BF16.

The evidence is bound to PyTorch source commit
`7269437d655783a26cba32aa88195b741ff496aa` and SLEEF source commit
`5a1d179df9cf652951b59010a2d2075372d67f68`. The dispatch predicate requires
the immutable E4B topology, BF16 runtime, head dimension 256 or 512, scale 1,
F32 softmax and absence of an attention softcap. A mismatch remains
fail-closed instead of receiving these semantics by tensor name.

The source-removed scalar view for `layer_0_attention_context[0,0,0]` exposes
the same calculation as indexed assignments:

```text
dot[k] = BF16_RNE(ARM_NEON_BF16_DOT_F32_pairwise(q[feature], k[k,feature], feature=0..255))
effective_mask[k] = declared_mask[k] when it defines topology; otherwise F32(topology_mask[k] + declared_mask[k])
score[k] = BF16_RNE(F32(BF16_RNE(F32(dot[k] * 1)) + effective_mask[k]))
maximum = PYTORCH_F32_VECTOR_MAX_PAIRWISE(score[k], k=0..key_length-1, lanes=4)
exp_score[k] = SLEEF_EXP_F32(F32(score[k] - maximum))
total = PYTORCH_F32_VECTOR_SUM_PAIRWISE(exp_score[k], k=0..key_length-1, lanes=4)
probability[k] = BF16_RNE(F32(exp_score[k] * F32(1 / total)))
context[0] = BF16_RNE(ARM_NEON_BF16_DOT_F32_pairwise(probability[k], value[k,0], k=0..key_length-1))
```

## Reproducible evidence

The artifact was regenerated with:

```bash
npm run build
node dist/src/cli.js --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal
```

A fresh operation trace at nonzero token and position was compared with zero
tolerance while the source directory was renamed away:

```bash
npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop45-attention-operation-checkpoints.json \
  --input-tokens 184 --position-ids 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

mv ./gemma-4-E4B-dense ./gemma-4-E4B-dense.loop45-source-hidden
node dist/src/gemma4-paged-text-operation-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop45-attention-operation-checkpoints.json \
  --report /tmp/gemma4-loop45-attention-operation-final-report.json \
  --max-read-mib 16 --assert-source-unavailable ./gemma-4-E4B-dense \
  --absolute-tolerance 0 --relative-tolerance 0 \
  --allow-unverified-fidelity
mv ./gemma-4-E4B-dense.loop45-source-hidden ./gemma-4-E4B-dense
```

Observed result: 1,229/1,229 ordered assignments passed; logits and all 24
producer-owned KV key/value pairs had maximum absolute and relative error 0;
cosine similarity and top-k overlap were 1; argmax agreed;
`firstDivergentOperation` was null; `sourceCheckpointAccessed` was false; and
the candidate fidelity class was `lossless-within-dtype`.

The decisive cached-generation evidence used a fresh four-step authoritative
trace:

```bash
npm run capture:gemma4-text-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop45-attention-generation-4steps.json \
  --input-tokens 2 --position-ids 0 --max-new-tokens 4 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

mv ./gemma-4-E4B-dense ./gemma-4-E4B-dense.loop45-source-hidden
npm run compare:gemma4-paged-text-trace -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop45-attention-generation-4steps.json \
  --report /tmp/gemma4-loop45-attention-generation-4steps-final-report.json \
  --max-read-mib 16 --top-k 10 \
  --absolute-tolerance 0 --relative-tolerance 0 \
  --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
mv ./gemma-4-E4B-dense.loop45-source-hidden ./gemma-4-E4B-dense
```

Observed result: greedy tokens `184`, `3910`, `531`, `974` at positions 1..4
all passed. Every selection-logit tensor, all 24 producer caches at each step
(96 total), terminal logits and all 24 terminal caches had maximum absolute and
relative error 0. Cosine similarity and top-k overlap were 1, argmax agreed,
`firstDivergence` was null, `sourceCheckpointAccessed` was false, and the
candidate fidelity class was `lossless-within-dtype`.

## Acceptance boundary

This is candidate evidence produced by the implementing loop. It does not
independently accept the product checkpoint, so
`.agent-loop/checkpoints/gemma4-dense-lossless/` was not created.

The dense text forward and cached greedy-generation path is now exact under
the pinned CPU BF16 runtime evidence. The remaining product limits are real
Gemma 4 image/video/audio source-removed differential replay and the complete
SLEEF large-angle `rempif` range reducer; unsupported large-angle trigonometry
continues to fail closed.
