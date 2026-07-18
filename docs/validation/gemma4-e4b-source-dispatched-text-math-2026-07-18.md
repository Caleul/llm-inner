# Gemma 4 E4B source-dispatched text math — 2026-07-18

This is candidate evidence produced by the implementation loop. It does not
accept the dense-lossless checkpoint and does not create
`.agent-loop/checkpoints/gemma4-dense-lossless/`. The real multimodal replay
and the nontrivial cached-attention boundary remain unresolved and require an
independent review.

## Immutable source and artifact

- Model: `google/gemma-4-E4B`.
- Hub revision: `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Container: dense, unquantized BF16 Safetensors.
- Reference runtime: Transformers 5.5.0 and PyTorch 2.12.1 CPU eager on
  Darwin arm64.
- PyTorch source commit: `7269437d655783a26cba32aa88195b741ff496aa`.
- Bundled SLEEF commit: `5a1d179df9cf652951b59010a2d2075372d67f68`.
- Artifact: `artifacts/gemma4-e4b-dense.literal.json`.
- Artifact bytes: `21,327,229,087`.
- Artifact SHA-256:
  `fa57810c28575a95797c87f72ca09dbabd3a3fb722dd13c675b4505762cf6e1a`.
- Embedded constants/storage decoders: `2,130` / `2,130`.

The generated text program contains 1,229 ordered assignments. Every one of
its 344 bias-free transposed BF16 linears carries the same source-, dtype- and
layout-dispatched ARM 32-lane register tree; no assignment ID or layer shape
selects it. All 302 RMS reductions carry PyTorch's contiguous F32
`cascade_sum` schedule. All 84 GELU assignments and the final softcap pin the
installed SLEEF tanh implementation. All 66 RoPE assignments pin the SLEEF
sin/cos implementation and declare BF16 casts for sin/cos, both products and
the final sum.

The dispatch evidence is the pinned source path:

- `F.linear -> matmul -> mm_out_cpu -> addmm_impl_cpu_ -> cpublas::gemm ->
  gemm_transa_ -> bf16_dot_with_fp32_arith` for compatible linears;
- `Gemma4RMSNorm hidden_states.float().pow(2).mean(-1) -> mean_out CPU ->
  sum_out F32 -> cascade_sum -> vectorized_inner_sum` for compatible RMS
  reductions;
- `Sleef_tanhf4_u10advsimd`, `Sleef_sinf4_u10advsimd` and
  `Sleef_cosf4_u10advsimd` from the installed `libtorch_cpu.dylib` for the
  captured transcendental paths.

## Source-removed exact assignment replay

The final authoritative operation capture used token `184`, position `1`, so
RoPE and every downstream operation were nontrivial. The candidate command ran
while `gemma-4-E4B-dense` was renamed and required zero absolute and relative
tolerance:

```bash
npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop44-cascade-operation-checkpoints.json \
  --input-tokens 184 --position-ids 1 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

mv ./gemma-4-E4B-dense ./gemma-4-E4B-dense.loop44-source-hidden
node dist/src/gemma4-paged-text-operation-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop44-cascade-operation-checkpoints.json \
  --report /tmp/gemma4-loop44-cascade-source-removed-operation-report.json \
  --max-read-mib 16 --assert-source-unavailable ./gemma-4-E4B-dense \
  --absolute-tolerance 0 --relative-tolerance 0 \
  --allow-unverified-fidelity
mv ./gemma-4-E4B-dense.loop44-source-hidden ./gemma-4-E4B-dense
```

Observed evidence:

- `sourceCheckpointAccessed: false`;
- 1,229/1,229 assignments passed;
- terminal logits shape `[1,1,262144]`, max absolute/relative error `0`;
- cosine similarity and top-k overlap `1`;
- argmax agreement `true`;
- all 24 producer-owned KV key/value pairs passed with zero error;
- `firstDivergentOperation: null`;
- fidelity class `lossless-within-dtype`.

The candidate acknowledgment flag is still required because this implementing
loop does not promote the artifact's fidelity declaration or accept its own
checkpoint claim.

## Source-removed scalar audit views

The indexed reader rendered these formulas while the checkpoint directory was
absent:

```text
layer_0_q_rot[0,0,0,0] =
  BF16(F32(BF16(F32(layer_0_q_normalized[0,0,0,0] * cosine))
  - BF16(F32(layer_0_q_normalized[0,0,0,128] * sine))))
cosine = BF16(SLEEF_COS_F32(angle))
sine = BF16(SLEEF_SIN_F32(angle))

softcapped_logits[0,0,184] =
  BF16(F32(30 * BF16(SLEEF_TANH_F32(BF16(F32(logits[0,0,184] / 30))))))
```

The large learned tensors remain embedded losslessly, with their decoder,
layout, dtype bits and scalar-substitution path available from the same
artifact; neither view opened the original Safetensors package.

## Cached-generation boundary

A fresh one-step native trace for prompt token `2` was compared with the same
artifact while the checkpoint directory was absent and with zero tolerances.
The prompt matched, selection logits were exact, and both runtimes selected
token `184` at position `1`. After the RoPE and RMS changes, the incremental
layer-0 key and value caches also matched exactly.

The first remaining divergence is
`generation:step-0-kv-cache-layer-1`. Layers 1–23 then diverge, and terminal
logits have maximum absolute error `0.23828125`, top-k overlap `0.9`, cosine
similarity `0.9998959201289614`, and argmax agreement `true`. This isolates the
unresolved calculation class to nontrivial two-key eager attention/softmax and
its BF16 batched-matmul boundaries. It is not evidence of exact generation.

The scalar SLEEF sin/cos replay currently implements the installed fast-range
argument reducer and fails closed for `abs(angle) >= 125`; the full SLEEF
`rempif` path is not yet transcribed. Multimodal source-removed differential
execution is also not established. These limits, plus the cached-attention
divergence, prohibit the Gemma checkpoint marker.
