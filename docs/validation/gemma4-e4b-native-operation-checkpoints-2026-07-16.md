# Gemma 4 E4B native operation checkpoints — 2026-07-16

This diagnostic evidence narrows the numerical-fidelity gap for the immutable
`google/gemma-4-E4B` dense BF16 package.  It is deliberately a **partial
native-module checkpoint trace**, not a Gemma 4 checkpoint claim and not a
complete operation-level differential trace.

## Evidence boundary

- Hub revision: `411aa17b749aa952df1359d2dcea73917a544d9a`
- Source tensor: `model.safetensors`, 15,992,595,884 bytes, SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- Native runtime: PyTorch 2.12.1 / Transformers 5.5.0,
  `Gemma4ForConditionalGeneration`, eager BF16
- Candidate: indexed `gemma4-e4b-dense.literal.json`, executed only through
  declared literal payload ranges with a 16 MiB read window
- Prompt: token ID `2`, absolute position `0`

The capture registers stable module outputs for embeddings, PLE projection and
norm, every text-layer projection/norm/complete attention/layer result, final
norm, tied LM head and final softcap.  Q/K/V norm values are transposed to the
same declared `BHSD` layout as the literal IR before comparison.  It captures
691 named boundaries; residual, activation, RoPE and attention-internal values
remain outside this diagnostic subset.

## Reproduction

Capture while the immutable source package is present:

```bash
npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-e4b-operation-checkpoints.json \
  --input-tokens 2 --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Then remove the complete source directory for the candidate process and run:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable
node dist/src/gemma4-paged-text-operation-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-operation-checkpoints.json \
  --report /tmp/gemma4-e4b-operation-checkpoints-report.json \
  --max-read-mib 16 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense
mv ./.gemma-4-E4B-dense-source-unavailable ./gemma-4-E4B-dense
```

`sourceCheckpointAccessed` is recorded as `false` in the report.  The capture
is integrity-bound to `config.json` and the contributing Safetensors file and
also rejects an IR fingerprint mismatch.

## Result

The previous earliest mismatch was `token_embedding`: the literal F32 path
used `sqrt(2560)` directly, while
`Gemma4TextScaledWordEmbedding` first casts that scale to BF16 and then casts
the multiplied embedding result to BF16.  The literal adapter now records the
widened BF16 scale (`50.5`) and its paged embedding kernel applies the declared
BF16 result cast.

The regenerated artifact SHA-256 is
`83d63c1be28a8cdb6f35a6b81a889552777da57d662b5367a88138f707a9c001`.
Its program records F32 reduction plus BF16 result casts for every Gemma4Text
assignment, and the paged executor applies those casts before the next named
assignment or cache transition.

The source-removed probe now passes 12 of 691 captured boundaries exactly:
both embeddings; layer 0 input norm, Q/K/V projections, Q/K/V norms,
pre-FFN norm, PLE projection, and post-PLE norm. The first divergent boundary
is still `ple_context_projection`, but its maximum absolute error is now one
BF16 ULP (`0.0000152587890625`) rather than `0.1101226806640625`. The remaining
679 boundaries diverge at zero tolerance. This establishes that the result
cast was a real missing contract; the unresolved native reduction tree and
uncaptured internal operations still prohibit a lossless-within-dtype or Gemma
4 checkpoint claim.
