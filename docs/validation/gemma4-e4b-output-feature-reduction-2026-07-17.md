# Gemma 4 E4B cross-input output-feature reduction evidence — 2026-07-17

This campaign distinguishes a globally declared reduction schedule from a
profile that only happens to fit a limited output coordinate. It extends the
bounded `layer_0_up_proj` evidence with `outputFeatureCoverage`: run-length
encoded sets of candidate profiles that match every observed row of an output
feature for every distinct declared prompt. It is a probe report, not an
adapter policy or a Gemma 4 checkpoint claim.

The authoritative captures used the immutable dense public
`google/gemma-4-E4B` revision `411aa17b749aa952df1359d2dcea73917a544d9a`,
with BF16 `model.safetensors` SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
Each token below was captured twice with passive hooks; each repeat was
bitwise-equal for its producer and projection result.

```bash
for token in 3 4 5; do
  for repeat in a b; do
    npm run capture:gemma4-linear-reduction -- \
      --source ./gemma-4-E4B-dense \
      --output "/tmp/gemma4-e4b-loop21-up-proj-token${token}-${repeat}.json" \
      --input-tokens "$token" --position-ids 0 \
      --operation-id layer_0_up_proj --python ./venv/bin/python \
      --model google/gemma-4-E4B \
      --revision 411aa17b749aa952df1359d2dcea73917a544d9a
  done
done
```

The candidate process used two prior independent token-2 captures plus those
six new captures. The source directory was moved away for the whole candidate
process and restored with a shell exit trap:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop20-up-proj-a.json \
  --trace /tmp/gemma4-e4b-loop20-up-proj-b.json \
  --trace /tmp/gemma4-e4b-loop21-up-proj-token3-a.json \
  --trace /tmp/gemma4-e4b-loop21-up-proj-token3-b.json \
  --trace /tmp/gemma4-e4b-loop21-up-proj-token4-a.json \
  --trace /tmp/gemma4-e4b-loop21-up-proj-token4-b.json \
  --trace /tmp/gemma4-e4b-loop21-up-proj-token5-a.json \
  --trace /tmp/gemma4-e4b-loop21-up-proj-token5-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-loop21-four-input-output-feature-source-removed.json \
  --max-read-mib 16 --lane-counts 32 \
  --lane-reduction-orders ascending,descending,balanced-pairwise \
  --min-distinct-inputs 4
```

The report records `sourceCheckpointAccessed: false`, eight traces in four
distinct input groups, `outputFeatures: 10240`, no uncovered feature, and no
globally exact profile. The closest global profiles remain the 32-lane
balanced F32/FMA candidates, with three BF16 mismatches across the four
prompts; the largest observed absolute error is
`0.0001220703125`. The coverage has 41 profile-set spans. A static choice can
fit the observed features (for example, the balanced profile does not cover
features 1602, 8354, and 8777), but that is not an established semantic rule:
the native macOS ARM PyTorch build uses Accelerate BLAS and exposes no
authoritative per-output-feature accumulation contract in the model package.

Accordingly, no coordinate-specific schedule is written into the Gemma4Text
adapter or literal artifact. The current text path remains explicitly
approximate beginning at `layer_0_up_proj`; this report narrows the remaining
blocker without replacing it with a fitted lookup table.
