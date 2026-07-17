# Gemma 4 E4B tiled linear-reduction evidence — 2026-07-17

This experiment expands the bounded native `layer_0_up_proj` investigation
without treating a CPU/SIMD hypothesis as an artifact semantic.  It adds a
fully declared tiled schedule: each tile has `laneCount * termsPerLane` input
coordinates, and contiguous groups of `termsPerLane` products feed one F32
lane before an explicitly ordered F32 horizontal fold.  The schedule is
different from the existing `i mod laneCount` mapping and can describe BF16
dot-product instructions that reduce adjacent input terms into one lane.

Two independent hook-stable captures were produced from the immutable public
`google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, dense
`model.safetensors` SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`:

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-loop18-up-proj-a.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-loop18-up-proj-b.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

The candidate then ran with the checkpoint directory unavailable; it read only
the self-contained literal artifact and the two evidence bundles:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable-loop18-wide
npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop18-up-proj-a.json \
  --trace /tmp/gemma4-e4b-loop18-up-proj-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-loop18-wide-tiled-reduction-source-removed.json \
  --max-read-mib 16 --lane-counts 4,8,16,32 \
  --tiled-lane-counts 4,8,16,32 --tiled-terms-per-lane 2,4,8,16 \
  --lane-reduction-orders ascending,descending,balanced-pairwise
mv ./.gemma-4-E4B-dense-source-unavailable-loop18-wide ./gemma-4-E4B-dense
```

The report has `sourceCheckpointAccessed: false`, two bitwise-equal native
captures, 122 profiles, and an empty `exactProfileIds`.  The closest overall
result remains the existing 32-lane modulo mapping with a balanced fold:
one mismatched BF16 coordinate at index 8354 and absolute error
`1.1920928955078125e-7`.  The best tiled candidates also miss one coordinate:
8 lanes, 2 contiguous terms per lane, any listed horizontal fold, first differ
at index 9997 with absolute error `2.384185791015625e-7`.

Therefore no tiled profile is installed in the Gemma4Text adapter.  The
artifact stays explicitly approximate from `layer_0_up_proj`; this is search
evidence, not an exact runtime-kernel contract or a Gemma 4 checkpoint claim.
