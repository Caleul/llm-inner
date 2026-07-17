# Gemma 4 E4B blocked tiled BF16 linear-reduction evidence — 2026-07-17

This experiment separates two calculation contracts that a vague “SIMD
reduction” would otherwise conflate. `tiled-f32-lanes` retains its F32 lanes
over the full feature axis. `blocked-tiled-f32-lanes` instead initializes new
lanes for every finite `laneCount * termsPerLane` contiguous tile, folds that
tile in a declared order, and adds its rounded result to an ordered F32 block
accumulator. Product rounding versus FMA is declared per tile.

Two fresh passive-hook captures used the immutable dense public
`google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, with
`model.safetensors` SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`:

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-loop20-up-proj-a.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-loop20-up-proj-b.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

The candidate campaign ran after moving the source directory away. Its reader
opened only `artifacts/gemma4-e4b-dense.literal.json` and the two captures;
the move was protected by a shell exit trap and the source was restored:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop20-up-proj-a.json \
  --trace /tmp/gemma4-e4b-loop20-up-proj-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-loop20-blocked-tiled-source-removed.json \
  --max-read-mib 16 --lane-counts 32 \
  --blocked-tiled-lane-counts 4,8,16,32 \
  --blocked-tiled-terms-per-lane 2,4,8 \
  --lane-reduction-orders ascending,descending,balanced-pairwise
```

The resulting report has `sourceCheckpointAccessed: false`, two bitwise-equal
captures, 81 profiles, and `exactProfileIds: []`. The best newly added
blocked-tiled result is 32 lanes × 4 terms, ascending fold, with either product
boundary: one BF16 mismatch at coordinate 9997 and maximum absolute error
`2.384185791015625e-7`. The prior 32-lane modulo schedule remains closest:
one mismatch at coordinate 8354 and maximum absolute error
`1.1920928955078125e-7`.

No candidate was installed as Gemma 4 semantics. The literal reader also now
accepts the older artifact header only if the embedded graph uses exclusively
the reduction grammar that header named; newer blocked/tiled schedules require
the newer, more specific header.
