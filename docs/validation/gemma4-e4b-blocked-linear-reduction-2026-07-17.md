# Gemma 4 E4B blocked BF16 linear-reduction evidence — 2026-07-17

This experiment tests an explicit adjacent-term partial-reduction family
against the pinned eager-BF16 `layer_0_up_proj` boundary.  It does not infer a
kernel contract from the tensor shape.  The literal schedule declares the
number of contiguous terms in each partial, the term order, whether the
product is rounded before the partial F32 addition, and the subsequent ordered
F32 accumulator boundary.

Two independent passive-hook captures were freshly produced from the immutable
public `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, dense
`model.safetensors` SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`:

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-loop19-up-proj-a.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-loop19-up-proj-b.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

The candidate process then ran directly from the literal artifact while the
entire checkpoint directory was temporarily unavailable:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable-loop19-blocked
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-loop19-up-proj-a.json \
  --trace /tmp/gemma4-e4b-loop19-up-proj-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-loop19-blocked-source-removed.json \
  --max-read-mib 16 --blocked-terms-per-block 2,4,8,16
mv ./.gemma-4-E4B-dense-source-unavailable-loop19-blocked ./gemma-4-E4B-dense
```

The source-removed report records `sourceCheckpointAccessed: false`, two
bitwise-equal captures, 61 profiles, and an empty `exactProfileIds`.  The 16
new blocked candidates cover contiguous blocks of 2, 4, 8 and 16 terms, both
ascending and descending term order, and separately-rounded versus FMA product
boundaries.  Their best result is the 16-term ascending family with four
mismatched BF16 coordinates and maximum absolute error
`3.814697265625e-6`; the 2-term pair candidates miss five coordinates with
maximum absolute error `7.62939453125e-6`.

The earlier 32-lane interleaved F32 balanced fold therefore remains the closest
tested candidate: one mismatch at coordinate 8354 and absolute error
`1.1920928955078125e-7`.  This experiment rules out the declared blocked
families for this capture; it does not promote any approximate profile to a
Gemma 4 runtime contract or checkpoint claim.
