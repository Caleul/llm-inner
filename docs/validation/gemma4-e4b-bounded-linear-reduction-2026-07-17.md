# Gemma 4 E4B bounded native linear-reduction evidence — 2026-07-17

This review independently reproduces the current dense E4B numerical boundary
without serializing the complete 1,229-assignment operation trace. It does not
claim an exact Gemma 4 replay or a checkpoint.

## Contract

`capture:gemma4-linear-reduction` accepts only an adapter-declared,
bias-free `layer_<n>_(gate|up|down)_proj` operation. It derives the matching
named producer from the Gemma4Text program, checks the source package is dense
BF16 Safetensors, hashes its complete contributing source set, and invokes the
version-pinned PyTorch 2.12.1 / Transformers 5.5.0 eager BF16 runtime.

The Python helper records exactly two values: the input passed to that native
`Linear` module and its returned output. Its pre/post hooks return no value;
the helper runs an unhooked forward first and rejects the capture unless the
hooked forward has bitwise-equal logits and every DynamicCache key/value. The
written execution bundle carries the full Gemma4Text IR fingerprint, source
checksums, declared tokens/positions, distinct capture UUID, and no external
tensor references.

The capture device is now an explicit part of that contract (`--device cpu` or
`--device mps`) and is written to `reference.executionDevice`; a reduction
campaign cannot combine the two. These historical captures ran on CPU: the
previous helper never transferred its model or input tokens to MPS.

## Reproduction

The two captures below were made from the immutable public package
`google/gemma-4-E4B`, revision `411aa17b749aa952df1359d2dcea73917a544d9a`;
the dense `model.safetensors` SHA-256 is
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-up-proj-bounded-a.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-e4b-up-proj-bounded-b.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu
```

Both completed and produced independent 227 KB bundles. The source-removed
candidate probe was then run with only the literal artifact plus those bundles:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable-loop17
npm run probe:gemma4-linear-reduction -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-up-proj-bounded-a.json \
  --trace /tmp/gemma4-e4b-up-proj-bounded-b.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-e4b-up-proj-bounded-source-removed-probe.json \
  --max-read-mib 16 --lane-counts 32 --lane-reduction-orders balanced-pairwise
mv ./.gemma-4-E4B-dense-source-unavailable-loop17 ./gemma-4-E4B-dense
```

## Result

The source-removed report states `sourceCheckpointAccessed: false`, binds the
two captures to `layer_0_pre_ffn_norm -> layer_0_up_proj`, and has no exact
profile. The measured candidates were:

| Profile | Mismatched BF16 coordinates | Max absolute error | First index |
| --- | ---: | ---: | ---: |
| ordered F32 scalar | 4 | 0.00390625 | 5883 |
| ordered F64 scalar | 4 | 0.000003814697265625 | 7720 |
| 32 interleaved F32 lanes, balanced fold | 1 | 0.00000011920928955078125 | 8354 |
| 32 interleaved FMA lanes, balanced fold | 1 | 0.00000011920928955078125 | 8354 |

This independently confirms the existing fail-closed boundary: a close
profile is not registered as native semantics. The real dense artifact remains
approximate from `layer_0_up_proj`; full multimodal paged execution and
source-removed authoritative generation comparison are still absent.
