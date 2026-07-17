# Gemma 4 E4B multi-input linear-reduction evidence — 2026-07-17

This campaign independently rechecked the preceding ARM/vector-reduction
candidate boundary, then tested the two closest remaining scalar schedules on
multiple declared Gemma 4 inputs. It is evidence about the pinned native
runtime, not permission to install a feature-specific or prompt-specific
reduction policy.

The model is public `google/gemma-4-E4B`, revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, using the dense BF16
`model.safetensors` whose SHA-256 is
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
Each capture used the same pinned PyTorch 2.12.1 / Transformers 5.5.0 eager
CPU environment and recorded the canonical BF16, contiguous row-major
`layer_0_pre_ffn_norm -> layer_0_up_proj` boundary. Two passive-hook captures
were required for every declared input before the probe accepted it:

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-loop30-token17.json \
  --input-tokens 17 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense --output /tmp/gemma4-loop30-tokens2-17.json \
  --input-tokens 2,17 --position-ids 0,1 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu
```

The single-token `2` pair from the preceding campaign, the repeated `17`
pair, and the repeated `2,17` pair were then replayed after temporarily
renaming the complete checkpoint directory. The probe read only the 21 GB
literal artifact and those six traces:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop29-arm-tree-a.json \
  --trace /tmp/gemma4-loop29-arm-tree-b.json \
  --trace /tmp/gemma4-loop30-token17.json \
  --trace /tmp/gemma4-loop30-token17-repeat.json \
  --trace /tmp/gemma4-loop30-tokens2-17.json \
  --trace /tmp/gemma4-loop30-tokens2-17-repeat.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-loop30-targeted-source-removed.json \
  --max-read-mib 16 --lane-counts 32 \
  --lane-reduction-orders balanced-pairwise --min-distinct-inputs 3 \
  --profile-id interleaved-f32-lanes-32-balanced-pairwise \
  --profile-id interleaved-fma-lanes-32-balanced-pairwise
```

`sourceCheckpointAccessed` is `false`, `traceCount` is 6, and the report has
no exact profiles. Both selected schedules have four mismatched BF16 output
coordinates across the campaign. For input token `2`, feature 8354 differs by
`1.1920928955078125e-7`; for token `17`, features 5614 and 5936 differ, with
maximum absolute error `0.000003814697265625`; the two-token capture again
misses feature 8354. The report's common output-feature coverage leaves
`[5614, 5936, 8354]` uncovered.

The original single-input near match therefore does not establish an output
tile schedule. Its discrepancy changes with declared activation values, while
the native eager-BF16 reduction semantics remain absent from the literal
program. The Gemma 4 dense-lossless checkpoint remains unclaimed.
