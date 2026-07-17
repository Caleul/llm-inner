# Gemma 4 E4B device-bound native capture — 2026-07-17

Gemma 4 native evidence now declares the exact execution device. This closes a
trace-contract gap: CPU and MPS can use different BF16 kernels and reduction
orders, so their outputs cannot be combined into one numerical-fidelity claim.
It does not establish an exact Gemma 4 replay or a checkpoint.

## Contract

All three Gemma 4 capture commands require `--device cpu` or `--device mps`.
The helper transfers both model and token/position tensors to that requested
device, reports the actual first-parameter device, and TypeScript rejects a
different report. The persisted trace has `reference.executionDevice`; the
linear-reduction probe rejects missing device metadata and rejects a campaign
whose source/runtime/IR/dtype/device contract differs between captures.

The earlier E4B traces were CPU traces: the previous helper left model and
inputs on PyTorch's default CPU device. They are therefore not MPS evidence.

## Live CPU revalidation

On this Mac, the pinned environment reported PyTorch `2.12.1`, Transformers
`5.5.0`, and successful BF16 allocation on both `cpu` and `mps:0`. Two fresh
CPU captures of the immutable public `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a` used the dense Safetensors SHA-256
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`:

```bash
npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop24-device-contract-cpu.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

npm run capture:gemma4-linear-reduction -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop24-device-contract-cpu-repeat.json \
  --input-tokens 2 --position-ids 0 --operation-id layer_0_up_proj \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu
```

Both traces declared `reference.executionDevice: "cpu"`, distinct capture IDs,
and the named boundary `layer_0_pre_ffn_norm -> layer_0_up_proj`.

The source directory was then renamed for the entire candidate command and
restored through a shell exit trap:

```bash
node dist/src/gemma4-linear-reduction-probe-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop24-device-contract-cpu.json \
  --trace /tmp/gemma4-loop24-device-contract-cpu-repeat.json \
  --operation-id layer_0_up_proj \
  --output /tmp/gemma4-loop24-device-contract-source-removed-probe.json \
  --max-read-mib 16 --lane-counts 32 --lane-reduction-orders balanced-pairwise
```

The source-removed report recorded `sourceCheckpointAccessed: false`,
`traceCount: 2`, and `reference.executionDevice: "cpu"`. No profile was exact:
the closest 32-lane balanced F32 and FMA candidates each differed in one BF16
coordinate with maximum absolute error `1.1920928955078125e-7`. Thus the new
contract is exercised against the real source-independent literal reader, but
the existing `layer_0_up_proj` approximate-fidelity boundary remains intact.
