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
Each bounded-linear capture also declares `reference.operationDtypes` for the
native module input, output, and weight before the tensors are widened to the
trace's F32 payload. The probe requires that record and rejects a trace whose
native dtype boundary differs from the literal assignment; serializing a value
as F32 is not treated as proof that the module executed in F32.

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

## Loop 26 native-dtype revalidation

The former CPU traces predated native operation-dtype evidence, so they cannot
be used by the strengthened probe. Loop 26 captured two fresh `cpu` traces and
two fresh `mps` traces for the same token/position and immutable source.
Every capture reported the actual module boundary below:

```json
{
  "operationId": "layer_0_up_proj",
  "inputDtype": "bfloat16",
  "outputDtype": "bfloat16",
  "parameterDtype": "bfloat16"
}
```

The source directory was unavailable for both candidate probe processes. The
CPU report has `sourceCheckpointAccessed: false`, `traceCount: 2`, and no
exact profile: its closest 32-lane balanced F32/FMA candidates each miss one
BF16 coordinate by `1.1920928955078125e-7`. The MPS report independently has
`sourceCheckpointAccessed: false`, `traceCount: 2`, and no exact profile; its
closest ordered-F64 candidate misses two BF16 coordinates by
`1.1920928955078125e-7`. Thus the remaining reduction blocker is not caused
by a hidden CPU/MPS dtype promotion, and neither backend establishes the dense
lossless checkpoint.

## Loop 27 native-kernel environment boundary

Module dtypes and device family still do not identify an eager BF16 reduction
kernel. The bounded capture now persists `reference.nativeKernelEnvironment`,
and the source-removed probe rejects a missing or different environment before
it compares one literal byte range. The record includes the SHA-256 of
`torch.__config__.show()`, intra-op/inter-op worker counts, deterministic mode,
and MKLDNN availability/enabled state. It is evidence about the authoritative
runtime process, not a hidden execution dependency of the literal artifact.

Two fresh CPU captures for token `2`, position `0`, of the same immutable E4B
revision both reported:

```json
{
  "torchBuildConfigSha256": "606e3853213dea3faabc6d58b66ed7e419ee4452a6d53c2b27495a2ecc4e07a7",
  "intraopThreads": 10,
  "interopThreads": 14,
  "deterministicAlgorithms": false,
  "mkldnnAvailable": false,
  "mkldnnEnabled": true
}
```

`mkldnnEnabled: true` with `mkldnnAvailable: false` is retained as the actual
Apple PyTorch state; it is not rejected or normalized. With the source
directory unavailable, the expanded 2/4/8/16/32/64/128-lane campaign still
found no exact profile. Its best result remains 32 interleaved F32 lanes with
a balanced fold: one BF16 coordinate (index 8354) differs by
`1.1920928955078125e-7`. Thus the added environment evidence rules out an
unrecorded build/thread/backend difference between the two captures, while the
unresolved literal reduction schedule remains the checkpoint blocker.

## Loop 28 native-layout boundary

The BF16 dtype and kernel-environment records still left one semantic question
open: `tensor_payload` converts native tensors to contiguous F32 before
serializing them. A trace could therefore look compatible with the literal
row-major Safetensors decoder even if native `Linear` had received an offset,
strided, or transposed view. The bounded capture now records the original
`shape`, `strides`, `storageOffset`, and contiguity for the native linear input,
output, and parameter. The TypeScript boundary validates the record and the
source-removed probe requires the exact canonical row-major layout declared by
the literal formula; its report retains that accepted record.

Two fresh CPU captures of the same pinned E4B `layer_0_up_proj` reported the
same compatible layouts:

```json
{
  "input": { "shape": [1, 1, 2560], "strides": [2560, 2560, 1], "storageOffset": 0, "isContiguous": true },
  "output": { "shape": [1, 1, 10240], "strides": [10240, 10240, 1], "storageOffset": 0, "isContiguous": true },
  "parameter": { "shape": [10240, 2560], "strides": [2560, 1], "storageOffset": 0, "isContiguous": true }
}
```

With `./gemma-4-E4B-dense` renamed for the candidate process, the expanded
2/4/8/16/32/64/128-lane source-removed campaign again had
`sourceCheckpointAccessed: false`, no exact profile, and the same closest
32-lane balanced F32 result: one BF16 coordinate at output feature `8354`
differs by `1.1920928955078125e-7`. Thus this change proves that the known
mismatch is not explained by an erased native tensor-view layout, but it does
not invent the still-unknown native accumulation schedule.
