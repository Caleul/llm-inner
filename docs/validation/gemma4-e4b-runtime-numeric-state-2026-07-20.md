# Gemma 4 E4B: pinned runtime numeric state — 2026-07-20

## Completed boundary

Schema 53 closes numeric process state omitted by schema 52 around the 100
Gemma 4 BMM operations whose Apple Accelerate scalar schedule remains
unpublished. Live probing established a concrete drift path:
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` changes
`torch.get_float32_matmul_precision()` from `highest` to `high` without
changing the pinned host, binaries, dependency trees or the five schema-52
execution-state fields.

`runtimeExecutionState` schema 2 now declares and the embedded adapter
configures before reading artifact operands:

```json
{
  "schemaVersion": 2,
  "intraopThreads": 10,
  "interopThreads": 14,
  "deterministicAlgorithms": false,
  "deterministicAlgorithmsWarnOnly": false,
  "float32MatmulPrecision": "highest",
  "defaultDtype": "torch.float32",
  "defaultDevice": "cpu",
  "cpuCapability": "DEFAULT",
  "flushDenormal": false,
  "subnormalProbe": {
    "encoding": "ieee-f32-little-endian",
    "inputBits": 1,
    "multipliedByOneBits": 1
  },
  "cFloatingPointRoundingMode": "FE_TONEAREST",
  "mkldnnAvailable": false,
  "mkldnnEnabled": true
}
```

Torch has no getter for `set_flush_denormal`, so the adapter attests the
effective behavior rather than trusting the setter: it creates the minimum
positive IEEE-F32 subnormal and requires its bits to remain `1` after
multiplication by one. It also calls and reads back the process C floating-point
rounding mode. Unsupported setters, a failed `fesetround`, a changed CPU
capability or any readback difference fail before the requested BMM executes.

The authoritative execution contract, structural integrity manifest,
fidelity-gate navigation and every runtime-reduction receipt carry the same
closed object. An integration test starts the helper under both
`OMP_NUM_THREADS=1` and `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`; the embedded
program restores the pinned thread count and `highest` precision, then returns
the complete attestation.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 53:

- artifact bytes: `21,409,366,588`;
- artifact SHA-256:
  `e1fb30baa093263dd4f5400e6fc61847748b18ebc2f851dfba80e761f184a3fe`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 27,034 UTF-8 bytes / SHA-256
  `19c97ea27e525a89de8524140b5271113193027291cc3982b176c8ea82a2345d`;
- integrity sections: `24`;
- integrity root:
  `ed3cacb5a977bf4861da71003a25cfb174f1a4e46f83a33d1325bdf554c6b7a6`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present report has SHA-256
`06e7856a7dee8b29c9aa6ae5e949409b08d8cd411177eaad07502b22cb9f9430`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to distinct explicit
paths under `/private/tmp` with an `EXIT`/`INT`/`TERM` restoration trap. While
both declared paths were absent, the streaming reader recomputed the
24-section manifest and rehashed all 2,130 embedded payloads. It recorded
`sourceCheckpointAccessed=false`; the 43,234,663-byte report has SHA-256
`af4f39134c3b48d9119c2ef8f0469d976506513eace079c85d2a225da3594979`.

The same report rendered `layer_0_q_proj[0,0,0]` and substituted its first four
addressed BF16 learned values as exact decoded literals:

```text
-0.006195068359375
 0.0018463134765625
-0.0498046875
 0.033203125
```

The trap restored both paths and explicit postconditions verified that the
temporary paths no longer existed.

## Source-removed authoritative differential

In the same source-hidden window, the schema-53 artifact was compared at zero
absolute and relative tolerance with the immutable authoritative image, video
and audio captures. The provider used `venv/bin/python` and the adapter source
embedded in the artifact, not the checkout helper.

| modality | prefill | generation | KV layers | numeric-state-bound receipts |
| --- | --- | --- | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | 24 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | 24 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | 24 | 36 |

The aggregate records 3/3 zero-tolerance prefills, 3/3 zero-tolerance greedy
generations, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. All 100 receipts contain execution-state
schema 2, `float32MatmulPrecision=highest`, the successful subnormal probe and
`cFloatingPointRoundingMode=FE_TONEAREST`. Its 1,028,833-byte report has
SHA-256
`9a029d5207cff7bf7e3e7cf5ae4c2dff8c44c399e36560353120443eda49f6a0`.

## Preserved fidelity boundary

Schema 53 prevents mutable numeric process state from silently satisfying the
pinned native replay contract. It does not publish Apple Accelerate SGEMM's
scalar product rounding, lane, tile or fold schedule. The navigable scalar
audit therefore remains fail-closed for all 100 reductions and
`exactReplayClaim=forbidden` remains the only valid artifact claim.
