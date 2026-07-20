# Gemma 4 E4B: closed runtime process environment — 2026-07-20

## Completed boundary

Schema 54 closes the launch environment omitted by schema 53 around the 100
Gemma 4 BMM operations whose Apple Accelerate scalar schedule remains
unpublished. The embedded adapter previously received the complete parent
environment even though its receipts attested only the host, runtime binaries,
dependency trees and post-import Torch/C numeric state. Loader and dispatch
variables could therefore reach the native path without becoming artifact data.

`runtimeProcessEnvironment` now declares:

```json
{
  "schemaVersion": 1,
  "inheritance": "none",
  "variables": {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x47"
  }
}
```

`Gemma4TorchRuntimeReductionProvider` passes exactly this map to `spawnSync`
instead of inheriting `process.env`. The adapter captures the complete launch
environment before importing Torch and rejects any extra, missing or changed
entry before reading artifact operands. The authoritative execution contract,
structural integrity manifest, fidelity-gate navigation and every reduction
receipt carry the same closed object.

The integration test contaminates the parent with
`OMP_NUM_THREADS=1`, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`,
`VECLIB_MAXIMUM_THREADS=1` and an invalid `DYLD_INSERT_LIBRARIES`. The embedded
helper still launches successfully, reports only the serialized map and
restores the already-pinned Torch numeric state. Independent tests also mutate
the artifact process-environment contract and provider receipt and verify
fail-closed rejection.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 54:

- artifact bytes: `21,409,381,135`;
- artifact SHA-256:
  `f56cd4f125b87aa094ffa2e9b57cd1f0cffef10ba1f9a360f9538b39ce2d7585`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 28,275 UTF-8 bytes / SHA-256
  `5fa0286427c92a557a4893219779f6eb11b7229e5f63d49863ed11ad18c74950`;
- integrity sections: `24`;
- integrity root:
  `0e452eaf9d365e25b9fa9206670ac29808a3f8688a4b2249af8528ae13995621`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present report has SHA-256
`3d443599f33136919f5a69e9a070b1df9cb82f6a77dca5c886ee62f64e2ecab9`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to one explicit private
temporary directory under an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the streaming reader recomputed the 24-section
manifest and rehashed all 2,130 embedded payloads. It recorded
`sourceCheckpointAccessed=false`; the 43,250,450-byte report has SHA-256
`e39d35d3bf9ecfa181b42ff55cd1ee87c5c1adfea7603adce5b2a3c15f1204c8`.

The same report rendered `layer_0_q_proj[0,0,0]` and substituted its first four
addressed BF16 learned values as exact decoded literals:

```text
-0.006195068359375
 0.0018463134765625
-0.0498046875
 0.033203125
```

The trap restored both paths, and explicit postconditions verified that the
temporary directory no longer existed.

## Source-removed authoritative differential

In the same source-hidden window, the schema-54 artifact was compared at zero
absolute and relative tolerance with the immutable authoritative image, video
and audio captures. The provider used `venv/bin/python` and the adapter source
embedded in the artifact, not the checkout helper.

| modality | prefill | generation | token / position | KV layers | environment-bound receipts |
| --- | --- | --- | --- | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 36 |

The aggregate records 3/3 zero-tolerance prefills, 3/3 zero-tolerance greedy
generations, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. All 100 receipts declare
`runtimeProcessEnvironment.schemaVersion=1`, `inheritance=none`, the exact
closed variable map and `sourceCheckpointAccessed=false`. Its 1,075,833-byte
report has SHA-256
`0e0afa3b9c8c8a38edca5dde618ab5047ea3c8d772ff20400ed50c360cd92f44`.

## Preserved fidelity boundary

Schema 54 prevents inherited process state from silently changing the pinned
native replay path. It does not publish Apple Accelerate SGEMM's scalar product
rounding, lane, tile or fold schedule. The navigable scalar audit therefore
remains fail-closed for all 100 reductions, and `exactReplayClaim=forbidden`
remains the only valid artifact claim.

## Commands

```text
npm run typecheck
npm test
git diff --check
```

The real artifact regeneration, complete source comparison, source-removed
payload verification and zero-tolerance modality suite all exited zero.
