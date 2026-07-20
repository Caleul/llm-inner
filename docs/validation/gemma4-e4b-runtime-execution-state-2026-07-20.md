# Gemma 4 E4B: pinned runtime execution state — 2026-07-20

## Completed boundary

Schema 52 closes the effective Torch process state around the 100 native BMM
operations whose Apple Accelerate scalar schedule remains unpublished. Schema
51 already bound the host, runtime binaries, Python source trees and BLAS
shared-cache image, but a process with different thread or algorithm settings
could still present those same identities.

`runtimeExecutionState` now declares:

```json
{
  "schemaVersion": 1,
  "intraopThreads": 10,
  "interopThreads": 14,
  "deterministicAlgorithms": false,
  "mkldnnAvailable": false,
  "mkldnnEnabled": true
}
```

The embedded adapter configures the mutable fields, reads the effective state
back through Torch, and rejects any difference before `torch.matmul`. The
authoritative execution contract, structural integrity manifest, fidelity-gate
navigation and every runtime-reduction receipt contain the same closed object.
Independent tests mutate the artifact contract and provider receipt and verify
that both paths fail closed. The embedded-adapter integration starts its helper
with `OMP_NUM_THREADS=1` and proves that the serialized configuration restores
and attests the authoritative 10-thread intraop state before matmul.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 52:

- artifact bytes: `21,409,364,037`;
- artifact SHA-256:
  `4004e48c072e535af1e0fdd86f50b3c4b1bcf1fb6d9f09265db276db230e69c0`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 24,921 UTF-8 bytes / SHA-256
  `86b33db05f281013ee4214dd9caa18bc9db2f910a903707493ae2543cee2d6c8`;
- integrity sections: `24`;
- integrity root:
  `5da4fad4c816f9cd782ee740f0d25c1aee4937b7d56da0ebfd6f9e75c98d4366`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present report has SHA-256
`4e7e96de545a5b96161d05b7932a2ef8821a9b150f3cbc63063d8bd6efc59689`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to explicit private
temporary paths under an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the streaming reader recomputed the 24-section
manifest and rehashed all 2,130 embedded payloads. It recorded
`sourceCheckpointAccessed=false`; the 43,141,980-byte report has SHA-256
`ee388d900ad421563371c296e536d46f1c713c90c385601ec0a55d9a8d6af2fc`.
The trap restored both paths and explicit postconditions verified them.

## Source-removed authoritative differential

In the same source-hidden window, the schema-52 artifact was compared at zero
absolute and relative tolerance with the immutable authoritative image, video
and audio captures. The provider used `venv/bin/python` and the adapter source
embedded in the artifact, not the checkout helper.

| modality | prefill | generation | token / position | KV layers | execution-state-bound receipts |
| --- | --- | --- | --- | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 36 |

All 100 receipts contain the exact declared `runtimeExecutionState`. The
aggregate records 3/3 zero-tolerance prefills, 3/3 zero-tolerance greedy
generations, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. Its 973,051-byte report has SHA-256
`db8ed730ab9e759c8b235ebe38c2c3e6c97e94e5326b2f265abf7c747be5c239`.

## Preserved fidelity boundary

Schema 52 prevents thread-count and Torch algorithm-state drift from silently
satisfying the native replay contract. It does not publish Apple Accelerate
SGEMM's scalar product rounding, lane, tile or fold schedule. The scalar audit
therefore remains fail-closed for all 100 reductions and
`exactReplayClaim=forbidden` remains the only valid artifact claim.

## Commands

```text
npm run typecheck
npm test
git diff --check
```

The real artifact regeneration, complete source comparison, source-removed
payload verification and zero-tolerance modality suite all exited zero.
Typecheck exited zero. The repository suite reported 258 tests: 255 passed,
0 failed and 3 explicitly optional non-Gemma Transformers integrations were
skipped. Both Gemma 4 embedded-adapter tests executed against the pinned local
Torch runtime, including all five invocation classes.
