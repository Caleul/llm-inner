# Gemma 4 E4B: integrity-bound runtime binary identity — 2026-07-20

## Completed boundary

Schema 49 pinned the runtime version, Torch commit/configuration, macOS build,
kernel, CPU and machine model used by the 100 still-opaque native BMMs. A
different CPython or Torch binary could nevertheless retain all of those
identifiers. That left the executable replay contract coarser than the code
which actually reaches Apple Accelerate.

Schema 50 adds a closed `runtimeBinaryIdentity` to the authoritative contract,
embedded adapter, integrity root and every execution receipt. Before the first
`torch.matmul`, the adapter now derives and compares:

- byte count and SHA-256 of the CPython runtime selected by
  `sys.base_prefix/Python`;
- byte count and SHA-256 of `torch._C`, `libtorch_python.dylib`,
  `libtorch_cpu.dylib` and `libc10.dylib`; and
- install name, `arm64e` architecture and Mach-O UUID
  `F078C775-D8DC-3C4D-879F-A9BB228DBE06` for Accelerate
  `libBLAS.dylib` in the dyld shared cache.

The locators are semantic and independent of the checkout path. The file list,
roles and ordering are exact; a missing, extra, resized or byte-different file,
or a different shared-cache BLAS image, fails before native calculation. The
TypeScript receipt validator requires the same immutable identity.

A live scalar probe also preserved the reason this boundary must not clear the
fidelity gate. Seven ordered/lane candidates all reproduced the 1,944 BF16
vision-score coordinates because the final BF16 cast erased their differences.
For the F32 audio position BMM, the same candidate family missed 46 to 79 of
1,248 coordinates. Neither result identifies Apple Accelerate's tile and fold
schedule, so no candidate was installed.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 50.

- artifact bytes: `21,409,344,765`;
- artifact SHA-256:
  `4d334b8358b1f0fd19a30882d017b0eb9d0c8dd6d8fa38d60cc16346cde268d6`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 19,728 UTF-8 bytes / SHA-256
  `705474b719aaecb639400a0997e2309eee02de223a779feaedf6bfe9d1648e1f`;
- integrity root:
  `eef690cf2c58ace5c59632598c03dd7a3d97c039a349e9992ccdd85ec73be7c2`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present audit report has SHA-256
`5eeef49c490293ec9e883543bb118d28898788e7056e9145748a9653c1961a4b`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved under one private
temporary directory with an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the streaming reader revalidated the 24-section
integrity manifest and every embedded payload. It recorded
`sourceCheckpointAccessed=false`, compared all `15,992,314,836` learned bytes
and reproduced the literal storage digest above. The trap restored both paths.

The 43,120,725-byte source-removed report has SHA-256
`a038592504379d05d01dad2604290fd2555ca0556938c6b3bbe90f1e24a73570`.

## Source-removed authoritative differential

With the checkpoint and checkout adapter absent again, the schema-50 artifact
was compared at zero absolute and relative tolerance with the immutable
authoritative image, video and audio captures. The provider was constructed
only from `venv/bin/python` and the adapter embedded in the JSON.

| modality | prefill | generation | token | KV layers | binary-bound receipts |
| --- | --- | --- | ---: | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | 184 | 24 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | 184 | 24 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | 184 | 24 | 36 |

All three prefills and all three one-token greedy generations passed with
`firstDivergence=null`. Every receipt repeats all five file hashes and the
Accelerate BLAS Mach-O UUID. The 661,251-byte suite report has SHA-256
`bb8a9de2d062f8b4ce72940fb9899b8cbb11051ebedb34c99d85d5759f8f19e3`.

## Preserved fidelity boundary

Schema 50 prevents a byte-different Python/Torch kernel chain or shared-cache
BLAS image from silently satisfying the executable replay contract. It does
not publish Apple Accelerate SGEMM's scalar product rounding, lane, tile or fold
schedule. The mathematical scalar view therefore remains
`fail-closed-runtime-reduction`, all 100 reductions remain addressable, and
`exactReplayClaim=forbidden` remains correct.

Fresh repository validation after implementation and regeneration:

```text
npm run typecheck
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
```

The final suite reported 258 tests: 255 passed, 0 failed and 3 explicitly
optional non-Gemma Transformers integrations were skipped. The embedded Gemma
adapter tests executed against the pinned local runtime and all five serialized
invocation programs.
