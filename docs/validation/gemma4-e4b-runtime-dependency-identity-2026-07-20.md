# Gemma 4 E4B: integrity-bound runtime dependency identity — 2026-07-20

## Completed boundary

Schema 50 bound the CPython executable, four Torch native files and the
Accelerate BLAS shared-cache image used by the 100 still-opaque native BMMs.
The embedded adapter also executes Python code from the CPython standard
library and Torch package, and the declared native libraries link four other
bundled Torch/OpenMP files. Those dependencies were not part of the artifact
identity, so byte drift outside the schema-50 subset could change replay while
retaining the recorded runtime versions and hashes.

Schema 51 replaces that subset with a closed `runtimeDependencyIdentity`:

- nine exact native files are ordered by semantic role, byte count and SHA-256:
  the CPython runtime, `torch._C`, `libtorch_python.dylib`,
  `libtorch_cpu.dylib`, `libc10.dylib`, `libtorch.dylib`, `libshm.dylib`,
  `libomp.dylib` and `libtorch_global_deps.dylib`;
- the CPython standard-library tree commits 1,848 `.py`/`.pyi` files and
  35,754,693 bytes;
- the Torch Python-package tree commits 2,230 `.py`/`.pyi` files and
  46,148,427 bytes;
- every tree leaf is canonicalized as relative POSIX path, byte count and
  content SHA-256 in lexical order before the tree SHA-256 is calculated;
- `__pycache__` is excluded from both source trees and `site-packages` from the
  standard-library tree, so generated caches and unrelated packages are not
  silently promoted into the runtime contract; and
- the existing Accelerate `libBLAS.dylib` install name, `arm64e` architecture
  and Mach-O UUID remain mandatory.

The adapter recalculates all 4,078 source leaves / 81,903,120 source bytes and
all nine native-file identities before any invocation program reaches
`torch.matmul`. The TypeScript receipt validator requires the same immutable
object. Altering either a native-file digest or a Python source-tree digest is
covered by independent fail-closed tests. The contract remains operation-class
driven and applies to all five serialized native-BMM invocation programs.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 51.

- artifact bytes: `21,409,350,146`;
- artifact SHA-256:
  `33070a641fdd2d891310a789e4bc7508ebb6ccb0469d29d5577e9339a4ee5099`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 23,465 UTF-8 bytes / SHA-256
  `6cb4b2d076a08536eabeb4bfaa2a5368de75f88b904aad291fd847e39d725b08`;
- integrity sections: `24`;
- integrity root:
  `0c97380a3ae5d6599e29c4b45eda9d6c546a6c3e58259292f674a74be591a41c`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present audit report has SHA-256
`359fd8f00212ff607cce24e76f2f6d5675488222b247bf2644f42afd652e1776`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to explicit private
temporary paths under an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the streaming reader recomputed the 24-section
manifest and rehashed all 2,130 embedded payloads. It recorded
`sourceCheckpointAccessed=false`; the 43,126,998-byte report has SHA-256
`034dc1d49d01b4a5ff685e1ee36eced0330a358b85e01e005fab65ce9ff3ad1a`.
The trap restored both paths, and explicit postconditions verified them.

## Source-removed authoritative differential

In the same source-hidden window, the schema-51 artifact was compared at zero
absolute and relative tolerance with the immutable authoritative image, video
and audio captures. The provider was created from `venv/bin/python` and the
adapter embedded in the artifact; it did not read the checkout adapter.

| modality | prefill | generation | token / position | KV layers | dependency-bound receipts |
| --- | --- | --- | --- | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | `184 / 2` | 24 | 36 |

The aggregate records 3/3 zero-tolerance prefills, 3/3 zero-tolerance greedy
generations, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. Its 941,433-byte report has SHA-256
`aeb8aa479e927b7d2e18dfaac9f1efca01f6facfd38f859f36da2031ca52ea55`.

## Preserved fidelity boundary

Schema 51 prevents drift in the declared Python sources and bundled native
libraries from silently satisfying the executable replay contract. It does not
publish Apple Accelerate SGEMM's scalar product rounding, lane, tile or fold
schedule. The scalar audit therefore still exposes every input product and
fails closed at all 100 reductions; `exactReplayClaim=forbidden` remains the
only valid artifact claim.

## Fresh repository validation

```text
npm run typecheck
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
git diff --check
```

Typecheck and whitespace validation exited zero. The test suite reported 258
tests: 255 passed, 0 failed and 3 explicitly optional non-Gemma Transformers
capture integrations were skipped. The two Gemma 4 embedded-adapter tests ran
against the pinned local Torch runtime, including all five serialized
invocation programs.
