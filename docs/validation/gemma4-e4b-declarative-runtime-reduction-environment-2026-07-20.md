# Gemma 4 E4B: declarative native-reduction environment — 2026-07-20

## Completed boundary

Schema 47 embedded the complete stage graph for all five remaining native BMM
classes, but the TypeScript provider still selected tower values with a
`vision`/`audio` branch and the embedded Python adapter hard-coded their lower
bounds. The JSON therefore described the tensor transforms while part of the
invocation environment remained duplicated outside those declarations.

Schema 48 closes that boundary:

- every invocation program is schema 2 and declares its runtime-dtype binding;
- every consumed tower parameter declares its source path, `safe-integer`
  domain and inclusive minimum;
- the TypeScript provider materializes the environment by iterating only those
  serialized bindings, without modality dispatch;
- the embedded adapter validates the same closed environment before executing
  any stage;
- missing, extra, duplicate, altered-source, non-integer and out-of-range
  bindings fail closed; and
- the fidelity gate points each of the 100 native reductions to one of these
  integrity-bound invocation programs.

This is operation-, environment- and source-contract driven. It does not add a
layer or invocation allowlist and does not invent Apple Accelerate's scalar
reduction schedule.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 48.

- artifact bytes: `21,409,324,470`;
- artifact SHA-256:
  `97f7cea6d0b2b183ae6936ea8ade402501bae2cbadc92b703335cb51a105c420`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 14,484 UTF-8 bytes / SHA-256
  `b20bc36807dda69dd34fb9b47bada84abb0969f696e949b0cc8ad0ae9ddb302b`;
- integrity root:
  `eaa70f0d3a464829bd6ba19b894043160e0ffad9faf0852699e92b1eacab9894`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present audit report has SHA-256
`a59ca2e1d77c4ecee760cb33181fbfff0c826eb25e5edf6e1f27a7a5a7cf4f10`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to explicit
`/private/tmp` paths under an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the streaming reader revalidated the 24-section
integrity manifest and all 2,130 embedded payloads. The report records
`sourceCheckpointAccessed=false`, compares `15,992,314,836` bytes and
reproduces the literal storage digest above.

The 43,098,508-byte source-removed payload report has SHA-256
`3ccd3e2125178b8c9d4b3269f0ea518cd6bc22e4cedc2660a5fd548c7b632fdf`.
The restoration trap returned both paths after validation.

## Source-removed authoritative differential

In the same source-hidden window, the artifact was compared at zero absolute
and relative tolerance with the immutable authoritative composite captures.
The provider was instantiated from the artifact's embedded adapter and
declarative invocation programs, using the pinned local Torch 2.12.1 build
`7269437d655783a26cba32aa88195b741ff496aa` on Darwin arm64 with Apple
Accelerate.

| modality | prefill | generation | KV layers | native reductions | receipts |
| --- | --- | --- | ---: | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | 24 | 32 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | 24 | 32 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | 24 | 36 | 36 |

All three prefills and all three one-token greedy generations emitted token
`184`, passed at zero tolerance and reported `firstDivergence=null`. The
385,633-byte suite report has SHA-256
`f3ad0ad2fc98c66e197e6e97556506389e1170087c31b1344f60cae7d3a58343`.

## Preserved fidelity boundary

Schema 48 removes hidden modality dispatch and parameter-domain decisions from
the executable invocation environment. Apple Accelerate still does not publish
its product rounding, lane, tile or fold schedule. The scalar audit therefore
continues to expose every operand product and fail closed at those 100
reductions; `exactReplayClaim` remains `forbidden`.

Fresh repository validation after implementation and regeneration:

```text
npm run typecheck
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
```

The suite reported 258 tests: 255 passed, 0 failed and 3 explicitly optional
non-Gemma Transformers integrations were skipped. The Gemma 4 embedded-adapter
tests executed against the pinned local Torch 2.12.1 runtime.
