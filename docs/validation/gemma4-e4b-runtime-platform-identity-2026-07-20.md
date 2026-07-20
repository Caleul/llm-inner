# Gemma 4 E4B: integrity-bound runtime platform identity — 2026-07-20

## Completed boundary

Schema 48 fixed the embedded adapter, Torch build commit, Darwin-arm64,
Apple Accelerate backend and declarative invocation environment for all 100
remaining native BMM reductions. That contract could still accept a different
macOS build or Apple CPU generation. Those hosts may dispatch another SGEMM
implementation while reporting the same coarse platform and Torch commit.

Schema 49 closes that environment-identity gap:

- the authoritative execution contract embeds `runtimeEnvironmentIdentity`;
- the identity fixes CPython implementation/version, macOS product version and
  build, Darwin kernel release, machine model, CPU brand and the SHA-256 of the
  complete `torch.__config__.show()` output;
- the embedded adapter derives every field from the live runtime before the
  first `torch.matmul` and rejects any divergence;
- every native-reduction receipt carries the same complete identity;
- the TypeScript provider compares the receipt with the artifact contract;
- every unresolved entry in fidelity-gate schema 2 navigates to the shared
  environment identity as well as its operation-specific invocation program;
  and
- altered artifact identity or altered receipt identity fails closed.

The E4B replay identity is:

```json
{
  "pythonImplementation": "CPython",
  "pythonVersion": "3.14.3",
  "operatingSystem": "macOS",
  "operatingSystemVersion": "26.5.2",
  "operatingSystemBuild": "25F84",
  "kernelRelease": "25.5.0",
  "machineModel": "Mac15,10",
  "cpuBrand": "Apple M3 Max",
  "torchBuildConfigSha256": "606e3853213dea3faabc6d58b66ed7e419ee4452a6d53c2b27495a2ecc4e07a7"
}
```

This is an operation-, runtime- and source-contract boundary. It does not
invent Apple Accelerate's unpublished scalar product rounding, lane, tile or
fold schedule.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 49.

- artifact bytes: `21,409,339,988`;
- artifact SHA-256:
  `76eaf79b9eb4d8e4e0f651085aa9871767ebc454126e406fa2e90cc220e31ef8`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- embedded adapter: 16,325 UTF-8 bytes / SHA-256
  `77e3b2ba6c0d57aff3470447db85b1994262917ead82ada084d86b62544def34`;
- integrity root:
  `892684e8a198ceb1f28c4a57bd7020f032b2376334890fc8ac1ec78aa4ce79df`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present audit report has SHA-256
`57edaf885c7a9113c5de5299e59ea7831d363ecf3d096b3d6bea92f2d42b2f6d`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to a private temporary
directory under an `EXIT`/`INT`/`TERM` restoration trap. While both declared
paths were absent, the streaming reader revalidated the 24-section integrity
manifest and all 2,130 embedded payloads. The report records
`sourceCheckpointAccessed=false`, compares `15,992,314,836` bytes and
reproduces the literal storage digest above.

The 43,115,337-byte source-removed payload report has SHA-256
`c980b3c4753b9565e777f843d1b27c4b75f235c93d3fe806479de014a32fa01c`.
The trap restored both paths after validation.

## Source-removed authoritative differential

In the same source-hidden window, the artifact was compared at zero absolute
and relative tolerance with the immutable authoritative image, video and audio
captures. The provider was instantiated only from the schema-49 artifact and
`venv/bin/python`; its repository adapter path remained absent.

| modality | prefill | generation | KV layers | native reductions | receipts |
| --- | --- | --- | ---: | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | 24 | 32 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | 24 | 32 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | 24 | 36 | 36 |

All three prefills and all three one-token greedy generations emitted token
`184`, passed at zero tolerance and reported `firstDivergence=null`. Every one
of the 100 receipts contains adapter SHA-256 `77e3b2ba...f34`, the matching
invocation-program SHA-256 and the complete environment identity above. The
442,033-byte suite report has SHA-256
`e413505e73be581fc6b070225c51d3fe5b64841b0967b403f3e9540bf5c42b91`.

## Preserved fidelity boundary

Schema 49 prevents an OS or CPU dispatch change from silently satisfying the
pinned native-replay contract. Apple Accelerate still does not publish its
scalar schedule. The audit continues to expose every operand product and fail
closed at those 100 reductions; `exactReplayClaim` remains `forbidden`.

Fresh repository validation after implementation and regeneration:

```text
npm run typecheck
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
```

The suite reported 258 tests: 255 passed, 0 failed and 3 explicitly optional
non-Gemma Transformers integrations were skipped. The Gemma 4 embedded-adapter
tests executed against the pinned local Torch 2.12.1 runtime. An additional
run pointed those three integrations at the Gemma 4 environment and confirmed
their independent fail-closed version guard: they require Transformers 4.57.1
and Torch 2.7.1 rather than the pinned Gemma runtime's 5.5.0 / 2.12.1.
