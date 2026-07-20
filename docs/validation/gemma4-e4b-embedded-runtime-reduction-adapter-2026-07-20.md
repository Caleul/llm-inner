# Gemma 4 E4B: integrity-bound embedded runtime-reduction adapter — 2026-07-20

## Completed boundary

Schema v45 fixed and attested the Torch/Apple Accelerate runtime used to replay
the 100 still-opaque vision/audio BMM reductions, but its executable Python
adapter remained a repository file outside the literal JSON and its integrity
root. A missing or modified checkout could therefore prevent or alter replay
even though the artifact declared the correct provider contract.

Schema v46 makes the adapter part of the calculation artifact:

- `authoritativeExecution` embeds the complete UTF-8 Python program, language,
  entrypoint, byte count and SHA-256;
- the authoritative contract validator accepts only the pinned 7,813-byte
  program with SHA-256
  `6a2d8ab61c3ae2d2b5ca90c908e1d76ff3761615f0f7c662ae63323b0b611674`;
- the existing 24-section integrity manifest commits the complete
  `authoritativeExecution` section, including those program bytes;
- the provider opens and validates the artifact, materializes only its embedded
  adapter in a private temporary directory, and removes it after each call;
- every replay receipt now binds the adapter SHA-256 in addition to the pinned
  runtime, ordered operands and output; and
- a tampered embedded program fails before replay.

The adapter source is loaded only while building the artifact. Opening and
executing schema v46 no longer reads
`helpers/torch_gemma4_runtime_reductions.py`.

## Real artifact and payload audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 46.

- artifact bytes: `21,409,286,964`;
- artifact SHA-256:
  `2d16f1c79553841d6b92d08216ca6987105f2e376b88e66a049dd9ff7b8c555f`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes; and
- integrity root:
  `1b246b3b11c205119321d6bed483289b7a9e74aeb11b00fa2ecb2ebe674fe093`.

The 1,012-byte source-present audit report has SHA-256
`92f5970ae47566a5b549bda3c6c2d93161c95a81d2b5347e93f9b72584a6e38e`.

## Checkpoint and repository adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were both moved to explicit
`/private/tmp` paths under an `EXIT`/`INT`/`TERM` restoration trap. While both
paths were absent:

- the streaming reader revalidated the 24-section integrity manifest and all
  2,130 embedded payloads;
- the payload result recorded `sourceCheckpointAccessed=false`, compared
  `15,992,314,836` bytes and reproduced the literal storage digest above; and
- an isolated vision score BMM executed through the adapter extracted from the
  artifact, producing an attested receipt with the embedded adapter hash.

The 43,039,081-byte source-removed report has SHA-256
`e7944eff09a29f1a094a01af2aa797f3cc6e180b4bfebe4599f39a6db2a65702`.
The 1,222-byte adapter smoke receipt has SHA-256
`2bda6f6696b6cbc4ca89809247a5e422b2d64402759ede803c8417ee5b6b993c`.
The trap restored both paths.

## Source-removed authoritative differential

With the checkpoint and repository adapter absent again, the schema-v46
artifact was compared at zero absolute and relative tolerance against the
immutable authoritative composite captures. The provider for all 100 native
reductions was instantiated from the artifact itself.

| modality | prefill | generation | native reductions | embedded-adapter receipts |
| --- | --- | --- | ---: | ---: |
| image | lossless-within-dtype | lossless-within-dtype | 32 | 32 |
| video | lossless-within-dtype | lossless-within-dtype | 32 | 32 |
| audio | lossless-within-dtype | lossless-within-dtype | 36 | 36 |

All three prefill comparisons and all three greedy generations passed with
`firstDivergence=null`. Every receipt carries adapter SHA-256
`6a2d8ab61c3ae2d2b5ca90c908e1d76ff3761615f0f7c662ae63323b0b611674`.
The 368,265-byte suite report has SHA-256
`52847f99252cf4a6df53e3758b36a25565909ef6764ad2c91813cc80ea19e396`.
The trap restored both paths after the suite.

## Preserved fidelity boundary

Schema v46 removes the repository adapter as an undeclared replay dependency.
It does not invent Apple Accelerate SGEMM's unpublished scalar
product-rounding, lane, tile or fold schedule. The mathematical scalar view
therefore remains `fail-closed-runtime-reduction`, the gate remains
`blocked-on-runtime-reduction`, all 100 entries remain addressable, and
`exactReplayClaim=forbidden`.

Fresh repository validation after implementation and regeneration:

```text
npm run typecheck
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
```

The suite reported 257 tests: 254 passed, 0 failed and 3 explicitly optional
non-Gemma Transformers integrations were skipped. The Gemma 4 embedded-adapter
test executed against the pinned local Torch 2.12.1 runtime.
