# Gemma 4 E4B: runtime-reduction transcript commitments — 2026-07-20

## Completed boundary

Schema 56 transported every native-BMM operand and output losslessly, but its
execution receipt committed only the invocation-program digest, operand/output
hashes and runtime attestation. The exact request envelope was not committed.
In particular, tower values such as `attentionHeads`, `headDim`,
`attentionChunkSize`, `attentionContextLeft` and `attentionContextRight` can
change reshape, padding and window semantics without appearing in the receipt.

Schema 57 closes that evidence gap:

- `executionProtocol` schema 3 embeds a transcript commitment over the exact
  request-file and stdout UTF-8 bytes;
- both envelopes use compact JSON without whitespace or trailing bytes;
- each execution receipt schema 5 records byte count and SHA-256 for request
  and response;
- request reconstruction includes the serialized invocation program, closed
  tower environment and both lossless Base64 operand envelopes;
- response reconstruction includes operation identity, complete runtime
  attestation and the lossless output envelope; and
- the caller independently reconstructs both transcripts before accepting a
  provider receipt.

The transport responsibility now lives in
`src/gemma4-runtime-reduction-transport.ts`; the Torch provider only
materializes the artifact-owned launch and records the exact bytes it sent and
received. A regression changes only `attentionContextRight` and proves the
request digest changes. A forged digest and a response with a trailing newline
both fail closed.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 57:

- artifact bytes: `21,409,396,649`;
- artifact SHA-256:
  `6462872d8ab70dce290fae3c14274febea4d568264bf68e078e09231576ac532`;
- authoritative-execution schema: `13`;
- execution-protocol schema: `3`;
- embedded-adapter SHA-256:
  `ba7042d6f6771d14d1b1e51c752d147b18146da83e257014a5bfb46af99850ad`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes; and
- structural integrity root:
  `70e21fcf025b1ef44cb350c033df634260e1f99a8aae774185f5fbcbf03f8687`.

The 1,012-byte source audit has SHA-256
`cc6813098868ab656d982a8ed86531c5c4c5f39c5e64643a0e08bccd6c20381a`.

## Fresh authoritative captures

Fresh Transformers 5.5.0 / Torch 2.12.1 CPU eager
`torch.inference_mode` captures were produced from the immutable package:

| modality | trace bytes | trace SHA-256 |
| --- | ---: | --- |
| image | 8,232,797 | `48a7270be75d44ae31809ae04c77dd419b22937bd3ec5fdc1e4e705f031af68b` |
| video | 8,232,807 | `10618977609781f65a7c325c7cbff3018cfb9b2ad280cd1a9bdde898bdaa7e9d` |
| audio | 8,102,175 | `d07227accbccdd911f2e5d66b0152e12f8d908ad480503e45b55d8344fac0610` |

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved into one private
`mktemp` directory under an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the literal reader rehashed all 2,130 payloads and
reported `sourceCheckpointAccessed=false`. It also rendered
`layer_0_q_proj[0,0,0]` from the embedded BF16 decoder and substituted the
first four exact learned literals. The 43,268,340-byte inspection has SHA-256
`6c50f21fc4171aab631614c8f3a57282a1e717f017ba92e8c613564181400c1d`.

In a separate source-hidden window, the multimodal suite used only the schema
57 artifact and the pinned `venv/bin/python` runtime. It matched the fresh
authoritative captures at zero absolute and relative tolerance:

| modality | prefill | generation | token / position | caches | receipts |
| --- | --- | --- | --- | ---: | ---: |
| image | 5/5 pass | lossless-within-dtype | `184 / 2` | 24/24 | 32 |
| video | 5/5 pass | lossless-within-dtype | `184 / 2` | 24/24 | 32 |
| audio | 5/5 pass | lossless-within-dtype | `184 / 2` | 24/24 | 36 |

All 100 receipts use evidence schema 5, contain valid request and response
SHA-256 values, and carry execution-protocol digest
`4d6b55515bf100d36fb1ab24d85d50887d2b667dab258382c3ab5ec1883bdc23`.
The aggregate records 3/3 prefill passes, 3/3 greedy-generation passes,
`firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. The 1,131,835-byte report has SHA-256
`c6e19b8f95635ddbd39d5d5236c1ae1809daf5a2550426b1ad14e13c29a645bc`.
Both restoration traps restored the checkpoint and helper and removed their
private directories.

## Preserved fidelity boundary

The transcript commitments prove which complete request and response crossed
the artifact-owned provider boundary. They do not publish Apple Accelerate
SGEMM's scalar product rounding, lane, tile or fold schedule. The same 100 BMM
scalar reductions therefore remain `fail-closed-runtime-reduction`, and the
integrity-bound gate remains `blocked-on-runtime-reduction` with
`exactReplayClaim=forbidden`.

## Validation

```text
npm run typecheck
npm test
git diff --check
```

The focused transcript/transport/provider tests, real artifact regeneration,
complete source-to-artifact payload comparison, source-removed payload audit,
scalar substitution and zero-tolerance three-modality suite all exited zero.
The final repository run reported 260 tests: 257 passed, 0 failed and 3
explicitly optional non-Gemma integrations skipped. `git diff --check` also
exited zero.
