# Gemma 4 E4B: artifact-owned execution protocol — 2026-07-20

## Completed boundary

Schema 54 made the embedded adapter, invocation programs, closed process
environment and runtime attestations artifact-owned, but the TypeScript
provider still hard-coded the temporary filenames, working directory,
subprocess argument order, channel semantics, output bound and JSON envelope
shape. The caller-selected Python executable was also an undeclared resolver
input. The replay therefore retained launch semantics outside the JSON.

Schema 55 adds one integrity-bound `executionProtocol` to the authoritative
replay contract. It declares:

- a caller-supplied Python 3 path whose loaded runtime must satisfy the embedded
  `runtimeEnvironmentIdentity` attestation;
- `os.tmpdir`, the temporary-directory prefix, recursive-finally cleanup and
  temporary working directory;
- the adapter and request file roles, names, UTF-8 encoding and artifact-owned
  sources;
- synchronous argument order `adapter-file, request-file`;
- the serialized closed environment, empty stdin, single JSON stdout, UTF-8
  diagnostic stderr and a 134,217,728-byte output bound; and
- exact ordered request, response and tensor field sets.

The provider materializes those values from the artifact. Unknown protocol
versions, changed argument order, changed files and extra or missing envelope
fields fail closed. Runtime-reduction evidence schema 3 binds every receipt to
the protocol SHA-256. The fidelity gate is schema 4 and gives every unresolved
BMM a stable pointer to the same protocol.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 55:

- artifact bytes: `21,409,393,890`;
- artifact SHA-256:
  `0b0ed187750a41e0c42359e8211b98701a8154ebbb4737f0ef1997187cda63ce`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- structural integrity sections: `24`;
- integrity root:
  `c706a9390c6f16473d465c66ae23edff7c3554d9b990acd8309d300d0fe45abe`;
- execution-protocol SHA-256:
  `01b0cdd4a07e2e473dd0c3ae783ae3237ea77bdbd5ff9ff738f67ca2341fd559`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source-present report has SHA-256
`5783f9685a82bd4b772dd97b653fcec436b0fc891e6fa7da2333e36ab1b816a4`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to one explicit private
temporary directory under an `EXIT`/`INT`/`TERM` restoration trap. While both
paths were absent, the streaming reader validated the structural manifest and
rehashed every embedded payload. It reported 2,130 constants and
`sourceCheckpointAccessed=false`. The 43,265,284-byte report has SHA-256
`6a0fdacc96f1dae8783b02bb877eb1bc02eb4f0d69eb3605ba31ae4d1755dc5c`.

The same report rendered `layer_0_q_proj[0,0,0]` and substituted its first four
addressed BF16 learned values as exact decoded literals:

```text
-0.006195068359375
 0.0018463134765625
-0.0498046875
 0.033203125
```

Postconditions verified that both product paths were restored and the private
temporary directories were removed.

## Source-removed authoritative differential

The first attempted capture set failed closed because it declared the older
runtime string and omitted `executionMode` and `attentionImplementation`. No
fallback interpretation was accepted. The successful run used captures that
explicitly declare the required Transformers 5.5.0 / Torch 2.12.1 composite
runtime, `torch.inference_mode` and eager attention.

With source and checkout adapter still absent, zero-tolerance comparison passed:

| modality | prefill | generation | receipts | protocol SHA-256 |
| --- | --- | --- | ---: | --- |
| image | lossless-within-dtype | lossless-within-dtype | 32 | `01b0cdd4...1fd559` |
| video | lossless-within-dtype | lossless-within-dtype | 32 | `01b0cdd4...1fd559` |
| audio | lossless-within-dtype | lossless-within-dtype | 36 | `01b0cdd4...1fd559` |

The aggregate records 3/3 zero-tolerance prefills, 3/3 zero-tolerance greedy
generations, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. All receipts use evidence schema 3, carry the
complete protocol digest and embedded-adapter digest, and cover all five
invocation classes. The 1,086,733-byte report has SHA-256
`d7770f882e45f5b0d92999abf568fde522c484229dfdebca341b934215e53885`.

## Preserved fidelity boundary

Schema 55 eliminates hidden launch and transport conventions from executable
replay. It does not publish Apple Accelerate SGEMM's scalar product rounding,
lane, tile or fold schedule. The 100 BMM scalar reductions therefore remain
fail-closed and `exactReplayClaim=forbidden` remains the only valid claim.

## Validation

```text
npm run typecheck
npm test
git diff --check
```

The focused schema/provider tests, real artifact regeneration, complete
source-to-artifact byte comparison, source-removed payload verification and
zero-tolerance multimodal suite all exited zero.
