# Gemma 4 E4B: lossless runtime tensor transport — 2026-07-20

## Completed boundary

Schema 55 made the native BMM execution protocol artifact-owned, but its
operand and output tensors still crossed the TypeScript/Python boundary as JSON
decimal arrays. That representation was not lossless for every finite IEEE-F32
bit pattern: ECMAScript JSON serialization writes `-0` as `0`, and the envelope
did not declare dtype bits, byte order, layout, byte length or index-to-byte
decoding. Exact operand/output hashes were computed outside that transport, so
they did not repair the lossy channel itself.

Schema 56 replaces every runtime-reduction tensor with a closed binary
envelope:

- `dtype=F32` and `bitPattern=IEEE-754 binary32`;
- little-endian, row-major-contiguous storage;
- explicit shape and `byteLength`;
- canonical RFC 4648 Base64 with required padding; and
- byte address `4 * row-major-linear-index(shape, coordinate)`.

The TypeScript provider writes and reads each float with explicit little-endian
operations. The embedded Python adapter decodes the signed `int32` bit patterns
and views them as F32 before any declared cast or `torch.matmul`; it serializes
the output through the inverse bit-preserving program. Both sides reject wrong
fields, dtype, layout, byte order, cardinality, byte length, non-canonical
Base64 and non-finite values. Request/response schemas and the embedded adapter
were bumped, and runtime-reduction evidence schema 4 binds each receipt to the
new protocol digest.

An independent regression checks the exact payload bytes for `-0`, the minimum
positive F32 subnormal, `1` and `-2.5`:

```text
00000080 01000000 0000803f 000020c0
```

It also rejects an old `{shape, values}` decimal envelope, a wrong byte length
and non-canonical Base64.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 56:

- artifact bytes: `21,409,396,286`;
- artifact SHA-256:
  `ac0a61da0d569c70501f663517ed81f53d0e22dac9f9870fb3be9658474d86bd`;
- authoritative-execution schema: `12`;
- execution-protocol schema: `2`;
- embedded-adapter schema / SHA-256:
  `2` / `c11c6b2d0670c74d999fcf7a19da3f048bd60dfc6e7c742fde38dada70af2d50`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- structural integrity sections: `24`;
- integrity root:
  `e34b016eb5fb0f30048dbbde9f7d9bcc49a74efd8b127936e663850f855f33b0`; and
- fidelity gate: 100 unresolved native reductions,
  `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The 1,012-byte source audit has SHA-256
`93f7085fb27b231fa553e13aa0bb4322439cd50e23921328d5d33c22662c3c90`.

## Fresh authoritative captures

Fresh Transformers 5.5.0 / Torch 2.12.1 CPU eager `torch.inference_mode`
captures were produced from the immutable source package:

| modality | trace bytes | trace SHA-256 |
| --- | ---: | --- |
| image | 8,232,797 | `48a7270be75d44ae31809ae04c77dd419b22937bd3ec5fdc1e4e705f031af68b` |
| video | 8,232,807 | `10618977609781f65a7c325c7cbff3018cfb9b2ad280cd1a9bdde898bdaa7e9d` |
| audio | 8,102,175 | `d07227accbccdd911f2e5d66b0152e12f8d908ad480503e45b55d8344fac0610` |

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved to one private
`mktemp` directory under an `EXIT`/`INT`/`TERM` restoration trap. While both
declared paths were absent, the streaming reader rehashed all 2,130 payloads
and reported `sourceCheckpointAccessed=false` with the same
15,992,314,836-byte storage digest.

The same inspection rendered `layer_0_q_proj[0,0,0]` with its operation-declared
2,560-term ARM BF16 reduction and substituted the first four learned operands:

```text
bits 0xbbcb -> F32 0xbbcb0000 -> -0.006195068359375
bits 0x3af2 -> F32 0x3af20000 ->  0.0018463134765625
bits 0xbd4c -> F32 0xbd4c0000 -> -0.0498046875
bits 0x3d08 -> F32 0x3d080000 ->  0.033203125
```

The 43,267,783-byte source-removed inspection has SHA-256
`1202abef22910caf71096b6cb576cac7164fb42ba55411f243b30a6c6c1c4fef`.

In the same source-hidden window, the real artifact matched the fresh
authoritative captures with zero absolute and relative tolerance:

| modality | prefill | generation | token / position | caches | receipts |
| --- | --- | --- | --- | ---: | ---: |
| image | 5/5 pass | lossless-within-dtype | `184 / 2` | 24/24 | 32 |
| video | 5/5 pass | lossless-within-dtype | `184 / 2` | 24/24 | 32 |
| audio | 5/5 pass | lossless-within-dtype | `184 / 2` | 24/24 | 36 |

The aggregate records 3/3 zero-tolerance prefills, 3/3 zero-tolerance greedy
generations, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. All 100 receipts use evidence schema 4 and
carry protocol SHA-256
`8ed1a2951950e291d6d2186eaed952c2801daf208bd50446eaafde7240ccaa16`.
The 1,086,703-byte report has SHA-256
`0e79c42aa971ff7030fdbb3f5926b30aec75383d7d7608bf2e1425dd5ecf38ce`.
The trap restored both product paths and removed the private directory.

## Preserved fidelity boundary

Schema 56 closes the exact tensor transport to and from the artifact-owned
native provider. It does not publish Apple Accelerate SGEMM's scalar product
rounding, lane, tile or fold schedule. The 100 BMM scalar reductions therefore
remain fail-closed, and `exactReplayClaim=forbidden` remains the only valid
claim.

## Validation

```text
npm run typecheck
npm test
git diff --check
```

The focused transport/provider tests, composite writer/reader tests, real
artifact regeneration, complete source-to-artifact byte comparison,
source-removed payload verification, scalar substitution and zero-tolerance
multimodal suite all exited zero. The final repository run reported 259 tests:
256 passed, 0 failed and 3 explicitly optional non-Gemma integrations skipped.
