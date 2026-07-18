# Gemma 4 E4B dense address and IEEE decoder program

This is candidate artifact-format evidence from loop 68. It does not accept
the dense-lossless checkpoint: 100 instantiated native batched-matmul
operations remain explicitly fail-closed and the accepted audio differential
remains approximate.

## Boundary closed

The real `google/gemma-4-E4B` artifact now uses schema v12. Every one of its
2,130 dense storage decoders embeds two executable contracts:

- `address`: logical shape, canonical row-major element strides, the closed
  domain of every index, element byte width, safe-integer arithmetic and exact
  element/byte-offset equations;
- `decode`: the little-endian read and bit-level IEEE conversion. The selected
  real package contains BF16 only, so all 2,130 real decoders use
  `ieee-bfloat16-expand` with `resultBits = sourceBits << 16`. Unit fixtures
  additionally execute F32 bitcast and the complete F16 zero/subnormal/normal/
  infinity-or-NaN program.

The scalar substitution reader no longer recomputes row-major offsets or
selects conversion code from the constant dtype. It evaluates the serialized
`address`, reads exactly that byte range and executes the serialized `decode`.
Calculation slices and the end-to-end view carry each complete decoder rather
than only its ID and operation label.

## Real artifact and source-present proof

Command:

```bash
npm run build && node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Result:

- schema: 12;
- artifact bytes: `21,383,604,505`;
- artifact SHA-256:
  `a34ded7ed837b7fdff36a75bcc9c93289ec6d36b9f52cad8985432b27d030ec1`.

The source-present audit command was:

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop68-source-audit.json \
  --verify-gemma4-payloads
```

It recomputed six immutable package commitments totaling `16,024,773,810`
bytes. All 2,130 literal payloads totaling `15,992,314,836` bytes matched the
source, with both concatenated storage SHA-256 values equal to
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

## Source-removed proof

The source directory was physically renamed under an EXIT trap. With
`./gemma-4-E4B-dense` absent, the reader ran embedded-payload verification,
the complete end-to-end calculation view for two greedy tokens and five
representative learned scalar views spanning text linear, clipped vision
linear and scalar bounds, audio convolution, 2-D position lookup and audio
per-dimension scale.

The end-to-end report contains:

- `sourceCheckpointAccessed: false`;
- 2,709 forward assignments;
- 20 instantiated greedy assignments and 42 cache transitions;
- 2,076 reachable learned constants plus 54 explicitly runtime-unreachable
  shared-KV locals, for complete coverage of all 2,130 constants;
- payload verification of all `15,992,314,836` embedded bytes;
- 100 preserved `fail-closed-runtime-reduction` instances.

Its 58 MiB report SHA-256 is
`4d54e78f30ae2699675cf808b379724abf8c16d675e88cd5f3d447810f70d5a6`.

A bounded exhaustive decoder pass then opened the artifact with the checkpoint
still absent. For every decoder it evaluated, read and decoded the first and
last logical coordinate. All 4,260 endpoint reads were finite, every final
address ended exactly at its payload boundary, and all 4,260 decoded F32 bit
patterns equaled `sourceBits << 16` across 2,130
`ieee-bfloat16-expand` programs. Report SHA-256:
`5905e71214733c2f90a0d460b94c907c2b13db3089d8590ce15c350f9febdaa2`.

Representative literal substitutions also expose the concrete byte address:

| calculation | logical indices | byte offset | BF16 bits | F32 bits | decoded F32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `layer_0_q_proj`, first weight | `[0,0]` | 0 | `0xbbcb` | `0xbbcb0000` | `-0.006195068359375` |
| vision layer-0 Q input minimum | `[]` | 0 | `0xc0cc` | `0xc0cc0000` | `-6.375` |
| audio subsample kernel term 8 | `[0,0,2,2]` | 16 | `0xbdac` | `0xbdac0000` | `-0.083984375` |
| vision Y position scalar | `[1,0,2]` | 15,728,644 | `0x277f` | `0x277f0000` | `3.5388358909926865e-15` |
| audio layer-0 per-dimension scale | `[0]` | 0 | `0xc004` | `0xc0040000` | `-2.0625` |

## Validation and limits

`npm run typecheck` passed. `npm test` passes 228 tests after adding corruption,
rank-zero scalar, F32/F16/BF16 decode and slice/end-to-end decoder-ownership
coverage.

This change does not assign an invented schedule to Apple Accelerate. Vision
score/value BMM and audio content/position/value BMM remain
`runtime-defined`; the end-to-end artifact correctly remains fail-closed for
those 100 instantiated operations. No
`.agent-loop/checkpoints/gemma4-dense-lossless/` marker was created.
