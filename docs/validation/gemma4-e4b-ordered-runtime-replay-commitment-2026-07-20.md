# Gemma 4 E4B: ordered runtime-replay commitment — 2026-07-20

## Completed boundary

Schema 57 committed every native-BMM request and response separately, but its
aggregate replay evidence checked only receipt count and unique operation IDs.
A report with the right count could therefore omit one declared BMM and replace
it with another unique receipt, or reorder valid receipts, without invalidating
an aggregate commitment.

Schema 58 adds an artifact-owned `executionProtocol.replayCommitment`:

- execution order is `provider-append-order`;
- every complete schema-5 receipt is serialized with ECMAScript
  `JSON.stringify`;
- each serialized receipt is followed by one `LF`, including the final one;
- the replay records the ordered operation IDs, transcript byte count and
  SHA-256; and
- composite comparison derives the exact active operation sequence from the
  embedded vision or audio program and requires positional equality.

Validation rebuilds the aggregate evidence from its receipts and rejects a
changed digest, receipt reorder, duplicate/empty operation ID, receipt bound to
a different execution-protocol digest, or any mismatch with the artifact's
declared operation order. This is operation-level and works for all registered
layers and five native-BMM classes; it does not contain layer-specific lists.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 58:

- artifact bytes: `21,409,397,092`;
- artifact SHA-256:
  `325edd065364774c9277cc76f80490730b27e96620a8065cf82d5fbfd7b36eca`;
- authoritative-execution schema: `13`;
- execution-protocol schema: `4`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes; and
- structural integrity root:
  `a355a7f4e367d0ea46c6815b73da7b1dfc11dd6524416b3adff08a42af3cd8d6`.

The 1,012-byte source audit has SHA-256
`830d7eae02f83bd6b0243ed423b30f921ab538c7c247879a78d8db8b05660ea8`.

## Checkpoint and checkout adapter physically unavailable

`./gemma-4-E4B-dense` and
`helpers/torch_gemma4_runtime_reductions.py` were moved into one private
`mktemp` directory under an `EXIT`/`INT`/`TERM` restoration trap. While both
paths were absent, the literal reader rehashed all 2,130 embedded payloads and
reported `sourceCheckpointAccessed=false`. It rendered
`layer_0_q_proj[0,0,0]` and substituted the first four addressed BF16 learned
values as exact decoded literals:

```text
-0.006195068359375
 0.0018463134765625
-0.0498046875
 0.033203125
```

The 43,269,036-byte inspection has SHA-256
`36bebad820e9fb0b44a38ae6dfb5a2d01dd08e7662a5d7b4fa07306d4abc092f`.
The trap restored both paths and removed its private directory.

## Source-removed authoritative differential

In a separate source-hidden window, the schema-58 artifact and pinned
`venv/bin/python` runtime replayed the fresh schema-57 Transformers 5.5.0 /
Torch 2.12.1 eager captures at zero absolute and relative tolerance:

| modality | prefill | generation | ordered receipts | receipt transcript SHA-256 |
| --- | --- | --- | ---: | --- |
| image | lossless-within-dtype | lossless-within-dtype | 32 | `adbbf29e...a30440` |
| video | lossless-within-dtype | lossless-within-dtype | 32 | `adbbf29e...a30440` |
| audio | lossless-within-dtype | lossless-within-dtype | 36 | `8743338e...503a8c` |

For image and video the exact sequence starts at
`vision_layer_0_attention_scores` and ends at
`vision_layer_15_attention`. Audio starts at
`audio_layer_0_attention_content_scores` and ends at
`audio_layer_11_attention`. Declared and executed IDs matched positionally for
all 100 receipts. The aggregate reports 3/3 zero-tolerance prefills, 3/3
zero-tolerance greedy generations, `firstDivergence=null` and
`sourceCheckpointAccessed=false`.

The schema-2, 1,142,824-byte suite report has SHA-256
`22cbb9e0f4762607327ca89d54e60d1b6854e00ee2aea55d7c3108fdbbae91d4`.
Both checkpoint and checkout adapter were restored after the run.

## Preserved fidelity boundary

The ordered commitment proves which complete receipts executed and in which
artifact-declared order. It does not publish Apple Accelerate SGEMM's scalar
product rounding, lane, tile or fold schedule. The same 100 scalar reductions
remain `fail-closed-runtime-reduction`; the integrity-bound gate remains
`blocked-on-runtime-reduction` with `exactReplayClaim=forbidden`.

## Validation

```text
npm run typecheck
npm test
git diff --check
```

Focused tests cover aggregate reconstruction, reordering, expected-order drift
and protocol-digest drift. Real artifact regeneration, complete source-to-
artifact payload comparison, source-removed payload verification, scalar
literal substitution and the zero-tolerance three-modality suite all exited
zero. The final repository run reported 261 tests: 258 passed, 0 failed and 3
explicitly optional non-Gemma integrations skipped.
