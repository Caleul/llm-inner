# Gemma 4 E4B: closed scalar statement environments — 2026-07-19

## Completed boundary

Schema v39 stored every forward scalar statement as a closed syntax tree, but
identifier resolution was still implicit. Schema v40 embeds one
`statementEnvironment` per scalar calculation and validates it generically
from syntax plus the adjacent calculation contracts.

The environment binds:

- output-coordinate symbols and positional `orderedInputs`;
- every local write to its producer statement ordinal;
- reduction indices and human extent aliases to the serialized executable
  reduction domain;
- learned roles accepted by `decode(role)`;
- the finite intrinsic registry and `Infinity`; and
- tensor `.shape`, reduction-stage `.schedule`, and registered structured
  result members.

Unknown identifiers, callees, learned roles, named reduction arguments,
stages, members, read-before-produce locals and malformed ranges fail closed.
The same resolver rebuilds environments after image, video and audio call-site
binding; it has no operation, assignment, layer, shape or dtype dispatch.

This full-class audit exposed two real aliases which syntax parsing alone had
accepted. Audio `per-dim-softplus-scale` used free `feature` instead of the
actual final output axis. Proportional text RoPE used free `head_dim` instead
of its declared `rotaryDim`. Both formulas now embed the source-bound axis or
integer directly.

## Real artifact and source-present storage audit

The artifact was regenerated from dense unquantized `google/gemma-4-E4B` at
immutable revision `411aa17b749aa952df1359d2dcea73917a544d9a`:

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Result:

- artifact schema 40, formula-language schema 25, calculation-graph schema 8
  and scalar-calculation schema 6;
- 21,409,225,548 bytes, SHA-256
  `f0e8a36902b1bbb65c1f85ca32347b5cdfc05ae694a6a74af12cd3b8179543b7`;
- 2,130 constants and 15,992,314,836 embedded learned bytes; and
- no checkpoint path in the artifact.

The source-present audit compared every constant by name, dtype, shape and
bytes. Source and literal storage both produced SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The 1,012-byte report has SHA-256
`94eb047f91c7418e23227f863192ab318d88f81b0b5745a0f353d6c231237b52`.

## Source-removed environment and scalar audit

`./gemma-4-E4B-dense` was physically moved and restored under an EXIT/INT/TERM
trap. While the declared source path was absent, the reader verified all
embedded payload commitments and opened the schema-40 semantic tail:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-v40-source-removed-environments.json
```

The report recorded `sourceCheckpointAccessed=false` and:

| contract | count |
| --- | ---: |
| instantiated assignments / environments | 2,708 |
| statement programs | 4,144 |
| ordered-input bindings | 3,491 |
| local bindings | 1,431 |
| reduction bindings | 1,722 |
| intrinsic bindings | 10,727 |
| member-access bindings | 211 |
| expression nodes | 118,780 |

The 42,960,492-byte report has SHA-256
`a62014f06712eb219f94b8930967d2b2b275605e1afa882792c763adc369c4fd`.

A source-removed four-term window for `layer_0_q_proj[0,0,0]` decoded BF16
words `0xbbcb`, `0x3af2`, `0xbd4c`, `0x3d08` into exact literals
`-0.006195068359375`, `0.0018463134765625`, `-0.0498046875`, and
`0.033203125`. The report contains no `weight[...]` placeholder and has
SHA-256 `1c81364ce7ca2595c464fea44705adfa6ce6084cb43b2f9df79aad77e5b2bfb9`.

## Authoritative differential and preserved boundary

With the checkpoint path again physically absent, the registered image, video
and audio traces were compared at zero absolute and relative tolerance against
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`.
All three prefills and all three one-token greedy generations passed, every
generation selected token `184` at position `2`, all 24 KV caches per modality
matched, and `firstDivergence=null`.

The 169,857-byte report has SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.
It preserves exactly 100 `runtimeDefined` Apple Accelerate BMM reductions, so
the fidelity claim remains candidate modality coverage rather than full
literal scalar replay of those unpublished native reduction trees.

Fresh repository validation after implementation and regeneration:

```text
npm run typecheck  # exit 0
npm test           # 253/253, exit 0
git diff --check   # exit 0
```
