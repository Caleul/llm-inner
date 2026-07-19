# Gemma 4 E4B runtime-reduction product navigation

This is candidate loop-91 evidence for a direct navigation improvement to the
real dense Gemma 4 literal artifact. It does not resolve Apple Accelerate's
unpublished BMM reduction schedule, does not make those operations executable,
and does not authorize `.agent-loop/checkpoints/gemma4-dense-lossless/`.

## Independent starting review

The loop started from clean commit
`d8548ca78db1ecbc7d54788b8c1d26b9217e06e0` and inspected the schema-33
state-chain implementation and `HANDOFF-0090-20260719T040434Z`. The focused
Gemma suite passed after the new implementation with 32/32 tests. This accepts
the state-chain implementation as current repository evidence, but not the
dense-lossless checkpoint: the same 100 BMM assignments remain non-executable.

## Why no reduction schedule was assigned

The existing class-wide evidence had already rejected ascending, FMA,
interleaved, blocked and pairwise candidates. Loop 91 checked the remaining
general exact-dot direction against every audio layer before editing. A
correctly rounded mathematical dot still differed from the authoritative
Apple SGEMM output in 77/27,648 content-score coordinates (maximum absolute
error `0.000030517578125`) and 925/14,976 position-score coordinates (maximum
absolute error `0.0000457763671875`). Audio value dots happened to match
12,288/12,288, but that does not establish one schedule for all five classes.
No operation, shape or layer was promoted.

## Implemented boundary

The source-removed reader now exposes two fail-closed surfaces derived only
from the artifact's serialized calculation graph and authoritative execution
contract:

- `--list-runtime-reductions` lists every compatible operation with its
  operation class, output domain, executable reduction-domain binding and
  provider;
- `--runtime-reduction-audit` binds one output coordinate and renders a finite
  operand-product window. It resolves head/chunk coordinates, exact tensor
  addresses and padding predicates for vision score/value and audio
  content-score/position-score/value BMMs.

The audit deliberately has no output assignment. Its status is
`fail-closed-runtime-reduction`; `productRounding` and `accumulationOrder` are
both `unpublished-provider-boundary`, and the terminal text says that the Apple
SGEMM reduction is unavailable. The existing strict scalar view still rejects
the operation. Dynamic vision value reductions require an explicit
`--input-start/--input-count` window rather than guessing the caller's patch
extent.

## Real source-removed evidence

The real artifact remained unchanged:

- model: `google/gemma-4-E4B`;
- revision: `411aa17b749aa952df1359d2dcea73917a544d9a`;
- format: dense BF16/F32 Safetensors;
- artifact: `artifacts/gemma4-e4b-dense.literal.json`;
- schema: 33;
- bytes: `21,387,315,487`;
- prior recorded artifact SHA-256 from loop 90 (the 21 GB artifact was not
  regenerated or rehashed in this loop):
  `15eb8d4d8321bf2c42273ceb425392c52d7bb264632bc5390b1df435e2c144aa`.

The checkpoint directory was physically renamed for the commands below and
restored by an exit trap. Every output recorded
`sourceCheckpointAccessed=false`.

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-runtime-reductions \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop91-runtime-reductions.json
```

The catalog contained exactly 100 operations:

| operation class | assignments |
| --- | ---: |
| `vision-attention-score` | 32 |
| `vision-attention-value` | 32 |
| `audio-content-attention-score` | 12 |
| `audio-position-attention-score` | 12 |
| `audio-attention-value` | 12 |

Its SHA-256 is
`97de0b29149fbe86da26a8459121c74e0383e2da65090559c87f41b962f27c5e`.

One real coordinate from every class produced the following audits:

| class | rendered terms | complete | report SHA-256 |
| --- | ---: | --- | --- |
| vision score | 64 | yes | `36c68e4a91cce0bfc4de0b3394da05e651387e3df16fe305ed100f86731acfcc` |
| vision value | 9 | no, explicit dynamic window | `25552682cb63002991aeeb4eea0ba0b08bcb654a2be62eb0b832017d8a9dd904` |
| audio content score | 128 | yes | `3f7b065462e749cae08aa5c7efebc00d44d99a74f6d16878c15cff1f867462eb` |
| audio position score | 128 | yes | `53ce16c4ca9a5f22c6ffb3210b9b6e3062796b6da475221c596d92d9b83561bd` |
| audio value | 24 | yes | `06370fe20458acddf70c45063d910ae68d8fdef950ea578626f817258968c851` |

All five declared `provider=Apple Accelerate SGEMM` and
`status=fail-closed-runtime-reduction`. The complete audits enumerate the full
static head/context domain; the vision-value audit is intentionally partial
because patch count is a runtime input dimension.

## Preserved limitation

This change makes the known side of each native BMM addressable and auditable;
it does not make the unknown side smaller. Exactly 100 reductions still have no
authoritative scalar rounding tree. Therefore full literal replay remains
fail-closed, the modality suite still requires its diagnostic acknowledgement,
and no Gemma 4 checkpoint marker was created.
