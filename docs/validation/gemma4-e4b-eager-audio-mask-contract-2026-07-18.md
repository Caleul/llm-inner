# Gemma 4 E4B eager audio mask contract

This is candidate schema-v17 evidence from loop 73. It does not accept the
dense-lossless checkpoint: the loop which implemented the contract cannot
independently certify it, and 100 native vision/audio BMM assignments retain
their explicit unpublished, fail-closed scalar reduction schedule.

## Independent starting review

Loop 73 started from commit `f1d7f281bcc249b700ccf51a15430874b26dd70f`
and independently inspected loop 72's changes and evidence. Before this loop's
implementation, `npm run typecheck` and `npm test` passed with 231/231 tests.
That review accepted the schema-v16 runtime binding as repository evidence but
did not accept the Gemma 4 checkpoint.

## Root cause and class-wide contract

The loop-72 direct and composite audio traces used different attention
implementations. Loading `Gemma4ForConditionalGeneration` without an explicit
choice selected SDPA; the composite capture selected eager. All 2,131 loaded
state tensors and both audio inputs were bitwise equal. Repeated execution in
one eager-loaded process produced bitwise-equal direct and composite terminal
audio features, and three independent eager composite captures were stable.

Transformers 5.5.0 eager source constructs a four-dimensional **additive**
bidirectional/local mask: allowed entries are zero and rejected entries carry
the most-negative finite value. `Gemma4AudioModel` pads that numeric mask with
zero while gathering it into blocked five-dimensional form. Finally,
`Gemma4AudioAttention.forward` applies
`masked_fill(attention_mask.logical_not(), -1e9)`. The resulting pinned
semantics are therefore intentionally inverted relative to a conventional
boolean mask: zero allowed entries and zero block padding are filled, while
nonzero rejected source entries retain their pre-mask score.

Schema v17 records this operation-class contract as
`transformers-eager-additive-mask-logical-not-v1`. The executor applies it to
every compatible audio attention assignment by source position, shape and
mask value; there is no layer allowlist. Scalar views expose the additive-mask
construction, block padding, `logical_not` and fill rule. Artifact, capture and
trace boundaries additionally require both `torch.inference_mode` and
`attentionImplementation=eager`; blank, SDPA and no-grad trace contexts fail
closed.

For the one-frame authoritative input, every blocked mask entry is zero, so
the eager source replaces every score by `-1e9`. Softmax is uniform and the
remaining native BMM score differences no longer reach the terminal audio
feature. This causal chain explains the exact composite result below; it does
not invent a scalar reduction schedule for Apple Accelerate SGEMM.

## Real schema-v17 artifact

The artifact was regenerated from immutable `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`:

- path: `artifacts/gemma4-e4b-dense.literal.json`;
- bytes: `21,384,488,626`;
- SHA-256: `508bd8f26bf4a9d3e9dc7b363bab68e9a5d234c9eb285f6cb1841a70e003eb46`;
- constants/storage decoders: `2,130` / `2,130`;
- embedded learned payload bytes: `15,992,314,836`;
- calculation graph: 2,709 assignments and 3,363 explicit predecessor edges;
- literal-storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

The source checkpoint was physically moved out of its declared path for all
source-removed commands below, then restored. The payload reader verified all
2,130 constants and all `15,992,314,836` embedded bytes without source access.
Its report is
`/private/tmp/llm-inner-loop73-artifact-v17-source-removed.json`, SHA-256
`05946e95b8e7c98ff55b524842529219fdf20777f407acedc9a4aa7e8e6dfb96`,
and records `sourceCheckpointAccessed=false`.

## Source-removed differential evidence

The eager direct-audio trace is
`/private/tmp/llm-inner-loop73-audio-eager.json`, SHA-256
`19901d9809c14e7951948c1fac54a69715916615d6554bfd4bae781e7ef92b21`.
Its zero-tolerance source-removed comparison checked 444 operations: 385
passed and 59 native-BMM/pre-mask derivatives diverged, beginning at
`audio_layer_0_attention_position_scores` with maximum absolute error
`0.000030517578125`. Terminal `audio_features [1,2560]`, every
source-anchored score stage and every attention-context cast were exact. The
report is `/private/tmp/llm-inner-loop73-audio-eager-source-removed-v17.json`,
SHA-256
`e2af49d133263fc70c2a7da6c262eafcaf3f14f4f033d0623d57c9645623bbb6`.

The eager full-composite trace is
`/private/tmp/llm-inner-loop73-composite-audio-eager.json`, SHA-256
`d07227accbccdd911f2e5d66b0152e12f8d908ad480503e45b55d8344fac0610`.
With source absent and zero tolerance, all five prefill checkpoints were exact:
text embedding, terminal audio feature, audio scatter, PLE combine and logits.
One greedy decode produced token 184 at position 2 exactly; its full
`[1,2,262144]` selection logits, all step and terminal KV caches across 24
layers, and terminal logits had zero absolute and relative error. Prefill and
generation were both `lossless-within-dtype`, with no first divergence. The
report is
`/private/tmp/llm-inner-loop73-composite-audio-source-removed-v17.json`,
SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`,
and records `sourceCheckpointAccessed=false`.

`--allow-unverified-fidelity` remains mandatory for this comparison because
the artifact correctly retains the 100 unpublished native-BMM scalar schedules
as fail-closed. Thus the exact composite result is candidate evidence for this
specific input and one generation step, not a checkpoint declaration or proof
of an exact addressable scalar schedule for every native BMM reduction.

Final repository validation passed `npm run typecheck` and `npm test` with 233
tests and zero failures. No
`.agent-loop/checkpoints/gemma4-dense-lossless/` marker was created.
