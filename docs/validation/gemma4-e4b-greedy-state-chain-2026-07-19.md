# Gemma 4 E4B literal greedy state chain

## Boundary

Loop 90 independently reran `npm run typecheck` and all 235 tests against
schema v32 before changing the format. That accepted the prior executable
input/output boundary as repository evidence, but not the dense-lossless
checkpoint: 100 Apple Accelerate BMM reductions remain fail-closed.

Schema v33 closes a distinct artifact gap across the complete greedy operation
class. The serialized program previously retained selection logits and the
post-forward cache for every step, but not the post-forward logits paired with
that cache. Consequently the output contract could not independently reject a
result assembled from logits and cache belonging to different forward states.

The artifact now owns and executes these relationships for every step:

- `generation_logits_append` stores the complete F32
  `forward_state[step+1].logits` as `step_forward_logits[step]`;
- `step_forward_logits[step]` and `step_past_key_values[step]` are produced by
  the same incremental forward;
- `selection_logits[0]` is bitwise identical to prefill logits;
- `selection_logits[step]` for `step > 0` is bitwise identical to
  `step_forward_logits[step-1]`;
- `generated_token_ids[step]` equals the declared ascending-scan argmax of its
  selection logits, including lowest-ID tie behavior; and
- terminal logits and terminal cache are both identical to the final
  incremental snapshots, or both are the prefill state for zero steps.

The synchronous and paged interpreters retain these snapshots, the structured
generation control is schema v2, two-step navigation materializes 22 ordered
control assignments, and malformed token, intermediate-logit and terminal-logit
chains fail closed in tests.

## Real artifact and payload identity

Generation:

```bash
npm run build
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Observed schema-v33 artifact:

- bytes: `21,387,315,487`;
- SHA-256:
  `15eb8d4d8321bf2c42273ceb425392c52d7bb264632bc5390b1df435e2c144aa`;
- output-contract schema: `2`;
- generation-control schema: `2`;
- constants/storage decoders: `2,130` / `2,130`;
- embedded learned payload bytes: `15,992,314,836`;
- ordered forward assignments: `2,709`;
- explicit predecessor edges: `3,363`;
- reduction domains: `1,722`, including exactly `100` runtime-defined.

Source-present audit:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop90-v33-source-audit.json \
  --verify-gemma4-payloads
```

All `15,992,314,836` payload bytes matched. Source and literal storage SHA-256
were both
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The audit report SHA-256 is
`d0442dd1440a9d2bc18233bc510c17916997cc43520b6bd607dda776c7b42c3c`.

## Source-removed navigation and differential replay

The source directory was moved to a fixed temporary path under an EXIT trap
before both commands and restored only after they completed:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop90-v33-source-removed-end-to-end.json

node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-modality-suite-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --image-trace /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --video-trace /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --audio-trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop90-v33-composite-modality-suite.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --top-k 10 --allow-unverified-fidelity
```

The end-to-end report exposes `generation_logits_append[0]` and `[1]`, all 22
ordered generation assignments, complete storage coverage (`2,076` reachable
plus `54` declared runtime-unreachable shared-KV locals), and the executable
output contract. Its SHA-256 is
`d2a4ad14b4ee90fd651efae22aa0d4b9b1b2eeaed25fa1884451945b5ac89081`.

A separate source-hidden `--verify-payloads` pass recomputed all 2,130
commitments and all `15,992,314,836` embedded bytes without opening the
checkpoint. It reproduced literal storage SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`
and wrote report SHA-256
`c4a9f18cd69000d2561ad204a045223d754d202d2f1cc416e0aa3d0481f27e9a`.

The source-hidden modality suite passed image, video and audio prefill and
one-token greedy generation at zero absolute and relative tolerance. Each
modality matched token `184` at position `2`, terminal logits and all 24
producer caches; `firstDivergence` is null. The report retained the exact BMM
split `32 + 32 + 36 = 100` and has SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

## Fidelity boundary

This is candidate schema-v33 artifact evidence, not checkpoint certification.
The 100 native vision/audio BMM assignments still declare
`fail-closed-runtime-reduction` because Apple Accelerate SGEMM's addressable
scalar accumulation schedule is unpublished. The state-chain contract proves
which produced logits, tokens and caches belong together; it does not invent
the unresolved reduction tree or authorize
`.agent-loop/checkpoints/gemma4-dense-lossless/`.
