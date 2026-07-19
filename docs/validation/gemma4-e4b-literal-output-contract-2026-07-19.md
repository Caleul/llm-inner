# Gemma 4 E4B literal output/state contract

## Boundary

The real dense `google/gemma-4-E4B` artifact previously serialized complete
inputs, forward routing and greedy control, but its public result shapes, KV
ownership and terminal-state alignment were still checked through executor
conventions. Schema v32 adds one canonical `outputContract` derived from the
registered program and executed by both synchronous and paged literal replay.

The contract covers the complete compatible output class:

- `composite_llm_input_ids` as the exact I32 modal-to-PAD substitution;
- public F32 row-major `hidden_states_0`, `ple_inputs` and logits tensors with
  declared producers, axes and shapes;
- producer-only post-RoPE BHSD key/value caches, including exact
  `past_sequence + input_sequence` growth;
- greedy token, selection-logit and cache-snapshot cardinality;
- early termination only after a terminal EOS token; and
- terminal logits/cache selected from prefill at zero steps or the last
  completed incremental forward otherwise.

The artifact validator rejects altered shape expressions, producers, cache
ownership or terminal relations. Runtime tests also reject a missing producer
cache and a missing greedy cache snapshot. Formula schema v18 names
`/outputContract` as the normative output authority, and the source-removed
end-to-end view includes the same object.

## Real artifact generation and storage audit

```bash
npm run build
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1

node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /private/tmp/llm-inner-loop89-v32-source-audit.json \
  --verify-gemma4-payloads
```

Observed result:

- artifact bytes: `21,387,313,726`;
- artifact SHA-256:
  `afaacded56834426f38c33a52f40abae52a1660ed6a2514a981d03d679f9bf81`;
- schema: `32`; formula schema: `18`; output-contract schema: `1`;
- constants: `2,130`;
- embedded payload bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- audit report SHA-256:
  `002ea47f134dbc4ea8031733f75db1bc29450f381f5289036a23cde47116a7a8`.

## Source-removed navigation and replay

The directory `./gemma-4-E4B-dense` was moved aside before both commands and
restored only by the shell exit trap after they completed:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --end-to-end-calculation --generation-max-new-tokens 2 \
  --output /private/tmp/llm-inner-loop89-v32-source-removed-end-to-end.json

node --max-old-space-size=4096 \
  dist/src/gemma4-literal-composite-modality-suite-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --image-trace /private/tmp/llm-inner-loop85-composite-image-eager.json \
  --video-trace /private/tmp/llm-inner-loop85-composite-video-eager.json \
  --audio-trace /private/tmp/llm-inner-loop73-composite-audio-eager.json \
  --report /private/tmp/llm-inner-loop89-v32-composite-modality-suite.json \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --top-k 10 --allow-unverified-fidelity
```

The 65.4 MB end-to-end report exposed the output contract with `2,709` forward
operations, `20` greedy operations and all `2,130` constants. Its SHA-256 is
`a71b9b5ae0baea9b106c78011f1a3ab5c4a89cbf231983b4f65db1b7b2ad3743`.

Image, video and audio each passed prefill and one-token greedy generation at
zero absolute/relative tolerance, including terminal logits and all 24
producer caches. The aggregate report retained the exact native BMM split
`32 + 32 + 36 = 100` and has SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

## Fidelity boundary

This is candidate evidence for the output/state contract, not checkpoint
certification. Exactly 100 Apple Accelerate BMM assignments remain
`fail-closed-runtime-reduction` because their addressable scalar accumulation
schedule is not established authoritatively. The schema-v32 contract validates
their produced tensor/cache boundaries but does not invent that missing scalar
schedule.
