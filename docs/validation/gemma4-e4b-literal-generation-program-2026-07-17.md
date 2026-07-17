# Gemma 4 E4B literal generation program — 2026-07-17

This is candidate evidence that greedy generation is now declared inside the
real dense literal artifact. It does not accept the dense-lossless checkpoint:
authoritative BF16 forward, KV-cache and greedy-token equivalence remain
approximate.

## Closed artifact boundary

Schema v2 adds required-in-generation `max_new_tokens` and optional
`eos_token_id` to the declared
inputs and serializes twelve ordered generation assignments. They bind prefill,
last-row logits, lowest-ID argmax tie-breaking, token append, position advance,
incremental inputs, post-RoPE cache carry, the repeated declared Gemma 4
forward calculation, cache snapshots, EOS timing and terminal logits/cache
selection from the same final forward state. The incremental input
assignment explicitly removes `attention_mask`, `mm_token_type_ids` and every
image/video/audio input after prefill.

Both object and streaming readers derive the canonical generation program from
the embedded forward graph. A missing or changed input, assignment, state
transition, semantic rule or output fails closed. No generic decoder or source
checkpoint invocation appears in the generation program.

## Real source-removed evidence

The dense artifact was regenerated from `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, then the source directory was
renamed away under a shell exit trap while the reader exposed the generation
program and verified every embedded payload commitment:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --show-generation-program \
  --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop42-generation-program.json
```

Observed result:

- artifact SHA-256:
  `bca3508a12de591e6cf0761f4c8baf77ff340c16e8b9ce63bdb065723a056510`;
- artifact bytes: `21,326,381,648`;
- 2,130 constants and 2,130 storage decoders;
- all `15,992,314,836` embedded payload bytes verified against their internal
  commitments with sorted-storage SHA-256
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- the independent source-to-literal audit compared those same 2,130 tensors
  and produced that digest on both the Safetensors and literal sides;
- first declared forward assignment `composite_placeholder_masks`, final
  assignment `final_logit_softcap`;
- twelve generation assignment IDs from `generation_prefill` through
  `generation_terminal_cache` in canonical order;
- declared outputs `generated_token_ids`, `selection_logits`,
  `step_past_key_values`, `terminal_logits` and `terminal_past_key_values`;
- `sourceCheckpointAccessed: false`;
- report bytes: `8,807`; report SHA-256
  `687d984dfd4fafc90a2037f194e2fe5665d77915289e3a1badabc599c46173a0`.

The trap restored the source directory. This evidence establishes the
self-contained generation state-machine declaration and storage integrity. It
does not establish that the current paged BF16 executor reproduces the
authoritative runtime's logits, caches or selected tokens exactly.
