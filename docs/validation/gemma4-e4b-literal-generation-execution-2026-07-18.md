# Gemma 4 E4B literal generation-program execution — 2026-07-18

This is candidate evidence that the real dense artifact's serialized greedy
generation program is executable and produces a navigable value for every
declared transition. It does not accept the dense-lossless checkpoint: the
underlying paged BF16 text calculation remains numerically approximate and the
multimodal towers are not executed by this command.

## Closed execution boundary

`src/gemma4-literal-generation.ts` is a state-machine interpreter for the
twelve assignments stored in artifact schema v2. It validates the serialized
program against the embedded forward graph, then executes those assignments in
artifact order. Each concrete execution records the assignment ID, operation,
step, instantiated output name and produced value. For example,
`forward_state[step+1]` becomes `forward_state[1]`, and
`generated_token_ids[0..step]` becomes `generated_token_ids[0..0]` on the
first decode step.

Forward calculation enters through a narrow injected port. The paged literal
path supplies the source-independent text executor for prefill and incremental
forward, while the interpreter itself owns last-row logits capture, finite
lowest-ID argmax, token append, integer position advance, post-RoPE cache
carry, incremental input construction, EOS timing, cache snapshots and
terminal logits/cache selection. The former second greedy loop was removed.
Generation now also rejects caller masks and initial `pastKeyValues`, because
neither is an input of the serialized generation program.

The CLI report summarizes every assignment value without serializing whole
logit/cache tensors again: scalars and token lists remain literal, tensors get
shape plus F32 byte hash, and cache-bearing values record producer layers and
shapes.

## Real source-removed execution

The source directory was renamed under an exit trap for the entire candidate
process:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-loop43-unavailable
node dist/src/gemma4-paged-text-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --input-ids 2 --max-new-tokens 1 --max-read-mib 16 \
  --allow-unverified-fidelity \
  --output /tmp/gemma4-loop43-literal-generation-execution.json
mv ./.gemma-4-E4B-dense-loop43-unavailable ./gemma-4-E4B-dense
```

Observed result:

- artifact bytes: `21,326,381,648`;
- `sourceCheckpointAccessed: false`;
- all twelve serialized assignment IDs executed once, from
  `generation_prefill` through `generation_terminal_cache`;
- instantiated outputs ran from `forward_state[0]` and `position[-1]` through
  `forward_state[1]`, `step_past_key_values[0..0]`, `terminal_logits` and
  `terminal_past_key_values`;
- generated token: `184` at position `1`;
- terminal logits shape: `[1,1,262144]`;
- terminal logits F32-byte SHA-256:
  `1a23dee2cd1f4c03e0b30d8300b00c9196ab8b11263903b58078af05cd528c96`;
- producer-owned KV layers: `0..23`;
- elapsed time: `56,046.884 ms`; maximum RSS: `784,400 KiB`;
- report bytes: `40,622`; report SHA-256:
  `63d1229fceb4fd5a049be8326ebfa6e63555a23aeb2f09c40fdf43616fa40c0d`.

The emitted token agrees with the prior authoritative PyTorch BF16 trace for
prompt `[2]`, but token agreement is not exact numerical fidelity. The report
truthfully records `unverified-fidelity-explicitly-acknowledged`; no checkpoint
marker is created.
