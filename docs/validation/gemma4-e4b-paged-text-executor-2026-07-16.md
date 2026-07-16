# Gemma 4 paged text literal executor — 2026-07-16

`src/gemma4-paged-text.ts` is the source-independent execution boundary for
the declared `Gemma4Text` part of a streamed composite literal artifact.

## Contract

`executeGemma4PagedTextLiteralF32` accepts an opened
`OpenGemma4CompositeLiteralArtifact`, text token IDs, optional positions and
the canonical post-RoPE producer-owned KV cache. It resolves every constant
through the artifact index. Dense two-dimensional F32/F16/BF16 weights are
fed to `pagedEmbeddingF32` or `pagedLinearF32`; one-dimensional norm and
scalar tensors are read only through `readPagedDenseF32Vector`. Each read is
bounded by `maxReadBytes`; no source shard, catalog path, or package-wide F32
weight map is available to this module.

The interpreter runs all declared text prelude, layer and epilogue operations:
PLE embedding/projection, RMS norms, scalar F32 elementwise boundaries,
projections, BHSD reshape, Gemma RoPE, causal/sliding or caller-declared mask
attention, producer-owned shared KV, GELU, residuals, and the final vocabulary
projection. Any non-F32 policy, quantized reference, unmatched literal
constant, unsupported bias, or missing/incompatible cache fails closed.

`generateGemma4PagedTextLiteralF32` evaluates the selected token through the
same path and returns the new producer cache, so its cache contract matches
the normal F32 reference generator.

## Validation

```bash
npm run typecheck
npm test
```

The `Gemma 4 paged text interpreter` regression writes a complete registered
Gemma 4 literal fixture, clears every source tensor, reopens the artifact and
compares paged prefill logits plus two cached greedy decode steps byte-for-byte
with the eager composite executor. The test uses a 64-byte range ceiling,
including projection rows that exceed a single F32 hidden vector.

## Real E4B status

The local 21.3 GB artifact can now be addressed by the text-only CLI:

```bash
npm run replay:gemma4-paged-text -- --artifact ./artifacts/gemma4-e4b-dense.literal.json --input-ids 2,106,3 --max-new-tokens 1 --output ./paged-text-report.json
```

## Real E4B source-removed text replay

On 2026-07-16, the source directory was atomically renamed away for the full
command, then restored after it exited. The only model input available to the
executor was the literal artifact. The source file and artifact hashes were
recomputed immediately afterwards:

- `gemma-4-E4B-dense/model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- `artifacts/gemma4-e4b-dense.literal.json` SHA-256:
  `e81feb9061cabb9a1982c0b2ce890d540ea91f22034c99b1382e6283076c61e1`
- Artifact bytes: `21,325,917,078`; source checkpoint access: `false`.

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable
npm run replay:gemma4-paged-text -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --input-ids 2 --max-new-tokens 1 --max-read-mib 16 \
  --output /tmp/gemma4-e4b-paged-text-replay-0018.json
mv ./.gemma-4-E4B-dense-source-unavailable ./gemma-4-E4B-dense
```

The command completed a one-token prefill and one cache-backed greedy decode
in `62,917.933 ms`, yielding generated token `184`, final logits shape
`[1, 1, 262144]`, and logits SHA-256
`9fbd1e292c8f0d493c1d848e06c4821b6564a00c996c294a4fa052ff73a2ab4d`.
It reported all 24 producer-owned KV layers (`0` through `23`), a 16 MiB read
ceiling, RSS before/after of `230,375,424` / `777,388,032` bytes, and a
process high-water mark of `789,472 KiB`. The CLI now includes that high-water
mark as `maxRssKiB` in every replay report.

This is source-independent **text-only candidate replay evidence**, not a
Gemma 4 dense-lossless checkpoint claim. It does not compare against the
official BF16 runtime and it does not execute image, video, or audio feature
replacement. Vision, video, and audio remain fail-closed on this paged path;
do not use this result to create the Gemma 4 checkpoint marker.
