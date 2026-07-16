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

This command is intentionally a candidate replay, not evidence that the real
E4B run has completed. No real E4B prefill/cached decode was run in this
change, and no pinned authoritative Gemma 4 runtime trace exists. Vision,
video, and audio replacement stay unavailable on this paged path; do not use
it for multimodal prompts or create the Gemma 4 checkpoint marker.
