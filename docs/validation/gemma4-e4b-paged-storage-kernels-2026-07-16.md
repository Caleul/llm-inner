# Gemma 4 literal paged dense kernels — 2026-07-16

`src/paged-dense.ts` establishes the bounded storage-to-kernel contract needed
by source-independent dense E4B replay. It is intentionally not a claim that
the full Gemma 4 model has replayed.

## Contract

`createPagedDenseF32Matrix` accepts only an artifact-backed, unquantized,
row-major two-dimensional `F32`, `F16`, or `BF16` tensor. It requires the
literal reader's `readTensorBytesRange`, validates each requested row interval,
and widens storage only through the declared IEEE decoder. Quantized tensors,
non-matrices and a missing range API fail closed.

`pagedEmbeddingF32` reads only rows selected by caller token IDs (and caches
duplicate token rows inside that operation). `pagedLinearF32` visits bounded
contiguous output-row chunks, preserving the scalar F32 accumulation order
used by `linearF32`: output row followed by input column. Neither operation
opens a checkpoint nor builds a package-wide F32 map.

## Validation

```bash
npm run typecheck
npm test
```

The regression writes a self-contained Gemma 4 composite literal fixture,
clears its source tensor map, reopens only the JSON artifact, and runs bounded
embedding plus linear kernels through artifact ranges. A separate BF16 range
fixture checks exact IEEE widening and asserts that every request remains at
the configured one-row (4-byte) ceiling.

## Remaining boundary

The current composite executor is synchronous and still expects eager
`DenseF32Tensor` maps. The next coherent implementation must route its text
prefill and cached decode operations through this asynchronous provider,
including PLE, norms, RoPE, attention, producer-owned KV cache and the
vocabulary projection. Only then can the real E4B literal artifact be
executed without its source checkpoint and compared to an authoritative
runtime.
