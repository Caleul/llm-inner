# Gemma 4 E4B literal artifact reader — 2026-07-16

This validation records the first bounded-memory reader for the real dense
Gemma 4 E4B literal calculation artifact. It is an artifact-structure and
storage-range result, not an authoritative numerical-fidelity result.

## Command and observed result

```bash
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --tensor model.language_model.embed_tokens.weight \
  --offset 0 --byte-length 4096
```

The command completed successfully against the 21,325,917,078-byte artifact:

- 2,130 constants and 2,130 dense storage decoders were indexed.
- The embedded text source is exactly
  `embedded://gemma4-composite-literal`; the reader opens only the literal
  JSON.
- `model.language_model.embed_tokens.weight` is `BF16`, shape
  `[262144, 2560]`, with a 1,342,177,280-byte embedded payload.
- The artifact-only first 4,096-byte range hash is
  `00291ac7eda6e65b0cdd44a304d18311bd1459d7b811cab4cf78d21356d9e22d`.

For a storage-integrity spot check, the same 4,096-byte range was read from
the pinned `gemma-4-E4B-dense/model.safetensors`; it produced the identical
SHA-256. This comparison used the source only as evidence. The literal-reader
command itself does not open it.

## Boundaries proved

`openGemma4CompositeLiteralArtifact` scans the writer's constrained base64
fields, validates every dense decoder and the complete embedded Gemma 4
assignment graph, and retains payload offsets instead of `payloadBase64`
strings. It can decode an arbitrary byte range by translating logical storage
bytes to the enclosing base64 quadruplets. The source-removed fixture test
deletes its checkpoint path before opening the literal artifact and verifies
both full-tensor and unaligned-range equality.

The real index run used 229,965,824 maximum resident bytes and completed in
5.10 seconds wall-clock on this machine. Those figures cover structural
indexing plus a 4 KiB range read, not model execution.

## Remaining checkpoint gap

The current reference executor still converts every embedded BF16 tensor into
an in-memory F32 `DenseF32Tensor`. The E4B artifact contains 15,992,314,836
storage bytes, which becomes roughly 29.8 GiB of F32 weights before runtime
activations, KV cache, and duplicate maps; this host reports 36 GiB physical
memory. Consequently, this reader is necessary but not sufficient for the
required source-removed E4B prefill and cached generation. No Gemma 4 success
marker is created by this result, and no BF16 runtime differential claim is
made.
