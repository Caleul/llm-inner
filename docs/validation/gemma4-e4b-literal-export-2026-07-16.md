# Gemma 4 E4B dense literal export — 2026-07-16

This records a real artifact-materialization boundary for the immutable dense
Gemma 4 package. It is not a Gemma 4 dense-lossless checkpoint claim: no
authoritative Transformers operation/KV/logit comparison or source-removed
forward/generation replay has yet been completed.

## Source and artifact

- Source: `google/gemma-4-E4B` revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Source identity remains the audit in
  [`gemma4-e4b-source-audit-2026-07-16.md`](gemma4-e4b-source-audit-2026-07-16.md):
  `model.safetensors` SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- Local artifact (ignored because it is 21.3 GB):
  `artifacts/gemma4-e4b-dense.literal.json`.
- Artifact SHA-256:
  `e81feb9061cabb9a1982c0b2ce890d540ea91f22034c99b1382e6283076c61e1`.
- Artifact bytes: `21,325,917,078`; embedded original storage bytes:
  `15,992,314,836`.

## Reproduction and validation

```bash
npm run build
node dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output ./docs/validation/gemma4-e4b-literal-export-2026-07-16.json
```

The streamed audit parses neither the 21 GB JSON nor a full payload into the
V8 heap. It hashes the raw artifact, counts the deliberately emitted base64
fields, totals their decoded storage bytes, compares both count and bytes to
the catalog, and rejects the absolute source directory if it appears in the
artifact. The recorded result is 2,130 constants, 15,992,314,836 embedded
bytes, and `forbiddenSourcePathPresent: false`.

The writer reads Safetensors ranges in 12 MiB, three-byte-aligned chunks and
atomically renames the complete JSON only after the final byte is flushed. This
avoids both package-wide base64 accumulation and Node's maximum single-string
limit for the multi-gigabyte token embedding.

## Shared-KV storage provenance

The established 42-layer Gemma 4 graph uses producer-owned KV state for layers
24–41. The dense checkpoint nevertheless contains 54 local tensors for those
consumer layers: `k_norm.weight`, `k_proj.weight`, and `v_proj.weight` for
each layer. They are embedded losslessly, but are listed in
`unreachableConstants` with the reason
`shared-kv-consumer-local-kv-is-runtime-unreachable` and the declared producer
layer. This exception is derived from the explicit `kvSharing` assignments;
it is not a generic tensor-name omission rule. The remaining 2,076 constants
are referenced by ordered composite, vision, audio, or text assignments.

## Remaining checkpoint gates

The artifact currently uses the declared scalar F32 replay policy, not the
official BF16 runtime kernel behavior. A source-removed artifact reader must
be able to selectively decode the 21 GB JSON without `JSON.parse`, then run a
real multimodal prefill and cached greedy decode. Finally, the same inputs and
states need a pinned authoritative Transformers trace with operation, KV,
logit, and generation metrics. Until those measurements exist, do not create
`.agent-loop/checkpoints/gemma4-dense-lossless/`.
