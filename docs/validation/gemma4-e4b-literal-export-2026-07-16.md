# Gemma 4 E4B dense literal export — 2026-07-16

This records the real artifact-materialization boundary for the immutable dense
Gemma 4 package, including a source-removed, authoritative-runtime operation
comparison. It is not a Gemma 4 dense-lossless checkpoint claim: the measured
text path remains approximate, and multimodal and source-removed generation
fidelity are not established.

## Source and artifact

- Source: `google/gemma-4-E4B` revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Source identity remains the audit in
  [`gemma4-e4b-source-audit-2026-07-16.md`](gemma4-e4b-source-audit-2026-07-16.md):
  `model.safetensors` SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- Local artifact (ignored because it is 21.3 GB):
  `artifacts/gemma4-e4b-dense.literal.json`.
- Artifact SHA-256 after the 2026-07-17 schema-v2 generation-program
  regeneration: `bca3508a12de591e6cf0761f4c8baf77ff340c16e8b9ce63bdb065723a056510`.
- Artifact bytes: `21,326,381,648`; embedded original storage bytes:
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
  --output ./docs/validation/gemma4-e4b-literal-export-2026-07-16.json \
  --verify-gemma4-payloads
```

The streamed audit parses neither the 21 GB JSON nor a full payload into the
V8 heap. It hashes the raw artifact, counts the deliberately emitted base64
fields, totals their decoded storage bytes, compares both count and bytes to
the catalog, and rejects the absolute source directory if it appears in the
artifact. The recorded result is 2,130 constants, 15,992,314,836 embedded
bytes, and `forbiddenSourcePathPresent: false`.

The Gemma-specific payload verification then opens the literal artifact through
its bounded range reader and compares every constant to the source tensor of
the same name, dtype and shape. On 2026-07-17 it compared all 2,130 payloads
and all `15,992,314,836` storage bytes with no mismatch. The canonical SHA-256
over source storage bytes in sorted tensor-name order was
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`, and
the literal payload stream produced the same digest. This proves the embedded
storage export while the source is available; it neither proves the BF16 kernel
schedule nor replaces the source-removed runtime differential.

The writer reads Safetensors ranges in 12 MiB, three-byte-aligned chunks and
atomically renames the complete JSON only after the final byte is flushed. This
avoids both package-wide base64 accumulation and Node's maximum single-string
limit for the multi-gigabyte token embedding.

## Source-removed payload integrity

The current streamed writer records one SHA-256 commitment per named payload
in the structural tail. `npm run inspect:gemma4-literal -- --artifact
<literal.json> --verify-payloads` range-decodes every payload and compares it
to that embedded commitment without opening a checkpoint. This is a
post-export corruption check, not proof that a newly edited payload/hash pair
originated in the source; the source-to-literal comparison above remains the
required pre-removal evidence.

The E4B artifact recorded above was regenerated on 2026-07-17 from the pinned
immutable source, so it carries all 2,130 commitments. The post-regeneration
source-byte comparison reproduced the 2,130-payload, 15,992,314,836-byte
digest equality stated above. A subsequent full commitment scan completed with
the source directory renamed away; it reported the same sorted storage digest
without opening a checkpoint. See
[`gemma4-e4b-literal-integrity-2026-07-17.md`](gemma4-e4b-literal-integrity-2026-07-17.md)
for the exact commands and result.

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

The artifact header now truthfully declares
`accumulationDtype: "operation-declared"`: Gemma4Text linear/RMSNorm
assignments state their complete reduction schedule. Most registered
linear/RMSNorm assignments use ordered F64 scalar accumulation. The measured
E4B `layer_0_gate_proj` and `layer_0_up_proj` instead declare the complete
32-lane, eight-register ARM BF16 FMA tree with its pairwise horizontal fold.
The streaming reader rejects missing or invalid reduction schedules. This is an
explicit trace-bound compatibility policy, not a claim that storage dtype alone
defines a native kernel. The source-removed exact-operation campaigns are
limited to these two layer-0 projections; text end-to-end fidelity remains
approximate, and the vision/audio towers still lack paged storage-backed
execution. Do not create
`.agent-loop/checkpoints/gemma4-dense-lossless/`.
