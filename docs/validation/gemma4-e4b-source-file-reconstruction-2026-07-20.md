# Gemma 4 E4B: source-file reconstruction from literal constants — 2026-07-20

## Completed boundary

The schema-58 artifact committed the complete SHA-256 and byte length of each
source Safetensors file, and embedded every tensor payload as a named literal
constant. It did not retain the exact eight-byte prefix, serialized header or
the source byte ranges connecting those constants back into the original
container. A source-removed reader could therefore verify every constant but
could not reconstruct and independently re-hash the committed weight file.

Schema 59 closes that gap for every dense Gemma 4 Safetensors shard:

- every source byte belongs to one ordered, gap-free segment;
- prefix/header and any non-tensor gap/trailer bytes are embedded losslessly;
- every tensor segment names exactly one literal constant and preserves its
  absolute byte offset and byte length;
- the embedded header's `dtype`, `shape` and `data_offsets` must match that
  constant;
- every literal constant must occur exactly once across all source shards;
- `--source-file` can read a bounded range spanning structural and tensor
  segments without opening the checkpoint; and
- `--verify-source-files` streams every segment, reconstructs every package
  file and checks its original immutable SHA-256.

Unknown, missing, duplicated, shifted, overlapping or incomplete mappings fail
before replay. The mapping is generic over source shard, tensor name, dtype,
shape and byte range; it contains no layer- or assignment-specific list.

## Real artifact and source-present audit

The dense unquantized `google/gemma-4-E4B` package at revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was regenerated as schema 59:

- artifact bytes: `21,410,076,484`;
- artifact SHA-256:
  `ca00b2b28d9e51f1ae0dd458cd65716d6b16d7c548b77a4fc980f001104d31c5`;
- constants / storage decoders: `2,130 / 2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable package files / bytes: `6 / 16,024,773,810`;
- embedded metadata bytes: `32,177,926`;
- embedded Safetensors structural bytes: `281,048`; and
- structural integrity root:
  `4fbd1492a6a67038a20ab991b67632e11409e7cbe82c94f08400f56e8426ac2c`.

The source-present audit compared every learned byte and confirmed that the
absolute checkpoint path does not occur in the artifact.

## Checkpoint physically unavailable

`./gemma-4-E4B-dense` was moved into a private `mktemp` directory under an
`EXIT`/`INT`/`TERM` restoration trap. While its declared path was absent, the
schema-59 reader reconstructed and re-hashed all six package files from the
JSON alone:

- reconstructed package bytes: `16,024,773,810`;
- reconstructed weight files: `1`;
- reconstructed `model.safetensors` bytes: `15,992,595,884`;
- reconstructed `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`;
- selected weight-file range: offset `0`, length `4,096`;
- selected range SHA-256:
  `55afe7d230c23b7d2519be6bf9f1c7d593fdb648995e5ba2bfaf5db2dd318860`;
- source-removed reconstruction report bytes: `778,467`; and
- report SHA-256:
  `acd9da644cc377ead82b505ff0ae480aa4a9e83941fe76e1136579426af5923d`.

After restoration, hashing the live first 4,096 source bytes independently
produced the same range digest. The trap restored the checkpoint and removed
its private isolation directory.

## Source-removed authoritative differential

The regenerated artifact then replayed the current image, video and audio
authoritative captures at zero absolute and relative tolerance while the
checkpoint path was again physically absent. The pinned artifact-owned runtime
adapter executed all native reductions in declared order:

- zero-tolerance prefill passes: `3/3`;
- zero-tolerance greedy-generation passes: `3/3`;
- runtime-defined reductions: `100`;
- first divergence: `null`;
- report bytes: `1,142,854`; and
- report SHA-256:
  `10c3527741ea5a3304c2f40efa4e093590893524608a6d1b1674e60452ead48e`.

## Preserved fidelity boundary

Source-file reconstruction proves that the artifact owns the exact original
package bytes, including the container structure around all learned payloads.
It does not publish Apple Accelerate SGEMM's scalar product rounding, lane,
tile or fold schedule. The fidelity gate therefore remains
`blocked-on-runtime-reduction`, `exactReplayClaim=forbidden`, with exactly
`100` unresolved native reductions.

## Validation

```text
npm run typecheck
npm test
```

The repository suite passed `261` tests: `258` passed, `0` failed and `3`
optional non-Gemma external integrations were skipped. The pinned Gemma 4
embedded-adapter integration tests ran and passed.
