# Gemma 4 E4B regenerated literal integrity — 2026-07-17

This records the storage-integrity boundary for the real dense E4B literal
artifact after regenerating it with the streamed writer's per-payload SHA-256
commitments. It is not a numerical-fidelity or checkpoint-success claim.

## Immutable source and regenerated artifact

- Hub model: `google/gemma-4-E4B`, revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Source `model.safetensors`: SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- Literal artifact: `artifacts/gemma4-e4b-dense.literal.json` (ignored,
  reproducibly generated), 21,326,356,937 bytes, SHA-256
  `fb46de6b4ed24cad96e37e3223d8a1c40942b87342f65290a59a0680e90bfd3d`.
- Embedded storage: 2,130 named dense payloads and 15,992,314,836 bytes.
- Sorted-name storage digest:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

## Reproduction and observed results

Regenerate atomically while the pinned source is present:

```bash
npm run build
node dist/src/cli.js --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal
```

Before removal, compare every source storage range to the name-matched literal
payload:

```bash
npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-e4b-payload-audit.json \
  --verify-gemma4-payloads
```

The observed result matched all 2,130 payloads and all 15,992,314,836 bytes.
The source and literal sorted-name storage SHA-256 values were equal to the
digest above; the artifact contained no source-directory path.

For the source-removed integrity check, temporarily make the exact source
directory unavailable for the entire reader command, then restore it:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable
npm run inspect:gemma4-literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-e4b-source-removed-integrity.json
mv ./.gemma-4-E4B-dense-source-unavailable ./gemma-4-E4B-dense
```

The artifact-only reader indexed 2,130 constants and 2,130 storage decoders,
reported `payloadIntegrityCommitted: true` and
`sourceCheckpointAccessed: false`, asserted the former source path was absent,
and range-decoded all
15,992,314,836 embedded bytes. Its computed storage digest matched the same
value above. This reader command has no source argument and opens only the
literal JSON.

## Independent source-removed rerun — loop 22

After the verifier gained the enforced `--assert-source-unavailable` guard,
the command above was rerun against the same ignored artifact while
`./gemma-4-E4B-dense` was moved to the exact temporary path shown. The path
was restored after the command. The persisted report recorded 2,130 constants,
15,992,314,836 decoded payload bytes, sorted-name storage digest
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`,
`sourceCheckpointAccessed: false`, and
`assertedUnavailableSource: "./gemma-4-E4B-dense"`.

## Limit

These checks prove the dense storage export and detect a changed embedded
payload after source removal. They do not establish the unresolved eager-BF16
`layer_0_up_proj` numerical schedule, multimodal paged execution, or
authoritative generation fidelity. No `gemma4-dense-lossless` checkpoint marker
is justified by this evidence.
