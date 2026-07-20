# Gemma 4 E4B: bounded source-identity navigation — 2026-07-20

## Completed boundary

The schema-58 artifact embeds the immutable package metadata, including the
32,170,070-byte `tokenizer.json`, inside `sourceIdentity`. The streaming reader
previously copied those Base64 payloads into every report even when the caller
requested no source file. A default inspection consequently emitted roughly
43 MiB before presenting the calculation graph.

The reader now exposes a bounded source-identity surface:

- the default report preserves model, revision, file roles, byte counts,
  SHA-256 commitments and storage/decoding contracts without copying metadata
  payloads;
- aggregate file, package-byte and embedded-metadata-byte counts make omission
  visible;
- `--source-file`, `--source-offset` and `--source-byte-length` decode one
  explicit metadata range from the artifact;
- each selected range carries Base64 bytes, range SHA-256 and the committed
  whole-file byte count and SHA-256; and
- weights fail closed on this route and remain navigable through named tensor
  constants, so learned bytes retain one authority.

The source-identity validator still verifies every embedded metadata payload
and its whole-file commitment before either summary or range navigation is
accepted. Unknown paths, weight files and out-of-bounds windows fail closed.

## Real source-removed validation

`./gemma-4-E4B-dense` was moved into a private `mktemp` directory under an
`EXIT`/`INT`/`TERM` restoration trap. While that path was absent, the real
artifact opened and navigated the first 4,096 bytes of `tokenizer.json`:

- artifact schema / bytes: `58` / `21,409,397,092`;
- immutable model / revision: `google/gemma-4-E4B` /
  `411aa17b749aa952df1359d2dcea73917a544d9a`;
- committed source files / bytes: `6` / `16,024,773,810`;
- embedded metadata bytes: `32,177,926`;
- tokenizer bytes / SHA-256: `32,170,070` /
  `12bac982b793c44b03d52a250a9f0d0b666813da566b910c24a6da0695fd11e6`;
- selected range: offset `0`, `4,096` decoded bytes;
- selected range SHA-256:
  `8bd82f028afb204dc534bc7f163506a6982814e9bb246d9812af9529dbfe472a`;
- report bytes: `280,606`; and
- `sourceCheckpointAccessed=false` with no `payloadBase64` field in the
  summarized identity.

The restoration postcondition verified the checkpoint directory was back in
place and the temporary isolation directory was removed.

## Preserved fidelity boundary

This closes source-package navigation only. It does not reinterpret the 100
Apple Accelerate BMM scalar schedules: the schema-58 fidelity gate remains
`blocked-on-runtime-reduction` with `exactReplayClaim=forbidden`.

## Validation

```text
npm run typecheck
npm run build
node --test dist/test/gemma4-composite.test.js
```

The focused run passed 35 tests with 0 failures. Coverage includes summary
payload omission, exact metadata range decoding, weight-route rejection and
out-of-bounds rejection.
