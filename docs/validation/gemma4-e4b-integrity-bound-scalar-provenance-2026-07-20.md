# Gemma 4 E4B: integrity-bound learned-scalar provenance — 2026-07-20

## Completed boundary

Schema 60 already committed every dense payload as ordered, gap-free SHA-256
chunks and bound the complete `payloadIntegrity` table into the artifact
integrity manifest. Range reads enforced those hashes, but scalar audit output
reported only the tensor coordinate, storage address, bits, decoder and decoded
literal. A reader could not navigate from a substituted number back through
the exact authenticated chunks to the artifact root.

The range reader now returns that proof as a cohesive storage-level contract.
Every learned scalar and every `--tensor` selection records:

- the exact requested byte offset and length;
- the whole payload byte length and SHA-256;
- the JSON pointer to that payload commitment;
- every covering chunk, with ordinal, byte range, SHA-256 and JSON pointer;
- the `payloadIntegrity` section SHA-256 and integrity-manifest pointer; and
- the artifact integrity root SHA-256.

Bytes are returned before the proof only after every covering chunk has passed
canonical RFC 4648 Base64 validation and its SHA-256 check. A scalar that spans
or a range that crosses a chunk boundary lists both chunks in order. The
implementation is tensor-, shape-, dtype-, operation- and coordinate-driven;
it contains no Gemma layer allowlist.

## Real source-removed evidence

The dense unquantized `google/gemma-4-E4B` artifact at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a` was opened while
`./gemma-4-E4B-dense` was physically moved into a private `mktemp` directory
under an `EXIT`/`INT`/`TERM` restoration trap.

Artifact identity:

- schema: `60`;
- bytes: `21,410,777,508`;
- artifact SHA-256:
  `62b18d0a34d0fbc8eb1c22b830a742a43afbe8559e0f7e5358042060981a7c66`;
- constants / storage decoders: `2,130 / 2,130`;
- integrity root:
  `ee45517d8cfa038f22ec496b36605b0e8c602e62566c705431678cad8ecb4404`;
- `payloadIntegrity` section SHA-256:
  `c456a582170123ec47e79c3bfc9713b89f635c92e6f55399469f6461235f54f8`;
  and
- unresolved native reductions: `100`.

The source-removed scalar request selected `layer_0_q_proj[0,0,0]` with the
explicit diagnostic input window `0..8`. All eight learned operands were real
numeric literals. The first was:

```text
tensor: model.language_model.layers.0.self_attn.q_proj.weight
indices: [0,0]
storage range: offset 0, length 2
storage bits: 0xbbcb
decoded F32 bits: 0xbbcb0000
literal: -0.006195068359375
payload pointer: /payloadIntegrity/1416
chunk pointer: /payloadIntegrity/1416/chunking/chunks/0
manifest pointer: /integrityManifest/sections/23
```

That BF16 payload contains `10,485,760` bytes and has SHA-256
`7578a28179d32008e2bdf6da2a5432090b66ea960cb1e449b71b64654cb6243d`.
The independently selected first 4,096 bytes produced SHA-256
`625682abe42e9ee5a1977af081cf328cde6a71da76e0d6e310f60354941ecef0`
and navigated through the same payload, chunk, manifest section and root.

The scalar-view report contains `892,314` bytes with SHA-256
`262097c59c3971e469f285226ec894119503c9708af5d29396a507414b1fd428`.
The tensor-range report contains `773,049` bytes with SHA-256
`0f5bd37c9349b315658ec8245fcd91a035e41e5e15c521723102b14bdd27f4ae`.
The trap restored the checkpoint and removed its private isolation directory.

## Commands

```text
npm run build
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --operation layer_0_q_proj --output-coordinate 0,0,0 \
  --input-start 0 --input-count 8 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /private/tmp/gemma4-e4b-authenticated-scalar-view.json

node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --tensor model.language_model.layers.0.self_attn.q_proj.weight \
  --offset 0 --byte-length 4096 \
  --output /private/tmp/gemma4-e4b-authenticated-tensor-range.json
```

## Preserved fidelity boundary

This closes the integrity provenance from each substituted learned literal to
the artifact root. It does not invent Apple Accelerate SGEMM's unpublished
scalar product-rounding, lane, tile or fold schedule. The fidelity gate remains
`blocked-on-runtime-reduction`, `exactReplayClaim=forbidden`, with exactly
`100` unresolved native reductions.
