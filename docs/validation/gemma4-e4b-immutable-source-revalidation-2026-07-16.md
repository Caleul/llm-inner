# Gemma 4 E4B immutable-source revalidation — 2026-07-16

This is an independent rerun of the dense Gemma 4 text-only evidence after
recovering the official source package locally. It validates the existing
candidate boundary; it does **not** accept a Gemma 4 checkpoint or alter its
fidelity class.

## Source identity

- Hub model: `google/gemma-4-E4B`
- Immutable Hub revision: `411aa17b749aa952df1359d2dcea73917a544d9a`
- `model.safetensors`: 15,992,595,884 bytes,
  SHA-256 `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- `config.json`: SHA-256
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`
- Literal artifact: `artifacts/gemma4-e4b-dense.literal.json`, SHA-256
  `e81feb9061cabb9a1982c0b2ce890d540ea91f22034c99b1382e6283076c61e1`
- Text IR fingerprint:
  `08d540db2bf55c0166c72c33eaa5bcc7e376912074e5740d2c700d19681f1cbc`

The trace capture recomputed the source-file hashes before producing its
bundle. The artifact hash is the already audited streamed literal export hash;
the artifact itself remains an ignored 21,325,917,078-byte generated file.

## Reproduction

Capture the authoritative reference while the source package is available:

```bash
npm run capture:gemma4-text-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-e4b-current-generation.json \
  --input-tokens 2 --max-new-tokens 1 \
  --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Then make the source unavailable for the entire candidate process:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable
npm run compare:gemma4-paged-text-trace -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-current-generation.json \
  --report /tmp/gemma4-e4b-current-source-removed-differential.json \
  --max-read-mib 16 --top-k 10 --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
mv ./.gemma-4-E4B-dense-source-unavailable ./gemma-4-E4B-dense
```

## Result

The source-removed candidate reports `sourceCheckpointAccessed: false` and
completes the token-only prefill plus one cached greedy decode using only the
literal artifact. Both executions emit token `184`. The native trace has
producer KV layers `0..23`, and the candidate retains the same layer set and
shapes.

This remains `approximate`, not lossless within BF16:

- first divergence: `generation:selection-logits-0`;
- terminal logits max absolute error: `0.2066211700439453`;
- terminal logits cosine similarity: `0.9999728960119434`;
- terminal logits top-10 overlap: `1.0`; argmax agreement: `true`;
- all 24 producer KV cache comparisons diverge under zero tolerance.

The candidate declares scalar F32 arithmetic, whereas the authoritative
runtime executes eager BF16 operations with runtime-specific reduction and
rounding boundaries. Matching one greedy token therefore supplies no basis for
a dense-lossless checkpoint marker. Multimodal feature replacement and
operation-level native capture are also not established by this token-only
rerun.
