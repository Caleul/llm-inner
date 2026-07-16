# Gemma 4 E4B native text differential — 2026-07-16

This is the first authoritative-runtime comparison for the local dense
`google/gemma-4-E4B` package. It is deliberately limited to text token
prefill plus one cached greedy decode; image, video and audio replacement are
not evidence from this run.

## Immutable inputs

- Checkpoint SHA-256: `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- `config.json` SHA-256: `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`
- Literal artifact SHA-256: `e81feb9061cabb9a1982c0b2ce890d540ea91f22034c99b1382e6283076c61e1`
- Reference: PyTorch `2.12.1`, Transformers `5.5.0`, eager BF16
  `Gemma4ForConditionalGeneration`; values are widened to F32 only for
  comparison serialization.
- Candidate: `llm-inner paged Gemma4Text literal F32`, reading bounded ranges
  only from the literal artifact.

## Reproduction

Capture the reference while the source checkpoint is present:

```bash
npm run capture:gemma4-text-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-e4b-transformers-text-generation.json \
  --input-tokens 2 --max-new-tokens 1 \
  --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651
```

Then make the source unavailable for the whole candidate command:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable
npm run compare:gemma4-paged-text-trace -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-transformers-text-generation.json \
  --report /tmp/gemma4-e4b-paged-native-source-removed.json \
  --max-read-mib 16 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense
mv ./.gemma-4-E4B-dense-source-unavailable ./gemma-4-E4B-dense
```

The trace binds both source-file hashes and a semantic IR fingerprint. The
fingerprint ignores source location and preview controls because the hashes
bind checkpoint identity and preview settings may not alter the program.

## Result

The source-removed candidate completed. Both sides selected token `184` at
position `1` for prompt `[2]`, and both retained all producer KV layers `0..23`
with matching shapes. The comparison is nevertheless `approximate`, with first
divergence `generation:selection-logits-0`:

- selection logits: max absolute error `0.19899940490722656`, cosine
  `0.9999734672320224`, top-10 overlap `1.0`, argmax agreement `true`;
- terminal decode logits: max absolute error `0.2066211700439453`, cosine
  `0.9999728960119434`, top-10 overlap `1.0`, argmax agreement `true`;
- every producer KV cache diverged under exact F32 comparison; the largest
  value-cache absolute error was `0.04873722791671753` at layer 17.

This is artifact-only text replay plus a real native differential, not lossless
BF16 fidelity. The next boundary is an explicit BF16 operation/rounding
contract and operation-level native capture; no Gemma 4 checkpoint marker is
justified by this text-only approximate result.
