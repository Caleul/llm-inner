# Gemma 4 E4B dense source audit — 2026-07-16

This is source-identity and semantic-contract evidence for the mandatory
Gemma 4 checkpoint. It is deliberately **not** a claim of literal export,
source-removed replay, or authoritative-runtime fidelity.

## Immutable source

- Hugging Face package: `google/gemma-4-E4B`
- Hugging Face revision: `411aa17b749aa952df1359d2dcea73917a544d9a`
- Container: one dense BF16 `model.safetensors`, 15,992,595,884 bytes
- `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`
- Authoritative runtime/source reviewed: Hugging Face Transformers
  `a8609bed2ad1593e7d756006525a10053d4d5bc6`,
  `models/gemma4/configuration_gemma4.py` and `modeling_gemma4.py`.

The machine-readable audit retains the tokenizer/generation metadata checksums
as well as every type-specific decoder-layer fact:
[`gemma4-e4b-source-audit-2026-07-16.json`](gemma4-e4b-source-audit-2026-07-16.json).

## Established contract

The inspected official checkpoint has 2,130 BF16 tensors and a full
multimodal `gemma4` outer configuration. Its 42-layer text model is hidden
size 2,560 with eight query heads, two KV heads, sliding-head dimension 256,
full-attention head dimension 512, a 10,240-wide gated MLP, and 18 trailing
shared-KV layers. The last non-shared sliding producer is layer 22; the last
non-shared full-attention producer is layer 23. Layers 24–40 reuse layer 22;
full-attention consumers 29, 35, and 41 reuse layer 23.

PLE is part of this package: the packed token table is
`[262144, 42 * 256]`, `per_layer_model_projection` is `[42 * 256, 2560]`,
and every layer has the input gate, per-layer projection, post-PLENorm, and
scalar. The authoritative implementation combines scaled token identity PLE
with RMS-normalized projected context, applies the per-layer gated residual
after the MLP residual, and applies `layer_scalar`.

The outer model carries image, audio, and video placeholder token IDs and
contains vision/audio towers plus their projections. Their injection order and
the PLE behavior for multimodal embeddings are therefore executable model
semantics, not preprocessing details. The source contract now validates every
registered tower tensor role, not merely the namespaces: the 16-layer vision
encoder is 768-wide MHA with 64-wide heads, 2-D patch positions, RoPE θ=100,
pooling, and a 280-token default output; the 12-layer audio encoder is
1024-wide with 128-wide heads, two convolutional subsamplers (128 then 32
channels), local convolution, chunked attention and a 1,536-wide output
projection. Every vision/audio projection is a `Gemma4ClippableLinear`: its
four checkpointed BF16 scalar bounds clamp the input and output around the
linear operation, so they cannot be discarded as calibration metadata.

## Reproduction

```bash
hf download google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --local-dir ./gemma-4-E4B-dense \
  config.json generation_config.json tokenizer.json tokenizer_config.json model.safetensors
npm run audit:gemma4 -- --source ./gemma-4-E4B-dense \
  --output docs/validation/gemma4-e4b-source-audit-2026-07-16.json
```

## What remains deliberately unsupported

The compiler still fails closed for Gemma 4. The text core now lowers PLE,
proportional RoPE, producer-owned KV and four-norm decoding; the remaining
vertical implementation must turn the established vision/audio contracts and
placeholder replacement into explicit executable assignments, then capture a
Transformers operation/KV/logit/greedy differential. It must finally
materialize the complete BF16 package into a literal source-independent
program. No checkpoint marker may be created before those steps pass.
