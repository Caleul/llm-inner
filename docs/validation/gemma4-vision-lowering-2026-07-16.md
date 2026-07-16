# Gemma 4 vision lowering boundary — 2026-07-16

This is an executable lowering of the image/video feature branch of the
immutable dense `google/gemma-4-E4B` package audited in
`gemma4-e4b-source-audit-2026-07-16.json`. It is not a Gemma 4 checkpoint
claim: audio lowering, composite text assembly, literal export, source-removed
replay, and authoritative differential validation are still absent.

## Semantic evidence

The package declares `transformers_version: 5.5.0.dev0`. The semantic source
used for this boundary is Hugging Face Transformers `v5.5.0`, commit
`c1c34249fa27deefbd4a377dfbf883a39baf5c6d`, specifically
`src/transformers/models/gemma4/modeling_gemma4.py`:

- `Gemma4ClippableLinear`: clamp input by four checkpointed scalar operands,
  project without bias, then clamp output.
- `Gemma4VisionPatchEmbedder`: `2 * (pixel_values - 0.5)`, patch projection,
  separate x/y table lookups, and zeroed padding embeddings.
- `Gemma4VisionAttention`: head-local RMSNorm; x/y-partitioned default RoPE;
  non-causal attention at scale `1`; F32 softmax; unscaled V RMSNorm.
- `Gemma4VisionPooler`: zero padding patches, pool by spatial coordinates,
  average in F32, then apply F32 `sqrt(hidden_size)` scale before stripping
  invalid pooled cells.
- `Gemma4MultimodalEmbedder`: unscaled RMSNorm of soft tokens before projecting
  into text hidden space; and `Gemma4Model.forward` requires
  exact image-placeholder/feature cardinality before `masked_scatter`.

The implementation is `src/gemma4-vision.ts`. Its program records stable
names for patching, positional embedding, every attention/MLP residual,
two-dimensional RoPE, spatial pooling, projection, and the future composite
image scatter. `executeGemma4VisionF32` returns each executed named value;
`scatterGemma4ImageFeaturesF32` implements the exact cardinality-checked
scatter boundary separately so it cannot be silently skipped by a later
composite adapter.

## Validation scope

`test/gemma4-vision.test.ts` builds a dense registered miniature composite
catalog, executes a four-patch image through the full vision sequence,
asserts every non-composite assignment output exists, and rejects malformed
padding coordinates and image-placeholder cardinality. It is synthetic scalar
F32 coverage, not a BF16 comparison against the real E4B runtime.
