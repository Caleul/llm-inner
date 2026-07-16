# Gemma 4 audio feature lowering — 2026-07-16

This document records the executable F32 lowering boundary for the audio
encoder in the immutable `google/gemma-4-E4B` dense BF16 package. It is not a
Gemma 4 checkpoint claim: BF16 casting, composite assembly, literal export,
source-removed replay and an authoritative numerical differential remain
outside this slice.

## Authoritative semantic source

The implementation follows Hugging Face Transformers commit
`a8609bed2ad1593e7d756006525a10053d4d5bc6`,
`src/transformers/models/gemma4/modeling_gemma4.py`, specifically
`Gemma4AudioSubSampleConvProjection`, `Gemma4AudioAttention`,
`Gemma4AudioFeedForward`, `Gemma4AudioLightConv1d`, `Gemma4AudioLayer`, and
`Gemma4AudioModel.forward`. Configuration values are read from the package's
explicit `audio_config`; no audio default is supplied by this lowerer.

`src/gemma4-audio.ts` records and executes, in dependency order:

- masking before each 3x3 stride-2, padding-1, bias-free Conv2d and the exact
  `mask[:, ::2]` transitions;
- channel-only LayerNorm, ReLU, flatten order and the bias-free input
  projection;
- the descending sin/cos relative-position table, per-dimension softplus Q
  scale, K scale, block/context extraction, source-equivalent relative shift,
  tanh logit softcap and F32 softmax;
- both residual-weighted feed-forward branches, checkpointed input/output
  clipping, all RMSNorms, the causal depthwise local convolution and GLU; and
- the biased tower output projection, unscaled multimodal-embedder RMSNorm,
  language projection, valid-frame strip and exact audio-placeholder
  cardinality-checked scatter.

The program preserves each stable intermediate in `values`, including the
subsample masks, every feed-forward boundary, attention context, convolution
path and final audio soft tokens. The result is intentionally a feature branch:
it cannot be confused with a complete `Gemma4ForConditionalGeneration`
program.

## Validation

```bash
npm run typecheck
npm test
```

The focused synthetic fixture proves named-output coverage, two subsampling
mask transitions, padding isolation, local attention/convolution execution and
audio scatter rejection. It is scalar F32 control-flow evidence only, not a
comparison against the E4B BF16 runtime.

After compiling, the following header-only check against
`./gemma-4-E4B-dense` built a 12-layer, 485-assignment feature program from the
actual audited BF16 catalog; it did not materialize the 15 GB checkpoint:

```bash
node --input-type=module -e 'import { openCatalog } from "./dist/src/catalog.js"; import { buildGemma4AudioProgram } from "./dist/src/gemma4-audio.js"; const opened = await openCatalog("./gemma-4-E4B-dense", false); try { const p = buildGemma4AudioProgram(opened.catalog); console.log(JSON.stringify({ kind: p.kind, layers: p.tower.layers, assignments: p.assignments.length, output: p.output })); } finally { await opened.close(); }'
# {"kind":"gemma4-audio-features","layers":12,"assignments":487,"output":"audio_features"}
```

## Remaining boundary

The package still fails closed as a composite `gemma4` model. A composite
adapter must prepare pad-token text/PLE inputs, replace image/audio/video
placeholder embeddings in the runtime's declared order, pass the resulting
embeddings into the text graph, and preserve generation/cache/mask behavior.
