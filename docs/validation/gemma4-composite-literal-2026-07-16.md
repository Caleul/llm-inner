# Gemma 4 composite literal replay — 2026-07-16

## Result

`Gemma4CompositeLiteralCalculationProgram` now serializes the established
dense Safetensors Gemma 4 composite execution boundary into one
source-independent JSON value. It embeds original storage bytes and a decoder
assignment for every registered text, vision, audio, projection and clipping
scalar tensor. It retains named scopes for the outer composite prelude, vision
tower, audio tower, text layers and terminal logits; those scopes make reuse of
the image tower for video explicit rather than collapsing it into an implicit
decoder call.

The artifact has no source checkpoint path. Its internal text model source is
the literal identifier `embedded://gemma4-composite-literal`; all actual data
comes from `constants[].payloadBase64` and `storageDecoders[]`.

## Validation

`node --test dist/test/gemma4-composite.test.js` passed 5/5 tests after build.
The new test creates the registered tiny dense F32 Gemma 4 composite catalog,
executes image prefill and two cached greedy steps against its original tensors,
builds the literal, removes the tensor source map, then executes the literal
again. It asserts bit-for-bit F32 logits, the greedy token sequence and final
generation logits. It also rejects a removed sliding-vision-mask assignment and
a replaced source path.

`npm run typecheck` passed.

## Deliberate boundary

This is not the mandatory E4B checkpoint evidence. The test uses a registered
small F32 fixture, not the 15 GB BF16 package, and does not compare to the
official Transformers runtime. The literal builder rejects a catalog tensor
without a registered semantic reference, so an unknown E4B tensor cannot be
silently omitted. E4B still requires every original BF16 payload to be read,
source-removed replay from the generated file, and a pinned authoritative
operation/KV/logit/generation differential before the checkpoint marker can
exist.
