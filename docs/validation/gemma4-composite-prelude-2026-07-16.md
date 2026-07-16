# Gemma 4 composite prelude boundary — 2026-07-16

This document records the executable F32 composite boundary for the immutable
dense `google/gemma-4-E4B` package. It is not a dense-BF16 checkpoint claim:
literal export, source-removed replay and authoritative runtime comparison
remain outside it.

## Authoritative order

The implementation follows Hugging Face Transformers commit
`a8609bed2ad1593e7d756006525a10053d4d5bc6`,
`src/transformers/models/gemma4/modeling_gemma4.py`, in
`Gemma4Model.forward`, `get_video_features`, and the Gemma4Text PLE helpers.
The explicit ordered program in `src/gemma4-composite.ts` now declares:

1. independent image/video/audio placeholder masks from `input_ids`;
2. replacement of every soft-token ID with `text_config.pad_token_id` before
   the initial text embedding and packed PLE identity lookup;
3. the existing image program, video frame flattening followed by that same
   vision program, then the audio program;
4. cardinality-checked scatters in the runtime order image, video, audio; and
5. PLE context projection, scale, reshape, RMSNorm and combination with the
   PAD-derived identity PLE *after* all three scatters, before entering the
   named Gemma4Text layer assignments.

The same pinned source's `get_block_sequence_ids_for_mask` and
`create_masks_for_vision_model` are now explicit F32 assignments too. Types
`1` (image) and `2` (video) form a monotonically numbered contiguous block;
other token types become `-1`. The `full_attention` mask remains strictly
causal. The `sliding_attention` mask admits a key iff it is within the strict
source window (`key > query - sliding_window`) and is either causal or belongs
to the same non-negative vision block. The layer-mask seam gives each lowered
attention operation this complete topology, so its usual local causal loop
cannot erase the permitted future vision keys.

`Gemma4ForConditionalGeneration.prepare_inputs_for_generation` removes
`mm_token_type_ids` after the first cached forward. The composite greedy F32
path preserves that transition: vision-aware masks are built for prefill only,
then each decode receives the producer-owned KV state and executes ordinary
causal masks. A caller-provided 4-D attention mask or `mm_token_type_ids` with
an existing cache is rejected rather than guessed, because the authoritative
runtime treats a prepared 4-D mask as an alternate mask contract.

`executeReferenceF32WithPreparedPrelude` is a narrow executor seam for this
contract. It requires every declared text-prelude output (`hidden_states_0`,
`ple_inputs`, and all named PLE intermediates) before it will execute a text
layer, so a composite caller cannot skip an embedding/PLE assignment and let a
default leak in.

## Validation

`test/gemma4-composite.test.ts` builds a registered dense miniature package
with image, video and audio placeholder IDs plus both full and sliding text
layers. It executes all three towers, checks PAD substitution, named handoff
to the text core, finite logits and the fact that the PLE context changes after
soft-token scatter. It additionally asserts block IDs, a future same-image
token admitted only by sliding attention, a future text token rejected by both
masks, and a two-token cached decode whose two KV producers grow from prefill
length four to length six. It rejects placeholder cardinality mismatches,
partial audio inputs, malformed type-ID shapes, ambiguous caller 4-D masks and
attempts to reuse `mm_token_type_ids` during cached decode.

The synthetic scalar F32 check is control-flow evidence only. It does not
compare BF16 E4B values to PyTorch, and it does not permit the checkpoint
marker to be created.

After compiling, the following header-only check against the pinned local E4B
catalog constructed the registered composite program without materializing its
15 GB BF16 payload:

```bash
node --input-type=module -e 'import { openCatalog } from "./dist/src/catalog.js"; import { buildGemma4CompositeProgram } from "./dist/src/gemma4-composite.js"; const opened = await openCatalog("./gemma-4-E4B-dense", false); try { const p = buildGemma4CompositeProgram(opened.catalog, { outputRows: 1, inputTerms: 1, includeWeights: false }); console.log(JSON.stringify({ kind: p.kind, textLayers: p.contract.text.layers, visionLayers: p.contract.modalities.visionTower.layers, audioLayers: p.contract.modalities.audioTower.layers, compositeAssignments: p.assignments.length, textPrelude: p.textProgram.prelude.map((x) => x.output) })); } finally { await opened.close(); }'
# {"kind":"gemma4-composite-prelude","textLayers":42,"visionLayers":16,"audioLayers":12,"compositeAssignments":18,"textPrelude":["hidden_states_0","ple_token_identity","ple_context_packed","ple_context_scaled","ple_context_reshaped","ple_context_normalized","ple_combined","ple_inputs"]}
```

## Remaining semantic boundary

The E4B configuration enables `use_bidirectional_attention="vision"` for
multimodal calls. Its source-defined mask and cached transition are now
lowered and scalar-executed, but this remains a miniature F32 semantic proof,
not an E4B numerical claim. Dense BF16 casts, literal serializer support for
every composite assignment and an authoritative trace remain separate required
gates.
