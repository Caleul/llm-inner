# Gemma 4 E4B expanded calculation order — 2026-07-17

This is candidate evidence for complete composite navigation. It does not
accept the dense-lossless checkpoint: authoritative BF16 forward/generation
replay remains approximate.

## Corrected execution boundary

The literal navigator now expands shared subprogram definitions at their real
composite call sites. Image and video receive distinct vision instances, audio
is inserted before its outer scatter, and prepared text begins with layer
operations instead of incorrectly replaying the standalone token/PLE prelude.
Video pixel and position flattening are separate named assignments, so the
video position table has a real predecessor rather than an implicit value.

The implementation rejects duplicate instance IDs, duplicate instantiated
outputs, and any linked producer whose ordinal is not lower than its consumer.
Scalar views keep the unique instance ID while reporting the reusable
`definitionId` and `invocationId`.

## Real source-removed evidence

The dense artifact was regenerated from `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, then the checkpoint directory was
renamed away under a shell exit trap while this command ran:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --list-operations \
  --operation composite_video_features/vision_patch_projection \
  --output-coordinate 0,0,0 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop41-expanded-navigation.json
```

Observed result:

- artifact SHA-256:
  `73adbf5b950a45509c565e033282c3d0d1039eaff16a5e3ee9d6d1c0c5dcb91e`;
- artifact bytes: `21,326,374,423`;
- 2,130 embedded constants;
- source/literal comparison covered all `15,992,314,836` storage bytes and
  produced the same sorted-storage SHA-256 on both sides:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- 2,549 unique operation IDs and 2,549 unique outputs;
- 394 image-vision, 394 video-vision, 522 audio and 1,221 prepared-text
  instantiated operations;
- first operation `composite_placeholder_masks`, final operation
  `final_logit_softcap`;
- no `token_embedding` in the prepared composite execution;
- every linked producer ordinal preceded its consumer;
- video pixels came from `composite_video_pixel_flatten`, video positions came
  from `composite_video_position_flatten`, and the final video projection fed
  `composite_video_scatter`;
- the selected real vision projection expanded all 768 terms, decoded 768 BF16
  learned values, reported `complete: true` and `omittedTerms: 0`, and contained
  no `weight[...]` reference;
- first decoded weight: BF16 `0xbd37` = `-0.044677734375`;
- last decoded weight: BF16 `0x3d34` = `0.0439453125`;
- `sourceCheckpointAccessed: false`;
- report bytes: `2,859,321` and report SHA-256
  `3baaa0f3c5a30d731d49e4b6d03d993c7ec9355b34565775f8a2febcf0b21f85`.

The source directory was restored by the trap and was present after the
command. This evidence proves corrected ordered navigation and literal learned
value substitution; it does not improve or certify the native reduction
profiles used by the source-removed text replay.
