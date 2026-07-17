# Gemma 4 E4B current source-removed text boundary — 2026-07-17

This independent review first repeated the loop-33 `layer_0_down_proj`
source-removed profile replay, then captured a fresh authoritative text trace
and replayed the current literal artifact with the checkpoint directory absent.
It accepts neither a native down-projection schedule nor the Gemma 4
dense-lossless checkpoint.

## Immutable inputs

- Source: `google/gemma-4-E4B`, revision
  `411aa17b749aa952df1359d2dcea73917a544d9a`.
- Dense BF16 `model.safetensors` SHA-256:
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`.
- `config.json` SHA-256:
  `f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`.
- Literal artifact: `artifacts/gemma4-e4b-dense.literal.json`, SHA-256
  `7a778d31269187f71a85458cf97b6b91f8c28eacd3fca78e43463bf833db7f0c`.
- Reference runtime: PyTorch `2.12.1`, Transformers `5.5.0`, eager CPU BF16
  `Gemma4ForConditionalGeneration`.
- Candidate runtime: `llm-inner paged Gemma4Text literal F32`, with declared
  BF16 result boundaries and a 16 MiB page-read bound.

## Independent reduction review

With `model.safetensors` temporarily renamed, the six loop-33 down-projection
traces were replayed against the literal artifact with only the six previously
matching profiles. The report at
`/tmp/gemma4-loop34-down-source-removed-selection.json` recorded
`sourceCheckpointAccessed: false`, six traces in three input groups, and
`candidateSelection: { "status": "ambiguous" }`. Every profile still had zero
mismatched BF16 elements, so the result supplies no `profileId` and cannot
alter the ordered-scalar F64 declaration for `layer_0_down_proj`.

## Fresh full-path evidence

The reference was captured from the present checkpoint:

```bash
npm run capture:gemma4-text-trace -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop34-current-text-generation.json \
  --input-tokens 2 --max-new-tokens 1 --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop34-current-operation-checkpoints.json \
  --input-tokens 2 --position-ids 0 --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu
```

For each candidate command, the entire source directory was renamed and an
exit trap restored it only after completion. The commands passed
`--assert-source-unavailable ./gemma-4-E4B-dense`; their reports declare
`sourceCheckpointAccessed: false`:

```bash
npm run compare:gemma4-paged-text-trace -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop34-current-text-generation.json \
  --report /tmp/gemma4-loop34-current-source-removed-generation-report.json \
  --max-read-mib 16 --top-k 10 --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense

npm run compare:gemma4-paged-operation-checkpoints -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop34-current-operation-checkpoints.json \
  --report /tmp/gemma4-loop34-current-source-removed-operation-report.json \
  --max-read-mib 16 --top-k 10 --allow-unverified-fidelity \
  --assert-source-unavailable ./gemma-4-E4B-dense
```

The operation report contains 1,229 declared assignments: 73 pass exactly and
1,156 diverge downstream. `layer_0_gate_proj` and `layer_0_up_proj` both pass
with zero error. The first divergent assignment is
`layer_0_down_proj -> layer_0_mlp_output` (shape `[1, 1, 2560]`, maximum
absolute error `0.001953125`, cosine `0.9999999999888474`), followed by
`layer_0_post_ffn_norm` (maximum absolute error `0.00048828125`). This proves
the current full-path boundary agrees with the deliberately fail-closed down
schedule result; it does not identify that schedule.

The generation report is `approximate` at
`generation:selection-logits-0`. Both implementations greedily emitted token
`184` at position `1`, but selection logits had maximum absolute error
`0.1875`, cosine `0.9999420031132565`, top-10 overlap `1.0`, and argmax
agreement. Terminal logits had maximum absolute error `0.1875`, cosine
`0.9999507089407171`, top-10 overlap `1.0`, and argmax agreement. KV cache
values are not exact. Thus token agreement is recorded as evidence only, never
as a lossless-replay claim.

## Limit

This is source-independent text-only evidence for one CPU-BF16 prompt/decode
path. The native down-projection reduction remains semantically unidentified;
full logits, KV cache, generation fidelity, and multimodal Gemma 4 execution
remain unproven. No checkpoint marker is created.
