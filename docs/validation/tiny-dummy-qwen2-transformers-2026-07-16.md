# Public Qwen 2 native Transformers replay — 2026-07-16

## Scope and fidelity class

This package-level differential replay executes the published Hugging Face
`Qwen2ForCausalLM` eager implementation as the reference; it does not use the
candidate IR to calculate the reference trace. The adapter is pinned to
PyTorch `2.7.1` and Transformers `4.57.1`, verifies hook transparency against
unhooked logits and canonical `DynamicCache`, and rejects a package outside
the dense-F32 Qwen 2 contract.

The candidate path is Safetensors catalog → explicit Qwen 2 adapter → IR →
range materialization → scalar F32 executor. It is **numerically equivalent
within the declared tolerance**, not bitwise equivalent, and is evidence only
for this exact model revision, runtime contract, and dense Safetensors storage.

## Immutable source and runtime

- Model: `fxmarty/tiny-dummy-qwen2`
- Hugging Face revision: `d2d85e2f738eb7f6241e052696d4636d8ca27ece`
- Container: single-file dense F32 Safetensors; quantization: none.
- Architecture: Qwen 2, 2 layers, hidden size 8, 4 query / 2 KV heads, derived
  head dim 2, vocabulary 151,936, SiLU, default RoPE with theta 1,000,000.
- Q/K/V projections carry biases; eager Qwen 2 fixes `o_proj` to bias-free.
- `config.json` SHA-256: `f44614f3f6850a2cbd9b7977c0130ccbe321896371af985d0c7de276e383b030`
- `model.safetensors` SHA-256: `33a97d50588456f0a7c6c490010c9c60800a4f9fe2e69e420d40539346ca0c26`
- Reference: PyTorch `2.7.1` / Transformers `4.57.1`, Qwen2ForCausalLM F32 eager native capture.
- Candidate: `llm-inner scalar IEEE-754 F32`; tolerance: absolute `1e-5`, relative `1e-4`.

## Execution and greedy generation

Input IDs `[1, 2, 3]` at absolute positions `[0, 1, 2]` passed all 41 stable
IR operations, both post-RoPE KV layers, and terminal logits with no first
divergent operation. The largest absolute error across the execution report
was `2.384185791015625e-7`; terminal logits had maximum absolute error
`4.470348358154297e-8`, cosine `0.9999999999998952`, top-10 overlap `1`, and
argmax agreement `true`.

Three greedy decode steps emitted `[64909, 64909, 64909]` at positions
`[3, 4, 5]` in both runtimes. Every selection-logit comparison, post-decode KV
snapshot, terminal logits, and final two-layer cache passed with no first
divergence. The largest generation-report absolute error was
`5.960464477539063e-8`; relative errors near zero are reported but do not
override the declared absolute tolerance, cosine, top-k, and argmax checks.

## Reproduction

```bash
MODEL=/tmp/llm-inner-tiny-qwen2
EVIDENCE=/tmp/llm-inner-qwen2-evidence
REVISION=d2d85e2f738eb7f6241e052696d4636d8ca27ece
PYTHON=/tmp/llm-inner-transformers/bin/python

"$PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download(\
  repo_id='fxmarty/tiny-dummy-qwen2', revision='$REVISION', local_dir='$MODEL', \
  allow_patterns=['config.json', 'model.safetensors'])"
shasum -a 256 "$MODEL/config.json" "$MODEL/model.safetensors"

npm run capture:transformers-trace -- --adapter qwen2 --source "$MODEL" --output "$EVIDENCE/execution.json" \
  --input-tokens 1,2,3 --position-ids 0,1,2 --python "$PYTHON" \
  --model fxmarty/tiny-dummy-qwen2 --revision "$REVISION"
npm run compare:trace -- --source "$MODEL" --trace "$EVIDENCE/execution.json" \
  --report "$EVIDENCE/execution-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4

npm run capture:transformers-trace -- --adapter qwen2 --source "$MODEL" --output "$EVIDENCE/generation.json" \
  --input-tokens 1,2,3 --position-ids 0,1,2 --max-new-tokens 3 --python "$PYTHON" \
  --model fxmarty/tiny-dummy-qwen2 --revision "$REVISION"
npm run compare:generation-trace -- --source "$MODEL" --trace "$EVIDENCE/generation.json" \
  --report "$EVIDENCE/generation-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4
```
