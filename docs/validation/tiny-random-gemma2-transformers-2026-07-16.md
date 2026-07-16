# Public Gemma 2 native Transformers replay — 2026-07-16

## Scope and fidelity class

This package-level differential replay executes Hugging Face's published
`Gemma2ForCausalLM` eager implementation as its reference. The candidate is
the independent Safetensors catalog → explicit Gemma 2 adapter → IR → range
materialization → scalar F32 executor path. It is **numerically equivalent
within the declared tolerance**, not bitwise equivalent.

The adapter is fixed to PyTorch `2.7.1` and Transformers `4.57.1`, proves that
instrumentation leaves native logits and the canonical cache unchanged, and
rejects any package outside its dense-F32 semantic contract. That contract
requires the four RMSNorm block, scaled embeddings, explicit local/global
layer types, `query_pre_attn_scalar`, attention and final-logit softcaps, and
the declared/default output-head binding. Transformers 4.57.1 Gemma 2 has no Q/K norm
modules, so a Q/K-normalized package is deliberately outside this adapter.

## Immutable source and runtime

- Model: `hf-internal-testing/tiny-random-Gemma2ForCausalLM`
- Hugging Face revision: `de7c11b6c25d26ddd1bf4324fcf479b61d18e440`
- Container: single-file dense F32 Safetensors; quantization: none.
- Architecture: Gemma 2, one sliding-attention layer, hidden size 32, two
  query/KV heads of dimension 16, GELU-tanh, default RoPE theta 10,000,
  `query_pre_attn_scalar=224`, attention softcap 50 and logit softcap 30.
- The serialized config omits `tie_word_embeddings`; Gemma2Config's
  authoritative registered default is `true`, and the native `lm_head` shares
  `model.embed_tokens.weight`. The Gemma 2 adapter owns this explicit default;
  it is not inferred from a missing tensor.
- `config.json` SHA-256: `e40fe77b1bfa4dfb4b4884f15ea87daa78e68b80ab0d966c0d8dcd29df1896f4`
- `model.safetensors` SHA-256: `a8f34b2cec7460d820591611df87bf0119b7d239fd7dbc6de9e7f6dcc2447798`
- Reference: PyTorch `2.7.1` / Transformers `4.57.1`, Gemma2ForCausalLM F32
  eager native capture. Candidate: `llm-inner scalar IEEE-754 F32`.

## Execution and greedy generation

Input IDs `[1, 2, 3]` at positions `[0, 1, 2]` passed all 25 stable IR
operations, the post-RoPE cache for the sliding-attention layer, and terminal
softcapped logits; no operation diverged. The execution maximum absolute error
was `2.086162567138672e-7`; terminal logits had cosine similarity
`0.9999999999998295`, top-10 overlap `1`, and matching argmax.

Three greedy decode steps emitted `[253745, 63587, 63587]` at positions
`[3, 4, 5]` in both runtimes. Every selection-logit comparison, post-decode KV
snapshot, terminal logits, and final cache passed. The largest observed
generation absolute error was `2.384185791015625e-7`; the terminal-logit
cosine similarity was `0.9999999999998845`, top-10 overlap `1`, and argmax
agreement was true.

## Reproduction

```bash
MODEL=/tmp/llm-inner-tiny-gemma2
EVIDENCE=/tmp/llm-inner-gemma2-evidence
REVISION=de7c11b6c25d26ddd1bf4324fcf479b61d18e440
PYTHON=/tmp/llm-inner-transformers/bin/python

"$PYTHON" -c "from huggingface_hub import snapshot_download; snapshot_download(\
  repo_id='hf-internal-testing/tiny-random-Gemma2ForCausalLM', revision='$REVISION', local_dir='$MODEL', \
  allow_patterns=['config.json', 'model.safetensors'])"
shasum -a 256 "$MODEL/config.json" "$MODEL/model.safetensors"

npm run capture:transformers-trace -- --adapter gemma2 --source "$MODEL" --output "$EVIDENCE/execution.json" \
  --input-tokens 1,2,3 --position-ids 0,1,2 --python "$PYTHON" \
  --model hf-internal-testing/tiny-random-Gemma2ForCausalLM --revision "$REVISION"
npm run compare:trace -- --source "$MODEL" --trace "$EVIDENCE/execution.json" \
  --report "$EVIDENCE/execution-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4

npm run capture:transformers-trace -- --adapter gemma2 --source "$MODEL" --output "$EVIDENCE/generation.json" \
  --input-tokens 1,2,3 --position-ids 0,1,2 --max-new-tokens 3 --python "$PYTHON" \
  --model hf-internal-testing/tiny-random-Gemma2ForCausalLM --revision "$REVISION"
npm run compare:generation-trace -- --source "$MODEL" --trace "$EVIDENCE/generation.json" \
  --report "$EVIDENCE/generation-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4
```
