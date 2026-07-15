# Public Llama package replay — 2026-07-15

## Scope and fidelity class

This is the first replay against an immutable, publicly published checkpoint
rather than a repository-authored fixture. It validates the full native
Safetensors catalog → explicit Llama adapter → IR → range materialization →
scalar F32 executor path against an independent MLX 0.32 kernel execution.

The result is **numerically equivalent within the declared tolerance**, not
bitwise-equivalent and not a capture from the Hugging Face `transformers`
model implementation. The MLX helper executes the already validated Llama IR
using MLX kernels; it is an independent numerical implementation, but it does
not independently establish model-family semantics. A version-pinned
Transformers instrumentation adapter remains required for that stronger
evidence class.

## Immutable source

- Model: `hf-internal-testing/tiny-random-LlamaForCausalLM`
- Hugging Face revision: `9fb191250dd56d0ba7ec9785a025ed29c03d5998`
- Container: single-file dense F32 Safetensors; quantization: none.
- Architecture: Llama, 2 layers, hidden size 16, 4 Q/KV heads, head dim 4,
  vocab size 32,000.
- `config.json` SHA-256: `5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382`
- `model.safetensors` SHA-256: `49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992`
- IR fingerprint: `15371913523c529628115e50115dc49864a82e5c86740ec04027a7915ed9632c`

## Runtime and policy

- Reference: MLX 0.32.0 / NumPy 2.5.1 / Python 3.13.12 on Apple Silicon.
- Candidate: `llm-inner scalar IEEE-754 F32`.
- Both use F32; tolerance is absolute `1e-5`, relative `1e-4`.
- Host: macOS 26.5.2 (`darwin arm64`), Node 25.9.0.

## Execution trace

Input IDs were `[1, 2, 3]` at absolute positions `[0, 1, 2]`.

- All 41 operation outputs and both post-RoPE KV-cache layers passed.
- Terminal logits (`[1, 3, 32000]`): maximum absolute error
  `5.960464477539063e-8`; maximum relative error `0.0014219841262707522`;
  cosine `0.9999999999999585`; top-10 overlap `1`; argmax agreement `true`.
- Across every operation, cache, and logit comparison: maximum absolute error
  `2.384185791015625e-7`; no first divergent operation.

## Greedy generation trace

The `[1, 2, 3]` prompt with `maxNewTokens=3` emitted `[18568, 1727, 8705]`
at positions `[3, 4, 5]` in both runtimes.

- All three selection-logit comparisons and every post-decode KV snapshot
  passed.
- Terminal logits (`[1, 1, 32000]`): maximum absolute error
  `5.960464477539063e-8`; maximum relative error `0.0007218479307025986`;
  cosine `0.9999999999999682`; top-10 overlap `1`; argmax agreement `true`.
- Across selection logits, step caches, terminal logits, and final cache:
  maximum absolute error `8.940696716308594e-8`; maximum relative error
  `0.009124087591240875`; no first divergence.

## Reproduction

Download from the pinned revision; do not substitute a moving branch name.
Trace and report paths remain outside the repository because traces serialize
model activations.

```bash
MODEL=/tmp/llm-inner-tiny-llama
EVIDENCE=/tmp/llm-inner-tiny-llama-evidence
REVISION=9fb191250dd56d0ba7ec9785a025ed29c03d5998
PYTHON=/tmp/llm-inner-mlx-0.32/bin/python

mkdir -p "$MODEL" "$EVIDENCE"
curl --fail --location -o "$MODEL/config.json" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$REVISION/config.json"
curl --fail --location -o "$MODEL/model.safetensors" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$REVISION/model.safetensors"
shasum -a 256 "$MODEL/config.json" "$MODEL/model.safetensors"

npm run capture:mlx-trace -- --source "$MODEL" --output "$EVIDENCE/execution.json" \
  --input-tokens 1,2,3 --python "$PYTHON" \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM --revision "$REVISION"
npm run compare:trace -- --source "$MODEL" --trace "$EVIDENCE/execution.json" \
  --report "$EVIDENCE/execution-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4

npm run capture:mlx-trace -- --source "$MODEL" --output "$EVIDENCE/generation.json" \
  --input-tokens 1,2,3 --max-new-tokens 3 --python "$PYTHON" \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM --revision "$REVISION"
npm run compare:generation-trace -- --source "$MODEL" --trace "$EVIDENCE/generation.json" \
  --report "$EVIDENCE/generation-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4
```

The MLX environment was provisioned with:

```bash
uv venv --python /opt/homebrew/bin/python3.13 /tmp/llm-inner-mlx-0.32
uv pip install --python /tmp/llm-inner-mlx-0.32/bin/python 'mlx==0.32.0' 'numpy>=2.0,<3'
```
