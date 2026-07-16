# Public Llama native Transformers replay — 2026-07-16

## Scope and fidelity class

This is the first package-level differential replay whose reference executes
the published Hugging Face `LlamaForCausalLM` implementation rather than an IR
lowered by this repository. The reference adapter is version-pinned,
fail-closed, uses PyTorch F32 eager attention, and proves that its hooks do not
alter logits or canonical `DynamicCache` tensors before it serializes the
trace.

The candidate is the native Safetensors catalog → explicit Llama adapter → IR
→ range materialization → scalar F32 executor path. The result is
**numerically equivalent within the declared tolerance**, not bitwise
equivalent. It is evidence for this exact dense-F32 Llama contract only; it
does not establish equivalence for another model family, Transformers version,
storage format, or quantization scheme.

## Immutable source and runtime

- Model: `hf-internal-testing/tiny-random-LlamaForCausalLM`
- Hugging Face revision: `9fb191250dd56d0ba7ec9785a025ed29c03d5998`
- Container: single-file dense F32 Safetensors; quantization: none.
- Architecture: Llama, 2 layers, hidden size 16, 4 Q/KV heads, head dim 4,
  vocab size 32,000, SiLU, default RoPE, no projection biases.
- `config.json` SHA-256: `5b52782dd9a2b8a4eb87f321448a61cd56cf0764493a1ff73f327259e14ab382`
- `model.safetensors` SHA-256: `49c20f32c6c597480fcaec5df2f86c645eabea765cbea1e67886dbae45e5c992`
- IR fingerprint: `15371913523c529628115e50115dc49864a82e5c86740ec04027a7915ed9632c`
- Reference: PyTorch `2.7.1` / Transformers `4.57.1`,
  `LlamaForCausalLM` F32 eager native capture.
- Candidate: `llm-inner scalar IEEE-754 F32`.
- Tolerance: absolute `1e-5`, relative `1e-4`.

## Execution trace

Input IDs `[1, 2, 3]` ran at absolute positions `[0, 1, 2]`.

- All 41 stable IR operation outputs and both canonical post-RoPE KV-cache
  layers passed; there was no first divergent operation.
- Terminal logits `[1, 3, 32000]`: maximum absolute error
  `5.960464477539063e-8`, maximum relative error
  `0.0042419708997900735`, cosine `0.9999999999999485`, top-10 overlap `1`,
  and argmax agreement `true`.
- Across all operations, caches, and logits: maximum absolute error
  `3.5762786865234375e-7`, maximum relative error
  `0.0042419708997900735`, and minimum cosine
  `0.9999999999999439`.

## Greedy generation trace

The same prompt with `maxNewTokens=3` emitted `[18568, 1727, 8705]` at
positions `[3, 4, 5]` in both runtimes.

- Every selection-logit comparison, each of the three post-decode KV snapshots,
  terminal logits, and final two-layer KV cache passed; no first divergence.
- Terminal logits `[1, 1, 32000]`: maximum absolute error
  `5.960464477539063e-8`, maximum relative error
  `0.0008673736580256611`, cosine `0.9999999999999738`, top-10 overlap `1`,
  and argmax agreement `true`.
- Across selection logits, snapshots, terminal logits, and final cache:
  maximum absolute error `5.960464477539063e-8`, maximum relative error
  `0.0042419708997900735`, and minimum cosine
  `0.9999999999999485`.

## Reproduction

Download the pinned revision; do not substitute a moving branch name. Trace
and report files remain outside the repository because they serialize model
activations.

```bash
MODEL=/tmp/llm-inner-tiny-llama
EVIDENCE=/tmp/llm-inner-transformers-llama-evidence
REVISION=9fb191250dd56d0ba7ec9785a025ed29c03d5998
PYTHON=/tmp/llm-inner-transformers/bin/python

mkdir -p "$MODEL" "$EVIDENCE"
curl --fail --location -o "$MODEL/config.json" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$REVISION/config.json"
curl --fail --location -o "$MODEL/model.safetensors" \
  "https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/$REVISION/model.safetensors"
shasum -a 256 "$MODEL/config.json" "$MODEL/model.safetensors"

npm run capture:transformers-trace -- --source "$MODEL" --output "$EVIDENCE/execution.json" \
  --input-tokens 1,2,3 --position-ids 0,1,2 --python "$PYTHON" \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM --revision "$REVISION"
npm run compare:trace -- --source "$MODEL" --trace "$EVIDENCE/execution.json" \
  --report "$EVIDENCE/execution-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4

npm run capture:transformers-trace -- --source "$MODEL" --output "$EVIDENCE/generation.json" \
  --input-tokens 1,2,3 --position-ids 0,1,2 --max-new-tokens 3 --python "$PYTHON" \
  --model hf-internal-testing/tiny-random-LlamaForCausalLM --revision "$REVISION"
npm run compare:generation-trace -- --source "$MODEL" --trace "$EVIDENCE/generation.json" \
  --report "$EVIDENCE/generation-report.json" --max-absolute-error 1e-5 --max-relative-error 1e-4
```

The pinned runtime was provisioned outside the repository with Python 3.13:

```bash
uv venv --python /opt/homebrew/bin/python3.13 /tmp/llm-inner-transformers
uv pip install --python /tmp/llm-inner-transformers/bin/python \
  'torch==2.7.1' 'transformers==4.57.1' 'safetensors==0.8.0' 'numpy==2.5.1'
```
