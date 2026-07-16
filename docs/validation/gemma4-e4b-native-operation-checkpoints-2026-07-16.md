# Gemma 4 E4B native operation checkpoints — 2026-07-16

This diagnostic evidence localizes the numerical-fidelity gap for the immutable
`google/gemma-4-E4B` dense BF16 package. It is a complete trace of the
declared `Gemma4Text` assignments, not a Gemma 4 checkpoint claim: vision,
video, audio, and source-removed generation remain outside this trace.

## Evidence boundary

- Hub revision: `411aa17b749aa952df1359d2dcea73917a544d9a`
- Source tensor: `model.safetensors`, 15,992,595,884 bytes, SHA-256
  `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651`
- Native runtime: PyTorch 2.12.1 / Transformers 5.5.0,
  `Gemma4ForConditionalGeneration`, eager BF16
- Candidate: indexed `gemma4-e4b-dense.literal.json`, executed only through
  declared literal payload ranges with a 16 MiB read window
- Prompt: token ID `2`, absolute position `0`

The pinned helper instruments the actual eager-BF16 forward and records all
1,229 declared assignments: PLE scale/reshape/combine, head reshapes,
Q/K RoPE, complete attention context, residuals, GELU, gated MLP, per-layer
embedding residuals, scalar boundaries, final norm, tied LM head, and softcap.
It first runs an uninstrumented native forward and rejects the capture unless
the instrumented logits and producer-owned KV cache are bitwise identical.
The TypeScript boundary then rejects missing, duplicate, output-drifting, or
unexpected assignment IDs before writing the trace.

## Reproduction

Capture while the immutable source package is present:

```bash
npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-e4b-full-operation-trace.json \
  --input-tokens 2 --python ./venv/bin/python \
  --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a
```

Then remove the complete source directory for the candidate process and run:

```bash
mv ./gemma-4-E4B-dense ./.gemma-4-E4B-dense-source-unavailable
node dist/src/gemma4-paged-text-operation-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-e4b-full-operation-trace.json \
  --report /tmp/gemma4-e4b-full-operation-report.json \
  --max-read-mib 16 --top-k 10 \
  --assert-source-unavailable ./gemma-4-E4B-dense
mv ./.gemma-4-E4B-dense-source-unavailable ./gemma-4-E4B-dense
```

`sourceCheckpointAccessed` is recorded as `false` in the report.  The capture
is integrity-bound to `config.json` and the contributing Safetensors file and
also rejects an IR fingerprint mismatch.

## Result

The previous earliest mismatch was `token_embedding`: the literal F32 path
used `sqrt(2560)` directly, while
`Gemma4TextScaledWordEmbedding` first casts that scale to BF16 and then casts
the multiplied embedding result to BF16.  The literal adapter now records the
widened BF16 scale (`50.5`) and its paged embedding kernel applies the declared
BF16 result cast.

The regenerated artifact SHA-256 is
`d74db021e785b081f5b84714368a23a8d3baf5da40048d971d124580190c491d`.
Its header declares operation-level accumulation because its program records
F32 products plus an ordered F64 scalar accumulator and BF16 result cast for
Gemma4Text linear/RMSNorm assignments; all other declared operations retain
their explicit F32 policy. The paged executor applies these declared boundaries
before the next named assignment or cache transition.

The source-removed probe covers every one of 1,229 declared assignments and
has no missing or unexpected trace IDs. It passes 69 assignments and one
producer-owned KV cache exactly. `ple_context_projection`,
`ple_context_scale`, `ple_context_reshape`, and `ple_context_norm` now match
exactly. The first divergent assignment is `layer_0_gate_proj`; the 10,240 by
2,560 projection differs in two BF16 coordinates under the portable ordered
F64 profile. A direct probe found that an interleaved 32-lane F32 tree matches
that one projection, while the 10,752-row PLE projection requires the ordered
F64 profile. Because the authoritative runtime does not expose a stable,
documented shape-to-reduction-tree contract, the artifact does not invent this
shape heuristic. The remaining 1,160 assignments and 23 producer KV caches
diverge at zero tolerance. This is therefore a complete **approximate**
operation comparison, not an incomplete probe: the observed evidence
identifies the first unresolved numerical boundary without claiming
lossless-within-dtype or the Gemma 4 checkpoint.
