# Gemma 4 E4B: integrity-bound fidelity gate — 2026-07-19

## Completed boundary

Schema v41 preserved the unresolved Apple Accelerate boundary in formulas,
numeric policies and differential reports, but the artifact did not contain a
single machine-readable completion gate derived from its canonical calculation
graph. Schema v42 closes that overclaiming gap:

- `fidelityGate` enumerates every assignment whose reduction order is
  `runtime-defined`;
- each entry records ordinal, operation/definition/invocation IDs, scope,
  operation class, output and JSON Pointers to the scalar reduction and output
  coordinate;
- construction fails unless scalar and output policies both declare
  runtime-defined accumulation, the scalar calculation is fail-closed, the
  reduction has exactly one domain and its operation class belongs to the
  authoritative Gemma 4 contract;
- the reader rebuilds the gate from `calculationGraph` and
  `authoritativeExecution` and rejects any divergence; and
- the integrity manifest commits the gate as its 24th ordered section.

A non-empty list requires `status=blocked-on-runtime-reduction` and
`exactReplayClaim=forbidden`. An empty list would mean only
`eligible-for-independent-certification` / `not-certified`: an implementation
session cannot use the gate to certify its own changes.

The operation classifier is shared by the gate and the scalar operand-product
auditor. The implementation is generic over assignments and authoritative
operation classes; it contains no layer, invocation, tensor, shape or ordinal
special case.

## Real artifact regeneration and source audit

The dense unquantized artifact was regenerated from `google/gemma-4-E4B` at
immutable revision `411aa17b749aa952df1359d2dcea73917a544d9a`:

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1

npm run audit:literal -- \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-v42-source-audit.json \
  --verify-gemma4-payloads
```

Result:

- schema: `42`;
- artifact bytes: `21,409,281,213`;
- artifact SHA-256:
  `b5b826dae8a89001c437341c82a4bf4c828d79a065bdeb66be40667ffb90a72f`;
- constants / payload commitments: `2,130` / `2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- integrity sections: `24`;
- integrity root:
  `ffe0578f74c82226d9eb6fb71f2035580fe8dcf52cf20cf84b685118134f5910`;
- unresolved native reductions: `100`; and
- gate state: `blocked-on-runtime-reduction` /
  `exactReplayClaim=forbidden`.

The source-audit report has SHA-256
`be5ae7798b357357db972b97f5823728fc6637ee1d46a0841dd332727a51abaf`.

## Full source-removed integrity and scalar audit

`./gemma-4-E4B-dense` was physically moved and restored by an
`EXIT`/`INT`/`TERM` trap. While the declared source path was absent, the reader
recomputed the 24-section manifest and rehashed every embedded learned byte.
It recorded `sourceCheckpointAccessed=false`, 2,130 constants and the same
`15,992,314,836`-byte literal-storage digest. The 43,028,819-byte report has
SHA-256
`09361967909f5f1b8042cac9de5c78e0fb67bf873b2886cc27ffe4b27200998c`.

In the same source-removed window, the scalar view for
`layer_0_q_proj[0,0,0]` decoded the first four BF16 learned operands to exact
numeric literals:

```text
-0.006195068359375
 0.0018463134765625
-0.0498046875
 0.033203125
```

It omitted the remaining 2,556 terms by an explicit requested audit window and
contained no unexplained learned-weight placeholder. The 43,118,400-byte
report has SHA-256
`2eff65a1f9a82af35a629119d9fbcb09c52535b35be127c9355e3ff4983a4cab`.

## Source-removed authoritative differential

The suite compared the regenerated artifact against the pinned authoritative
image, video and audio captures from
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`:

- image trace SHA-256:
  `48a7270be75d44ae31809ae04c77dd419b22937bd3ec5fdc1e4e705f031af68b`;
- video trace SHA-256:
  `10618977609781f65a7c325c7cbff3018cfb9b2ad280cd1a9bdde898bdaa7e9d`;
- audio trace SHA-256:
  `d07227accbccdd911f2e5d66b0152e12f8d908ad480503e45b55d8344fac0610`.

With the checkpoint path physically absent, the zero-absolute and
zero-relative-tolerance comparison produced:

| modality | prefill | generation | token / position | step caches | runtime-defined BMM |
| --- | --- | --- | --- | --- | ---: |
| image | 5/5 pass | lossless-within-dtype | 184 / 2 | 24/24 | 32 |
| video | 5/5 pass | lossless-within-dtype | 184 / 2 | 24/24 | 32 |
| audio | 5/5 pass | lossless-within-dtype | 184 / 2 | 24/24 | 36 |

The summary records 3/3 zero-tolerance prefill passes, 3/3 zero-tolerance
generation passes, `firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. The 169,839-byte report has SHA-256
`4cb322d67944155a15d54858bb9475751a50f256cf2e79af9490784f0ffc79dd`.
The source path was restored after the process exited.

## Preserved fail-closed boundary

The artifact now makes its exact-replay prohibition navigable, derived,
integrity-bound and independently revalidated. It does not invent Apple
Accelerate SGEMM's unpublished product-rounding and accumulation tree. The
same 32 image, 32 video and 36 audio reductions remain
`fail-closed-runtime-reduction`, and the differential claim remains
`candidate-modality-coverage-with-runtime-defined-reductions`.

Fresh repository validation after implementation, regeneration and the real
source-removed differential:

```text
npm run typecheck  # exit 0
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
                   # 250 pass, 0 fail, 3 optional integration skips
git diff --check   # exit 0
```

The three skipped captures are the repository's explicitly optional
Transformers/PyTorch Llama, Qwen 2 and Gemma 2 integrations; their separately
pinned external test runtime was not present at the configured path. All Gemma
4 artifact, reader, payload, source-removed replay and differential checks
passed.
