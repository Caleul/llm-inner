# Gemma 4 E4B embedded F32 transcendental programs

This is candidate artifact-format evidence from loop 74. It does not accept the
dense-lossless checkpoint: 100 instantiated native Apple Accelerate BMM
operations retain their explicit unpublished scalar schedules, so the complete
end-to-end calculation remains fail-closed.

## Independent starting review

Loop 74 independently inspected commit
`2180e6e1b60e2cb56f8381dc63902e923a7ccb52` and reran `npm run typecheck` plus
all 233 tests before implementation. The schema-v17 eager audio mask contract
and its exact recorded composite boundary were accepted as repository evidence;
the Gemma 4 checkpoint was not accepted because the artifact still contained
100 runtime-defined BMM reductions.

## Boundary closed

The scalar formulas previously named `SLEEF_EXP_F32`, `SLEEF_SIN_F32`,
`SLEEF_COS_F32` and `SLEEF_TANH_F32`, but several vision/audio operation
classes did not carry the implementation behind those names. Schema v18 adds
one class-wide `transcendentalPrograms` contract used by text, vision and audio:

- pinned PyTorch commit `7269437d655783a26cba32aa88195b741ff496aa`
  and bundled SLEEF commit `5a1d179df9cf652951b59010a2d2075372d67f68`;
- 28 named constants with their exact binary32 bits;
- explicit binary32 add/subtract/multiply/divide/FMA, round-to-even, signed-zero,
  bit and two-register pair semantics;
- 24 finite pair, polynomial, scale, `rempi`, large-angle and special-value
  subprograms;
- the 416-entry little-endian F32 `Sleef_rempitabsp` payload, SHA-256
  `9a623b9ff705f726ddb129e4b1c3c0311ac19b86a30667cdd395e6ea98d8c5c5`;
- complete ordered scalar statements and NaN/infinity/signed-zero branches for
  all four referenced kernels.

Formula-language schema v4 points directly to `/transcendentalPrograms`.
Artifact validation rejects any changed program, constant, table, kernel,
undeclared subprogram call, or `SLEEF_*` formula without a matching embedded
program. Scalar views and calculation slices carry both the formula-language
and transcendental contracts, so a source-removed audit does not need an
external SLEEF source tree or binary to interpret those operations.

## Real artifact

The immutable source is `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`, dense unquantized BF16
Safetensors. The final generation command was:

```bash
npm run build
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Result:

- schema: 18;
- bytes: `21,384,503,998`;
- SHA-256: `33211c487077735b8d13f7e0d522aeb1fa641f06115419ad10b989951adffe2e`;
- constants/storage decoders: `2,130` / `2,130`;
- embedded learned bytes: `15,992,314,836`;
- calculation graph: `2,709` instantiated forward assignments.

## Source-removed evidence

The directory `./gemma-4-E4B-dense` was physically moved outside its declared
path under an EXIT restore trap for every command in this section, and restored
afterward.

Embedded payload verification covered all 2,130 constants and all
`15,992,314,836` bytes. It recorded `sourceCheckpointAccessed=false` and the
same concatenated literal-storage SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The report is `/private/tmp/llm-inner-loop74-v18-payloads.json`, 42,922,435
bytes, SHA-256
`3ca9fbb6e3cd3b03071beed68955e70d213beb0b21e8635bedacbd59818407e1`.

A scalar view of
`composite_image_features/vision_layer_0_attention_weights[0,0,0,0]` was
rendered with the source absent. It is itself literal, carries formula-language
schema 4, all four programs, 28 exact constants, 24 subprograms and the verified
416-entry `rempi` table. The report is
`/private/tmp/llm-inner-loop74-v18-transcendental-scalar.json`, 42,961,537
bytes, SHA-256
`5898265546fa2b4d5350013a139cfca9848b2f3f6c1e608689c318c908b5e6ec`.

The one-token end-to-end view also recorded `sourceCheckpointAccessed=false`,
2,709 forward assignments, 12 instantiated generation-control assignments,
2,076 reachable learned constants and 54 declared runtime-unreachable shared-KV
locals, covering all 2,130 embedded constants. Its forward slice carries the
same four transcendental programs. The report is
`/private/tmp/llm-inner-loop74-v18-end-to-end.json`, 63,067,414 bytes,
SHA-256 `b411aae5df3c7547f53500c3120434e759e98d203737e0d3377637960529c1f3`.

Finally, the pinned eager full-composite audio trace was replayed from the final
schema-v18 artifact at zero absolute and relative tolerance with
`--allow-unverified-fidelity`. Prefill text embedding, terminal audio feature,
audio scatter, PLE combine and logits were all exact. Generation selected token
184 at position 2 exactly; selection and terminal logits plus every cache entry
across 24 layers had zero error. Prefill and generation were both
`lossless-within-dtype`, with no first divergence. The report is
`/private/tmp/llm-inner-loop74-v18-composite-audio.json`, SHA-256
`7f0564e86b82261f0ce2a65b7735f81f9704e71b4aac83632418348371d4c69f`.

## Validation and preserved limit

`npm run typecheck` passed. `npm test` passed 234 tests with zero failures.
Regressions execute representative exp/sin/cos/tanh results, validate the
embedded table and exact constant bits, reject a changed scalar statement,
reject a changed serialized kernel, and prove source-removed scalar/slice
ownership of the contract.

The end-to-end view correctly remains `fail-closed-runtime-reduction`: the 64
vision and 36 audio native BMM instances still have no published scalar
reduction tree. This change does not assign an inferred schedule to them, and
`.agent-loop/checkpoints/gemma4-dense-lossless/` was not created.
