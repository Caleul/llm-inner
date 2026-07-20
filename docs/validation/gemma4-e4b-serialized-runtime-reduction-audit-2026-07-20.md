# Gemma 4 E4B: serialized native-reduction product audit — 2026-07-20

## Completed boundary

Schema v42 embedded the exact scalar AST and coordinate graph for every Gemma
4 assignment, but the source-removed native-BMM product auditor still rebuilt
operand addresses with a five-case TypeScript switch over operation classes.
It also read fixed head/chunk/context dimensions from embedded tower metadata
instead of the reduction program being audited. That duplicated artifact
semantics in the reader at the last boundary before Apple Accelerate.

Schema v43 removes that parallel implementation:

- the auditor binds the requested coordinate to serialized
  `outputCoordinates`;
- it executes dependency-ordered local expressions from `statementPrograms`;
- it locates exactly one serialized `REDUCE`, its named reduction index and
  conditional padding predicates;
- it requires exactly one indexed access to each of the two `orderedInputs`;
- it renders every product address from those AST nodes; and
- it obtains completeness only from the serialized reduction domain.

Vision score (`64`), audio content score (`128`), audio position score (`128`)
and audio value (`24`) now carry constant architecture extents in their
reduction domains. Vision value keeps its input-shaped key-patch extent and is
therefore windowed unless a runtime shape is supplied. The implementation is
generic over the scalar language and reduction contract; no layer, invocation,
shape or tensor name is selected in the renderer.

A regression mutates `program.visionProgram.tower.headDim` after the artifact
has been opened and proves the product audit is unchanged. This would have
changed the schema-v42 reader result. Malformed coordinate rank, constant
output bounds, missing/duplicate input accesses, multiple reductions,
non-scalar preludes and divergent reduction indices fail closed.

## Real artifact regeneration and source audit

The artifact was regenerated from `google/gemma-4-E4B` at immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a`:

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
  --output /private/tmp/gemma4-v43-source-audit.json \
  --verify-gemma4-payloads
```

Result:

- schema: `43`;
- artifact bytes: `21,409,277,259`;
- artifact SHA-256:
  `8dbfd9e22126b5286d94f6a2906cfe31b392e2056ef80c7559b97c51a426461e`;
- constants / storage decoders: `2,130` / `2,130`;
- embedded and source-compared learned bytes: `15,992,314,836`;
- source and literal storage SHA-256:
  `e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`;
- immutable source identity: 6 files / `16,024,773,810` bytes;
- integrity sections: `24`;
- integrity root:
  `20ec2149a01250b350c174336627c33ee8c28924c5a2b1af7469a524b2c34c08`;
- unresolved native reductions: `100`; and
- gate: `blocked-on-runtime-reduction` / `exactReplayClaim=forbidden`.

The source-audit report SHA-256 is
`2c0ddc8564c705de239d6744707e8186cdfe9d51f5a01356912adb6df3fa7fd6`.

## Source-removed integrity and product programs

`./gemma-4-E4B-dense` was moved under `/private/tmp` with an
`EXIT`/`INT`/`TERM` restore trap. While the declared source path was physically
absent, the reader recomputed the 24-section manifest and rehashed all 2,130
embedded payloads. It recorded `sourceCheckpointAccessed=false`, the same
15,992,314,836 learned bytes and the same literal-storage digest. The
43,216,083-byte report SHA-256 is
`a023c6a0dea4ec9e46520fddbc7bd103e0b710703477688818052eadbaad71f8`.

One operation from every native-BMM class was then rendered from a single open
artifact:

| class | operation | terms | complete | first operand pair |
| --- | --- | ---: | --- | --- |
| vision score | `vision_layer_0_attention_scores` | 64 | yes | `q_rotated[0,0,0,0] * k_rotated[0,0,0,0]` |
| vision value | `vision_layer_0_attention` | 4 requested | no, explicit window | `attention_weights[0,0,0,0] * v_normalized[0,0,0,0]` |
| audio content score | `audio_layer_0_attention_content_scores` | 128 | yes | `q_scaled[0,0,0] * k_scaled[0,-12,0]` under the serialized bounds predicate |
| audio position score | `audio_layer_0_attention_position_scores` | 128 | yes | `q_scaled[0,0,0] * relative_keys[0,0,0]` under `query_index<sequence_length` |
| audio value | `audio_layer_0_attention` | 24 | yes | `attention_weights[0,0,0,0,0] * v[0,-12,0]` under the serialized key predicate |

The negative audio addresses are intentional audit output: their serialized
predicates select mathematical zero before any tensor read. The local prelude
also exposes `query_index`, `key_index`, `sequence_length`, head, block and
query-in-block calculations from the artifact rather than recreating them in
the reader.

## Source-removed authoritative differential

In the same source-hidden window, the regenerated artifact was compared at
zero absolute and relative tolerance with the pinned authoritative image,
video and audio captures from
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`.

| modality | prefill | generation | generated tokens | step caches | native BMM |
| --- | --- | --- | ---: | ---: | ---: |
| image | 5/5 pass | lossless-within-dtype | 1 | 24/24 | 32 |
| video | 5/5 pass | lossless-within-dtype | 1 | 24/24 | 32 |
| audio | 5/5 pass | lossless-within-dtype | 1 | 24/24 | 36 |

The aggregate records 3/3 prefill passes, 3/3 generation passes,
`firstDivergence=null`, 100 runtime-defined reductions and
`sourceCheckpointAccessed=false`. Its SHA-256 is
`4cb322d67944155a15d54858bb9475751a50f256cf2e79af9490784f0ffc79dd`.
The trap restored the source directory after the suite exited.

## Preserved fidelity boundary

This boundary removes reader-side reconstruction of every address and
predicate entering the five BMM classes. It does not invent Apple Accelerate
SGEMM's unpublished product-rounding or accumulation tree. The artifact still
contains 32 image, 32 video and 36 audio reductions marked
`fail-closed-runtime-reduction`; `exactReplayClaim` remains `forbidden`.

Fresh repository validation after implementation and regeneration:

```text
npm run typecheck  # exit 0
LLM_INNER_TRANSFORMERS_PYTHON=/private/tmp/llm-inner-transformers-missing-python npm test
                   # 250 pass, 0 fail, 3 optional integration skips
git diff --check   # exit 0
```

The skipped captures are the explicitly optional Transformers/PyTorch Llama,
Qwen 2 and Gemma 2 integrations. The Gemma 4 literal artifact, source-removed
payload replay, serialized BMM product audits and composite differential all
ran successfully.
