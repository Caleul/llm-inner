# Gemma 4 E4B: mandatory whole-program integrity — 2026-07-19

## Completed boundary

Schema v40 carried per-tensor payload digests only on streamed artifacts and
allowed the reader to open an artifact without them. It also did not commit
the non-payload program as one integrity surface. Schema v41 closes that gap:

- every in-memory and streamed artifact requires one SHA-256 commitment for
  every embedded tensor payload;
- `integrityManifest` commits 23 ordered sections, including immutable source
  identity, constant metadata, storage decoders, every executable semantic and
  navigation structure, forward/generation controls and the complete payload
  commitment table;
- each section declares its canonical UTF-8 byte count and SHA-256, while
  `rootSha256` commits the ordered section-name/length/digest sequence;
- the bounded streaming reader recomputes the manifest before returning an
  artifact, without loading Base64 payloads into V8; and
- in-memory replay rehashes actual decoded payload bytes before executing.

The contract is generic over named sections and payloads. It contains no
operation, layer, shape, dtype or tensor-name dispatch. Tests prove that a
structurally valid source-identity mutation and a missing manifest fail closed,
in addition to the existing decoder, numeric-bit, formula and source checks.

## Real artifact regeneration and structural audit

The artifact was regenerated from dense unquantized `google/gemma-4-E4B` at
immutable revision `411aa17b749aa952df1359d2dcea73917a544d9a`:

```bash
node --max-old-space-size=4096 dist/src/cli.js \
  --source ./gemma-4-E4B-dense \
  --output ./artifacts/gemma4-e4b-dense.literal.json \
  --gemma4-composite-literal \
  --model-id google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a \
  --max-features 1 --max-terms 1
```

Result:

- schema: `41`;
- artifact bytes: `21,409,228,758`;
- artifact SHA-256:
  `271147d1db4400e272911d603a31e02796e9793766468bbf6041602473820558`;
- constants / payload commitments: `2,130` / `2,130`;
- embedded learned bytes: `15,992,314,836`;
- integrity sections: `23`;
- integrity root:
  `eb1899dfb59bfe44eb9bc3f6f7e489b0436c2dbf95f76f2e96eb05e5d6f4b1b1`;
- instantiated assignments / statement programs / expression nodes:
  `2,708` / `4,144` / `118,780`; and
- reduction domains: `1,722`, including exactly `100` runtime-defined BMMs.

The streaming-open report is 42,964,315 bytes with SHA-256
`9695c43f7843d100140e6d8eff6cce33b54ddc8243af80b6423e902ecc384886`.

## Full source-removed integrity and scalar audit

`./gemma-4-E4B-dense` was physically moved and restored by an
`EXIT`/`INT`/`TERM` trap. While the declared path was absent, the reader opened
schema v41, recomputed the structural manifest and hashed every embedded
payload:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-v41-source-removed-integrity.json
```

The result recorded `sourceCheckpointAccessed=false`, compared all
`15,992,314,836` learned bytes and reproduced literal-storage SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The report SHA-256 is
`0c3eaee8944e12856b1ead3a4971aa02a2c4083b0fe37666f272189612ea73c1`.

In the same source-removed window, a four-term scalar view for
`layer_0_q_proj[0,0,0]` decoded BF16 storage to exact literals
`-0.006195068359375`, `0.0018463134765625`, `-0.0498046875`, and
`0.033203125`. The report has SHA-256
`214b15ac310d72ecdc03e3bc2880f7150102dfd02d0b1ebdb72e71a9e9b295b1`
and contains no learned-weight placeholder.

## Fresh authoritative differential

Fresh image, video and audio traces were captured from
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`
with `torch.inference_mode`, eager attention, the immutable revision above and
one greedy token. Their SHA-256 values are:

- image: `48a7270be75d44ae31809ae04c77dd419b22937bd3ec5fdc1e4e705f031af68b`;
- video: `10618977609781f65a7c325c7cbff3018cfb9b2ad280cd1a9bdde898bdaa7e9d`;
- audio: `d07227accbccdd911f2e5d66b0152e12f8d908ad480503e45b55d8344fac0610`.

With the checkpoint path physically absent again, the zero-absolute and
zero-relative-tolerance modality suite produced:

| modality | prefill | generation | token / position | step + terminal caches | runtime-defined BMM |
| --- | --- | --- | --- | --- | ---: |
| image | 5/5 pass | lossless-within-dtype | 184 / 2 | 24/24 + 24/24 | 32 |
| video | 5/5 pass | lossless-within-dtype | 184 / 2 | 24/24 + 24/24 | 32 |
| audio | 5/5 pass | lossless-within-dtype | 184 / 2 | 24/24 + 24/24 | 36 |

The summary records three zero-tolerance prefill passes, three zero-tolerance
generation passes, `firstDivergence=null`, and
`sourceCheckpointAccessed=false`. The 169,839-byte report SHA-256 is
`4cb322d67944155a15d54858bb9475751a50f256cf2e79af9490784f0ffc79dd`.

## Preserved fail-closed boundary

This integrity change prevents a payload, metadata or executable-program
mutation from being silently accepted. It does not invent Apple Accelerate
SGEMM's unpublished product-rounding and accumulation tree. The same 32 image,
32 video and 36 audio BMMs remain `fail-closed-runtime-reduction`, so the
differential claim remains
`candidate-modality-coverage-with-runtime-defined-reductions` and this session
does not certify the complete objective.

Fresh repository validation after implementation, regeneration and the real
source-removed differential:

```text
npm run typecheck  # exit 0
npm test           # 253/253, exit 0
git diff --check   # exit 0
```
