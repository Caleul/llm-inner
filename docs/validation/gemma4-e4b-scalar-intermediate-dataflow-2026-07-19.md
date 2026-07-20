# Gemma 4 E4B: scalar-intermediate dataflow — 2026-07-19

## Completed boundary

Schema v37 closed every external coordinate binding, but intra-operation
intermediates were still navigable only by reparsing `scalarAssignments`.
Values such as `score[key]`, `maximum`, `exponential[key]`, `angle`,
`paired_feature`, `acc[patch]` and `source_coordinate` had names and execution
order, but no serialized producer, reverse consumers or executable local-index
coordinates.

Schema v38 closes that boundary generically:

- scalar-calculation schema 4 adds one `statementDataflow` entry per
  `scalarAssignments` ordinal;
- every write is classified as local or terminal output;
- every distinct local read names its exact earlier
  `producerStatementOrdinal`;
- each producer stores deduplicated `consumerStatementOrdinals` reverse edges;
- indexed locals carry parsed `coordinatePrograms` under the same closed
  coordinate language used by calculation-graph edges; and
- construction rejects reads before production, malformed coordinates,
  missing or duplicate terminal output writes, free bindings and any serialized
  graph which differs from the class-wide source contract.

The algorithm dispatches on scalar statement syntax, not operation, layer,
dtype, shape or assignment ID. Calculation-graph schema 6 rebuilds dataflow
after vision/audio call-site binding so outputs containing `/` and their local
programs are exact instantiated names. Formula-language schema 23 binds the
new authority at
`/calculationGraph/assignments/*/scalarCalculation/statementDataflow`.

## Real artifact and source-present storage audit

Source: dense unquantized `google/gemma-4-E4B`, immutable revision
`411aa17b749aa952df1359d2dcea73917a544d9a`.

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

- artifact schema 38, formula-language schema 23, calculation-graph schema 6
  and scalar-calculation schema 4;
- 21,394,548,803 bytes, SHA-256
  `63e3b8da96d56593c3935176a555614e12fde534833fae391b0e1d13aca5f628`;
- 2,130 constants/decoders and 15,992,314,836 embedded learned bytes;
- 2,708 instantiated assignments and 4,144 scalar statements;
- 4,144 statement-dataflow entries, 1,431 local writes, 2,708 output writes,
  1,958 local reads/producer edges, 1,931 deduplicated reverse-consumer edges,
  3,188 indexed-local accesses and 9,340 local-coordinate programs; and
- 1,462 assignment reductions, 256 staged reductions, 1,722 executable domains
  and exactly 100 runtime-defined BMMs.

The source-present audit used:

```bash
node dist/src/literal-artifact-audit-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --source ./gemma-4-E4B-dense \
  --verify-gemma4-payloads \
  --output /tmp/gemma4-v38-source-audit-final.json
```

All 2,130 payloads matched by name, dtype, shape and bytes. Source and literal
storage both produced SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.
The audit report SHA-256 is
`bb535d618fc1eebb89f2c2f92be64af009246e05e6e878b775a229118b41ad93`,
and `forbiddenSourcePathPresent` is false.

## Source-removed replay, navigation and literal substitution

`./gemma-4-E4B-dense` was physically renamed under an EXIT/INT/TERM restore
trap. While that exact path was absent, the reader opened only the schema-38
JSON:

```bash
node dist/src/gemma4-composite-literal-reader-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --verify-payloads --list-runtime-reductions \
  --end-to-end-calculation --generation-max-new-tokens 1 \
  --assert-source-unavailable ./gemma-4-E4B-dense \
  --output /tmp/gemma4-v38-source-removed.json
```

The report recorded `sourceCheckpointAccessed=false`, independently verified
all 2,130 payload commitments / 15,992,314,836 bytes, and reproduced the same
literal-storage SHA-256. It opened 2,708 forward operations, 13 one-step greedy
operations, 2,076 reachable constants and 54 declared runtime-unreachable
shared-KV constants. Its SHA-256 is
`46e92449a446d72aa174aa31c242b196c37ed9a2b46aafa699d6ce94c2286af8`.

A source-removed scalar window for `layer_0_q_proj[0,0,0]` decoded BF16 storage
words `0xbbcb`, `0x3af2`, `0xbd4c`, `0x3d08` into the exact literals
`-0.006195068359375`, `0.0018463134765625`, `-0.0498046875`, and
`0.033203125`. Every product contains the literal directly and no
`weight[...]` placeholder. The report SHA-256 is
`a796a962b66ee95fb314ef53cbae4c97bbfae4d72fc2553c47de766ee27af2a4`.

## Authoritative differential and preserved fidelity limit

Still with the source absent, the registered image, video and audio suite
compared schema 38 to
`transformers-5.5.0/torch-2.12.1-Gemma4ForConditionalGeneration-CPU-eager-inference-mode`
at zero absolute and relative tolerance. All three prefills and all three
greedy generations passed `lossless-within-dtype`; every modality generated
token `184`, matched 24 KV caches and had `firstDivergence=null`. The report
retains 32 image, 32 video and 36 audio runtime-defined BMMs and has SHA-256
`33a885879c5a2954dd878104382dcba4f8be0bb3df599d43381e80ad067e0c33`.

The source directory was restored and the temporary hidden path was absent
after every source-removed command.

Repository validation after the implementation, documentation and artifact
regeneration:

```text
npm run typecheck  # exit 0
npm test           # 252/252, exit 0
git diff --check   # exit 0
```

The 100 Apple Accelerate SGEMM BMM assignments remain
`fail-closed-runtime-reduction`. This boundary makes their surrounding scalar
intermediate dataflow explicit; it does not claim the unpublished native
product-rounding, lane, tile, FMA or accumulation schedule.
