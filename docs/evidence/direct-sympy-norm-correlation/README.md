# Cross-normalization constraints and emitted mixed-sign expression

This change eliminates impossible combinations of decisions using relationships
between actual stored normalization vectors. It does not replace the original
numerical operations with real-number algebra. Expressions remain mathematical
strings, following the user's latest format instruction; JSON files here contain
compiler state and evidence only.

`mixed.expr` is the actual fully substituted expression for output coordinate 2,
position 0, with one token in either of these declared regions:

| X1 | X2 | Exhaustive input pairs |
| --- | --- | ---: |
| [-2, -0.0625] | [0.0625, 1] | 20,980,737 |
| [0.0625, 2] | [-1, -0.0625] | 20,980,737 |

It contains 10,253,949 characters, five paths and no compiler aliases. Both
regions independently produce byte-identical expressions. Both exhaustive runs
execute that file against a fresh Torch CPU checkpoint forward: zero bit
mismatches in all 41,961,474 pairs. These are genuine emitted-output checks,
separate from the analytic bound tests. The earlier same-sign expression had
92,413 mismatches in each mixed-sign region; it is not reused there.

The full coordinate and multiple-token compilation are still incomplete. No
other output coordinate is promoted by this evidence. The full finite-Half
input domain has 4,030,726,144 pairs, including signed zeros. `validation.json`
and `frontier.json` record current coverage and remaining work, rather than
claiming the listed regions are the whole input domain.

## Why the constraint is valid

For a source vector `x` and its actual rounded update `x+d`, the compiler bounds
`d` from the original ordered F32 operations and Half storage. It then bounds
the change of the real RMS mapping along the segment using
`sqrt(n) * ||d|| / (r - ||d||)`, where `r` is a certified positive lower bound
for `||x||`. Independent F32/Half normalization errors are added at both ends.
Different contexts, epsilon values, unsupported dependencies, missing finite
bounds or an update at least as large as `r` provide no usable relationship.

The resulting relationship augments the existing exact-Fraction linear-system
proof. If simultaneous small projections would force the normalization vector
below its certified norm floor, that branch combination is unreachable.
Otherwise it remains reachable. Original arithmetic is retained in every
surviving path. Compile-time sharing never becomes runtime intermediates.

Selected definitions may be recovered only through equality of their whole
AST in the guard's own frozen prefix. The emitted integer-grid conversion
composition is recognized only to enclose its numerical error; this is not
permission to reuse a normal conversion outside its context or discard its
subnormal decisions. Unrecognized offsets remain unproved.

Each dependency still follows substitution, CPU SymPy `factor()`/`simplify()`
to stability, condition propagation, then the next dependency. The five paths
are distributed after those simplifications. `emission.json` records producer
growth and the final file, not a proxy for completion. This region needs an
explicit 16 MiB CAS/artifact budget; the 8 MiB attempts failed without admitting
a partial result. The successful diagnostic took 105.97 seconds internally;
that timing is not an isolated sequential/parallel benchmark.

The separate matched-region comparison processes the same four signed regions
covering 16,842,752 input patterns. One worker takes 11.697 seconds and two
workers take 6.494 seconds (1.801x). Sampled parent/worker RSS rises from
551,682,048 to 847,675,392 bytes. Each mode validates 1,044 actual candidate
outputs with zero mismatches; all four file hashes match between modes. This
measures the listed regions, not the unfinished coordinate. Admission reserves
each worker's memory and enforces a 3 GiB total limit.

## Colab execution and native parser repair

The current sources and checkpoint were uploaded through Access Broker's
`colab_cli` to `llm-inner-norm-correlation`: two CPUs, about 12.67 GiB host RAM,
and a Tesla T4 with 15 GiB device RAM. Symbolic algebra runs on CPU. CUDA
validation runs in a separate process, without initializing CUDA in CPU CAS
workers. Its results admit only the numerical operations and domains tested;
they do not switch the model reference to a GPU backend.

| Same four regions | One worker | Two workers | Speed ratio |
| --- | ---: | ---: | ---: |
| Local, seconds | 11.697 | 6.494 | 1.801x |
| Colab, seconds | 29.123 | 25.063 | 1.162x |
| Colab sampled parent/worker RSS, MiB | 1452.2 | 1980.3 | |

Both modes have identical artifact hashes and zero corpus mismatches. The
mixed region takes 205.97 seconds to compile remotely and produces the same
10,253,949-character file/hash as locally. This Colab runtime is slower for the
symbolic workload; it is not evidence of a faster whole-coordinate compile.

The T4 passes 888,832 Half-product checks and 19,458 SiLU/threshold checks,
with zero CPU/CUDA bit mismatches in the declared domains. Product medians are
6.586 ms CPU, 0.218 ms on resident GPU tensors, and 2.414 ms including transfers
(2.729x over CPU). These measure the numerical kernel, not SymPy or inference.

The original remote validation failed because older Clang's default lexical
nesting limit was 256. The emitted expression was retained; both native
validators now specify `-fbracket-depth=4096`, retaining `-O3` and
`-ffp-contract=off`. `colab-recover.py` checks the terminal original report,
unchanged numerical source/checkpoint identity and exact artifact hash before
validating that existing file. The corpus passes all 8,196 cases. The two
exhaustive tests then pass all 41,961,474 inputs, zero mismatches, against the
Colab Torch CPU backend as well. Full local exhaustive checks also pass after
the parser adjustment. The original failure and successful recovery are both
retained in `colab/`.

Colab uses Python 3.13, Torch 2.11.0+cu130 and SymPy 1.14.0; local evidence
uses its separately recorded backend identity. No numerical savepoint is
reused across those differing environments. `record-colab.py` audits the
actual-file parity, source hashes and resource comparison before appending
`colabNormCorrelationValidation`. Repeated identical audits preserve every
existing test-map byte; differing evidence requires a new entry.

Two kernel observation calls lost their WebSocket connection and timed out.
The same operations were polled to termination. File downloads through the
same Broker connector recovered the authoritative terminal reports; the
compilation was not replayed because an observation failed. A session marked
IDLE was not taken as evidence that its child compilation had stopped.

## Validation, recovery and rejected experiment

The six new proof tests include all 83,922,948 original rounded pairs across
the four central sign combinations. Distance and exclusion violations are zero.
The compiler integration gate has 31 passing tests, zero failures and zero
skips. The full project run has 570 passes, 15 failures and 40 skips; all 15
failing locations match the preceding audited baseline. The prior baseline
reproduction and causes remain in `../direct-sympy-coupled-projections/`.

An attempted larger mixed region, with X1 approaching -0.03125 and X2
approaching 0.031280517578125, exposed an invalid assumption in the new test:
the MLP update is not eliminated throughout that domain, and there is an
additional normalization vector. The two-vector native test was corrected to
its proven domain. Actual compilation of the wider region hit the artifact
limit; zero inputs were admitted. Its failed logs are retained. That wider
domain remains outstanding work, not an input restriction on the final goal.

`recompile.py` imports only the audited geometry from the previous frontier.
It resets all 243 completed numerical leaves, rejects mismatched live identities
on resume, recompiles the old regions plus the two mixed regions and validates
each actual file before atomic admission. `record.py` verifies the restored
coverage, file hashes, source identities, exhaustive checks and existing test
map before appending `normCorrelationValidation`. Historical map entries remain
unchanged. Byte-identical files may share a reference after fresh compilation;
old numerical evidence is not silently accepted under changed sources.

Reproduction uses `/private/tmp/llm-inner-pytorch/bin/python` locally:

```sh
LLM_INNER_DIRECT_JSON_CHECKPOINT=docs/evidence/direct-sympy-test-checkpoint \
  python helpers/direct_sympy_norm_correlation_test.py
python docs/evidence/direct-sympy-norm-correlation/recompile.py
python docs/evidence/direct-sympy-norm-correlation/exhaustive.py \
  --artifact docs/evidence/direct-sympy-norm-correlation/mixed.expr --x-sign -1 --y-sign 1
python docs/evidence/direct-sympy-norm-correlation/exhaustive.py \
  --artifact docs/evidence/direct-sympy-norm-correlation/mixed.expr --x-sign 1 --y-sign -1
python docs/evidence/direct-sympy-norm-correlation/record.py
```

The CLI flag name `LLM_INNER_DIRECT_JSON_CHECKPOINT` is retained for compatibility
with the existing test harness; it does not change expression representation.
