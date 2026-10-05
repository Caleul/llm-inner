# Verified regional admission and streaming combination

The A100 was already exercised through Access Broker; see
[the actual A100 results](../direct-sympy-a100-512/README.md). Compatible resident
numerical batches accelerated, but the direct coordinate stopped at the same
512 MiB expression/condition cap. SymPy remained on CPU. This change addresses
compiler admission and artifact growth rather than allocating another GPU.

Previously the adaptive controller counted a compiled region before numerical
validation, checked accumulated size only during final combination, and retained
every combined piece in a list. The controller now isolates candidates, charges
the expression together with every final outer/inner condition before promotion,
and offers `--verify-regions --parity-cases 8192` to require fresh native checkpoint
parity first. It streams the final combination into an atomic temporary file.
Incomplete coverage still prevents publication of a full coordinate. The verifier
source and verification policy now participate in savepoint compatibility.

An initial implementation reopened already stabilized numeric conditions during
size accounting. It consumed one CPU core for more than three minutes; that exact
owned test process was terminated. `initial-admission-source.py` and
`initial-partition-tests.log` preserve the failure. The correction keeps each
closed numeric condition opaque only while factoring/simplifying the added input
bounds. The emitted condition and numeric body retain their original expressions.
The final nine partition tests completed in 18.389 seconds, including actual
checkpoint/native parity for 576 inputs with zero mismatches. Synthetic test
fixtures test promotion gates; their reported complete coverage is not checkpoint
progress.

Actual whole-domain geometry has 4,030,726,144 finite Half bit pairs, including both
zero signs. Reimporting only the diagnostic geometry and visiting the largest
pending region first produced zero complete expressions in twelve bounded attempts
(`recompilation.log`). Timed-out regions were split without losing coverage. Its
`coverage-frontier.json` records that negative result. Old numerical expressions
and proofs were not reused.

A fresh checkpoint-bound seed then compiled four outer regions. Each expression
was translated into native C++ with contraction disabled and compared against a
fresh CPU checkpoint forward before promotion. These regions cover 1,006,505,988
input bit pairs (24.970835%); their sampled native checks total 32,784 inputs and
zero mismatches. This is sampled parity within complete regional expressions,
not exhaustive verification of every covered pair. The four `.expr` files here
are the actual emitted expressions. `artifact-map.json` binds their domains,
hashes and parity. `frontier.json` retains all unfinished regions. Their eventual
flat combination costs 46,924 accumulated characters, conservatively including
input bounds and separators, below the 536,870,912-byte limit. All emitted syntax
is ASCII, so characters and UTF-8 bytes coincide for these artifacts.

RAM remains a distinct measured resource. `ram.json` sampled the last 124 seconds
of the largest-first run and observed 401,211,392 bytes aggregate RSS. Monitoring
started after dispatch, so this is an observed peak in that interval, not an
entire-run peak. `certified-ram.json` contains only one late sample and cannot
support a peak-memory claim or sequential/parallel memory comparison. No compiler
speedup is claimed by these partial runs.

Validation: TypeScript build passed; all four affected integrations passed; the
portable suite passed 571/620 with 49 optional tests skipped and zero failures.
`record.py` verifies real artifact hashes, source identity, geometry, native proof
bindings and admission charges, then appends evidence to both existing test maps.
The full coordinate, other logits, variable-token expressions, Rust generation
and their parity remain unfinished. No remaining logit was dispatched.

Reproduce the verified regional run in a fresh state:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python helpers/direct_sympy_partition_run.py \
  docs/evidence/direct-sympy-test-checkpoint /tmp/new-verified-regions \
  --seed-update-cells --dimension 2 --verify-regions --parity-cases 8192 \
  --max-attempts 4 --region-seconds 15 --max-paths 128 \
  --max-characters 2097152 --cas-characters 8388608 \
  --total-characters 536870912
```

Exit status 1 means this bounded run left pending regions, not parity failure.
