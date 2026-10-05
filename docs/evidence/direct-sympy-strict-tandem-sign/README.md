# Remove a duplicated producer only under a strict sign certificate

The composed F64 -> F32 -> Half conversion previously restored a sign by
reevaluating the source or a caller-supplied sign expression, even when its
branch-local finite interval proved a strict sign. This change replaces only
the sign field by zero or the IEEE sign bit for minimum>0 or maximum<0.
The magnitude calculation, both original storage boundaries, small/overflow
thresholds and order of evaluation are unchanged. No branch is introduced.

A strict source sign remains valid when F32 or Half rounding produces signed
zero or infinity. A zero-inclusive interval does not supply this certificate:
its existing sign expression is retained, preserving both zeros and lazy
branch contexts. The native test covers all nonzero finite Half inputs times
eight positive F32 factors, comparing the emitted positive/negative conversion
strings against actual native F32-then-Half storage, including signed zero
and overflow. All 507,888 cases match bitwise. The unchanged mixed-domain
corpus includes both zero signs and checks another 507,904 cases.

The previous and current regional checkpoint strings are generated afresh,
then their actual emitted bytes are evaluated against the original forward.
Both the established [1/32,5/128]^2 region and a small-output region are recorded.
Neither is a whole-domain coordinate. Historical arm measurements are used only
for expression-growth comparison, never as reusable numerical compiler state.

A new full-domain first-coordinate run uses the same accumulated 512 MiB
expression/condition ledger and separately measured aggregate RSS. Other logits
remain gated on a complete coordinate artifact and its parity. Measurements of
candidate arms are not substitutes for that final artifact. The previous
controller admission note in the harness is explicitly historical.

Expressions remain mathematical strings; JSON files here record evidence.
record.py binds source/checkpoint/backend/artifact hashes and appends to the
existing test history. Complete-domain coordinate, multiple-token and final
vector parity remain unmet until actual artifacts prove them.

## Current observed result

The established region still emits 217,033 characters and verifies 66,049 input
pairs; the small-output region [2**-18,5*2**-20]^2 emits 120,415 characters and
verifies all 289 input pairs. Both current outputs are byte-identical to their
fresh baselines. All 27 measured whole-domain candidate bodies, guards and
ordered guard skeletons are likewise unchanged. This transformation is exact,
but it has not demonstrated a reduction of this checkpoint's emitted output.
Further work must improve propagation/proof reachability or another source of
expression duplication; it must not infer a sign where the certificate is weak.

The full-domain attempt still completed 25 producers and generated 28 candidates
including the rejected final candidate, then stopped at the 512 MiB accumulated
artifact cap. Peak observed aggregate RSS was 484,933,632 bytes under a separate
6,738,188,697-byte RAM admission. It published no coordinate and no vector.
The run overlapped diagnostics/tests, so elapsed time is not a controlled speed
comparison. Complete-domain parity and multiple-token parity remain unexecuted.

All 40 focused integrations passed without skips or failures. The portable suite
passed 571 tests, zero failures and 49 existing explicit skips. After adding the
TS assertion for the new native strict-sign corpus, the affected elementary
conversion integration was rerun separately; its final result is recorded.
