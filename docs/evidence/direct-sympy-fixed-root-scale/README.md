# Exact fixed-scale root normalization

The preceding regional certificates reduced expensive square-root bodies while
preserving their F32 output. This change simplifies their normalized argument:
when the entire positive finite F32 branch enclosure has one binary exponent e,
widened source * 2**(-e) is exactly the [1,2) mantissa previously extracted by
Bits64/mask/or/Float64. Every finite F32, including its subnormals, is normal in
F64, and the normalization factor and result are safely representable in F64.
No rounding, new branch or runtime intermediate is introduced. For e=0, use the
source directly. The ordered rational evaluation and final F32 rounding remain
unchanged. Unknown scale ranges keep the original word extraction.

## Numerical and checkpoint evidence

All four native root tests passed, including 58,720,257 fixed-scale cases,
25,167,601 universal mantissa/parity/subnormal cases, 67,895,912 regional emitted
cases and 19,398,662 regional normalized cases with zero mismatches/midpoints.
The fixed-scale test explicitly rejects residual Bits64(X1) source extraction.
The integration suite repeats this test with the new assertion enabled.

A fresh complete coordinate in the validation region [1/32,5/128]^2 passed all
66,049 Half pairs against the original checkpoint forward. Its input-only string
fell from 268,753 to 229,153 characters, with two arms and zero compiler aliases.
The previous and new smoke artifacts were both generated afresh; the sole
numerical source difference is direct_sympy_sqrt.py. No historical numerical
states were reused. These artifacts validate a region, not the whole model.

The same 27 ordered branch skeletons were measured in both versions. Candidate
24's body fell from 96,061,439 to 85,416,287 characters and its guards from
65,564,183 to 58,300,703. The previous baseline measurements are retained from
the preceding version's verified run; the new measurements are fresh. This
comparison is about exact expression sizes, not a controlled timing benchmark.

The whole-domain first-coordinate worker used the same 512 MiB accumulated
expression/condition ledger and independent RAM monitoring (parent/descendants,
200 ms samples, 80% of initially available RAM, bounded wall clock). It resolved
25 producers and generated 28 candidates, including the rejected final candidate,
versus 27 previously. It stopped at the artifact cap, published no coordinate,
and did not advance other logits. Peak observed RSS was 551,059,456 bytes. The
harness's reservation-refusal annotation describes the earlier controller run,
not a new controller failure; its metadata field is explicitly marked previous.

Current regression counts and exact source/artifact hashes are reconciled by
record.py. The test map retains previous entries and adds this proof, artifact
and bounded-attempt evidence. Complete whole-domain coordinate parity,
multiple-token compilation and the complete output vector remain unfinished.
Expressions are mathematical strings; JSON files here are execution evidence.

The first focused run had 39 passes and one stale emitted-size assertion: the
projection fragment became 61,355 characters rather than the historical 62,059,
while both its native parity corpora still had zero mismatches. The exact size
expectation was updated; both affected root/projection integrations passed on rerun.
The initial failure remains recorded; no numerical assertion was relaxed. The
portable suite passed 571 tests, zero failures and 49 existing explicit skips.
