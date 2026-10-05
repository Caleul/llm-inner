# Certified normalized-root rounding and exact rescaling

The input normalization introduced by the preceding change is retained. This
change shortens the remaining normalized-result conversion and known exponent
adjustment. It does not change the rational coefficients or their F64 evaluation
order, restore a sqrt runtime primitive, or add numerical decisions.

For each certified rational root kernel, the native tests certify the emitted
expression with the normalized result rounded by (candidate + 2**29) - 2**29.
The addition rounds on the 2**-23 grid and the subtraction is exact by Sterbenz.
The normalized root is in [1,2], apart from tiny evaluation errors at its
endpoints, which the exhaustive certificates also cover. This is a property of
these ordered kernels, not permission to use that grid for arbitrary F64 values.
No round or nearest-even function is present in the emitted mathematical string.

For a branch with one proved binary scale exponent e, the widened rounded root
is then multiplied by 2**(e//2). The multiplication is exact: the significand has
at most 24 bits, and every output is normal F64 throughout the positive finite
F32 input range, including F32 subnormals. It produces the same bits as the
previous exponent-word adjustment. If e//2 is zero, the adjustment disappears.
Unknown scale ranges retain the original explicit exponent-word calculation.

The four native tests verify universal mantissas, both exponent parities, all
F32 subnormals, regional kernels and independently emitted fixed-scale
expressions. Their logs are preserved. Fixed-scale emitted expressions are
also checked structurally to contain neither Bits64 nor Float64.

Both the previous and current checkpoint smoke expressions are generated afresh
in the same region [1/32,5/128]^2. Their emitted bytes are verified against all
66,049 Half input pairs and bound to checkpoint, compiler and reference-backend
hashes. They prove regional coordinate parity only. They are not the final
whole-domain model artifact. No historical numerical compiler state is reused.

The complete-domain first-coordinate attempt uses the accumulated 512 MiB
expression/condition ledger and independently monitored aggregate RAM. It must
not dispatch other logits before complete-coordinate parity. Candidate growth
measurements likewise never stand in for a final coordinate.

Expressions remain mathematical strings; JSON files in this directory record
execution evidence. record.py verifies the facts and appends a new test-map
entry while preserving all earlier validation records. Full-domain coordinate,
multiple-token and whole-vector compilation/parity remain unproven until their
actual final artifacts are generated and validated.

## Observed checkpoint and regression results

The regional coordinate fell from 229,153 to 217,033 characters (two flat arms,
zero compiler aliases), with all 66,049 cases matching the checkpoint bitwise.
The same 27 ordered branch skeletons were measured. Candidate 24's body fell
from 85,416,287 to 81,677,811 characters and guards from 58,300,703 to 55,749,665.
All measured body/guard sizes were nonincreasing. Previous arm measurements are
retained historical evidence; current measurements and both smokes are fresh.
This is an expression-size comparison, not a controlled timing benchmark.

The whole-domain attempt still resolved 25 producers and generated 28 candidates,
including the rejected final candidate, then stopped at the 512 MiB artifact
budget. Peak observed aggregate RSS was 455,524,352 bytes and independently
admitted RAM was 7,184,069,427 bytes. No full coordinate was published, no complete
coordinate parity executed, and other logits were not dispatched. The harness's
previousControllerAdmissionStop field is explicitly a historical annotation,
not a new controller failure. Whole-domain worker elapsed time was 186.060s;
this run overlapped regression and diagnostic tasks and is not a speed benchmark.

The complete portable suite passed 571 tests, zero failures and 49 explicit
existing skips. The first 40 focused integrations passed 39 and failed only an
exact size snapshot: the projection fallback became 8,720 rather than 8,836
characters and its emitted fragment 60,187 rather than 61,355. Both native
numerical corpora reported zero bit mismatches. The snapshot was updated
without weakening numerical assertions; both affected integrations passed
on rerun, with their results retained separately from the initial failure.
