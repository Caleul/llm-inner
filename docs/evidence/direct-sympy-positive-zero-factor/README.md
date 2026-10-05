# Exact factoring with the original positive-zero guarantee

After proving every original and proposed F64 operation exact, the factor
pass can now restore the original positive-zero guarantee with a final
addition of +0. This admits multivariable common factors when CAS removes
a reduction's +0 seed. All nonzero values remain unchanged. A negative-zero
result becomes positive zero only when the original certificate excludes
negative zero. Without that certificate the expression remains a barrier.
No new branch, checkpoint lookup or runtime intermediate is introduced.

The new native regression compares both sum and difference forms over all
30,722 Half values in [-1,1], nine companion inputs per value, and two
expressions: 552,996 comparisons with zero bit differences. Companions include
both signed zeros and exact cancellation. The emitted C++ uses the actual
factored expressions, nearest rounding and disabled contraction. The test
also checks idempotence, fewer producer copies and refusal without the
original positive-zero proof. All seven exact-factor tests pass; existing
92,166 ordinary and 122,888 signed-zero factoring cases still match.

This extension does not reduce the checkpoint's measured sizes. Its full
logical composition stays 2,296,124,090,440 characters; the same first bounded
path remains 2,061,331 body characters and 1,554,142 guard characters. The
32/96 MiB diagnostics complete three/eleven paths, respectively, with the
same final measured dependency sizes as `532c0cc`. They stop at the artifact
budget. No complete unrestricted coordinate is admitted.

`current.expr` is the actual one-token coordinate 2, position 0, for both
inputs in [1/32,5/128]. Its 326,227 characters and two paths are unchanged.
All 66,049 input pairs agree bit for bit with Torch CPU. This regional parity
does not prove the unrestricted coordinate, other tokens or the full vector.
Single timing observations do not establish acceleration.

`run.py` reconstructs the `532c0cc` arithmetic helper in an isolated directory
and rebuilds baseline/current regional strings. `diagnose.py` measures bounded
full-domain growth. `full_run.py` attempts complete publication in a separate
worker. `record.py` verifies source identities, hashes, exact parity and
regression logs before appending `positiveZeroFactorValidation` to the test
map. Run drivers from the repository root with the compatible Python runtime.
JSON files are metadata, while expressions remain mathematical strings.

Next investigation: the existing branch bounds sometimes restrict a Half
source to one dyadic unit of either sign (plus signed zero). A homogeneous
rounded scale may then reduce to an exact scale over that entire certified
domain. This is not implemented or counted as validation in this evidence.
