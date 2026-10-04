# Exact completed-body coalescing and bounded continuation

The complete coordinate remains unfinished. The 14 completed input regions
cover 178,115,584 of 4,030,726,144 finite-Half input bit-pattern vectors
(4.4189453125%). The remaining 3,852,610,560 vectors stay pending; they are
not discarded, approximated or replaced by known-response lookup.

Readback of the actual completed expression files found only four byte-distinct
functions. Exact adjacent rectangle unions reduce 14 domain guards to five
rectangles. Grouping identical functions could avoid 4,958,910 repeated body
characters: 6,035,807 copied characters versus 1,076,897 unique characters.
This is a completed-body duplication correction, not proof that the numerical
expressions of pending regions have been sufficiently simplified.

`helpers/direct_sympy_partition_coalesce.py` audits the complete split tree,
checks each file digest and size, and only unions adjacent rectangles with the
same function and identical bounds on every other axis. Different numerical
branches, including signed-zero results, retain their individual bodies.
Nonadjacent equal functions use disjunctive domain guards without filling gaps.
Domain checks precede numeric checks, preserving short-circuit evaluation.
Final output is atomic and refused while any input region is pending.

SymPy factor/simplify checks the condition envelope with closed numerical
expressions protected. The original guards are retained to preserve their
execution order; this pass does not claim real-algebra simplification of the
protected floating operations. Expression files use mathematical strings;
JSON here records provenance, coverage, metrics and validation only.

Validation:

- Build passed; all 18 named integrations passed with no failures or skips.
- The final numeric-guard protection change additionally passed its focused
  integration and all three coalescing Python tests.
- A synthetic complete-domain emitted function was read back and compiled
  natively: 126,976 comparisons, zero bit mismatches.
- Actual 14 region expression files were compiled and compared against fresh
  CPU checkpoint execution: 3,640 comparisons, zero bit mismatches. This covers
  position zero, dimension two and one token, not the complete input domain,
  last-token semantics or the output vector.

The bounded local continuation used 24 additional attempts (36 cumulative),
154.5422225 seconds and 590,299,136 bytes maximum RSS reported by `time -l`.
This RSS is not a measurement of simultaneous aggregate worker memory.

Colab evidence records an actual A100 run through Access Broker. The initial
missing-clang failure was retained and repaired through the same interface.
The repaired job finished its dispatched tests and bounded 12-attempt run;
its three regions cover 3.125%, with 780 native region comparisons and no bit
mismatch. Completion of that job is not completion of the coordinate.

CUDA numerical primitive validation passed 888,832 Half products and 19,458
activation-domain values. Median product timing was 0.002098874 seconds on CPU,
0.000085956 seconds for the resident CUDA kernel and 0.002527398 seconds including
transfers. Kernel acceleration did not yield transfer-inclusive acceleration;
these are not full-model or symbolic-compilation timings. SymPy stayed on CPU.

The intended same-session sequential/two-worker comparison has no verified
result: readback on 2026-10-04 reported the named session missing, and the
session listing reported no active sessions. No uncertain job was redispatched.

`validation.json` and `docs/direct-string-validation.json` retain the test map
and unresolved work. Compiler/producer identities are unchanged by this
standalone coalescer; saved states still require source and artifact verification.
