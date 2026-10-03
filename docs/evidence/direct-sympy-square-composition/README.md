# Completed-square tandem composition

The certified Half quadratic x*(0.5+x*0.25) admits the one-occurrence magnitude ((x+1)^2-1)*0.25 for finite Half |x| <= 3/128. Exact Fraction checks cover all 19,458 Half patterns. Both raw values equal the same dyadic value, except that raw -0 becomes +0. The rewrite is therefore restricted to the F64/F32/Half conversion boundary, preserving the original operand sign in the subnormal arm. It is never registered as a generic same-sign raw expression.

Nonzero values have at most 37 significand bits; the existing exact tandem rounding route applies. Both candidate and original pass mandatory substitution/factor/simplify. The candidate is admitted only when the compact result is smaller; replacing its fewer source occurrences by the same larger literal preserves that size advantage.

The existing direct-X1 kernel remains smaller and is retained. A separately emitted candidate kernel is tested natively against scalar and vector Torch SiLU for every certified Half pattern, including both zero signs. Out-of-domain and untyped sources reject this certificate.

95 Python tests, 8 Node integrations and 9 optional legacy JSON certificate tests passed. Build passed. Default broad npm test produced 602 tests: 570 passed, the same 15 baseline failures, 17 skipped. It includes a previously excluded legacy model suite and omits environment-dependent integrations; raw totals are not comparable to the prior 599-test command. Focused environment-dependent suites were executed separately. See regression-comparison.json for scope.

Fresh local compilation reaches 13 dependencies and declines before allocating the 2,962,962,964-character activation at its 2-GiB budget. The previous activation was 4,279,835,121 characters. This is an allocation estimate, not a newly emitted local activation.

A fresh Colab run is active under /content/llm-inner-frontier-composition/square with a 16-GiB character budget. It passed remote conversion, savepoint, SiLU and CUDA checks, and 16 completed dependencies were observed; the actual gated producer contains 16,131,686,432 characters, SHA256 e531eede37afd1d2c0cc17fbc09b640067434b8199b4ee3635be44a3758a36f2. No full coordinate or new producer parity is claimed before terminal validation. The previous run is progress: implementation, commit/push and actual smaller activation evidence were completed; the full model goal remains incomplete.
