# Reduce substituted dependencies with a certified narrow root kernel

The existing universal F32-root expression contains five copies of its source.
This change admits a two-copy rational 2/2 expression when the prefix proves
that the entire source enclosure is within one binary exponent range and its
normalized mantissa lies in [1.03125,1.0625]. Selection occurs during compilation;
no runtime selector or response lookup is introduced. Other enclosures retain
the original 5/5 formula. All coefficients describe interpolation of sqrt,
independent of the checkpoint; every numerical evaluation order remains fixed.

The new coefficients are reproduced from five Lobatto samples at 100 decimal
digits. Native enumeration checks all 524,290 F32 mantissa/parity combinations
in the admitted interval, finding no mismatches and no F32 midpoints. A further
1,835,032 cases check the actual emitted expression at eight exponents, including
F32 subnormal sources and both parity scales. The universal fallback retains
zero mismatches over its 25,167,601 mantissa/subnormal/exponent cases and
58,720,257 fixed-scale cases. Adjacent out-of-range bounds and enclosures crossing
binary exponents cannot use the smaller kernel.

A second change tightens backward RMS constraints at small output magnitudes.
If H bounds the stored output norm, the existing relative-loss theorem gives
real_norm <= loss*(H+sqrt(n)*2^-25)/(1-2^-11). Intersect this with the older H+E
certificate before inverting n*T/(T+n*epsilon). The original numerical RMS
calculation is preserved. Independent source caps intersect existing grids and
nonzero exclusions; the proof never assigns a sign to zero. Native tests cover
37,773,316 central pairs, with 552,300 selected and zero bound violations, plus
319,488 full-axis/mixed cases of tiny outputs. All nine selected tiny-output
cases are zero-input pairs, including every input zero sign.

## Emitted artifacts and growth

The three newly emitted regional mathematical-string artifacts have no compiler
aliases and match the checkpoint bit for bit over all 66,317 regional pairs.
The same mixed-input region shrinks from 2,913,052 to 811,296 characters. This is
an actual artifact comparison over identical input domains. Regional parity does
not establish complete-domain, variable-token-length or full-vector parity.

With the same 32 MiB publication budget, the first rejected candidate remains
number 14, now 20,426,239 calculation and 10,378,893 condition characters. With
96 MiB the compiler reaches 22 candidates, compared with 14 previously; the
rejected candidate has 19,448,599 calculation and 10,152,959 condition characters.
The cumulative publication budget also matters. A separate capture reaches the
24th candidate before a single calculation alone exceeds 96 MiB. Its expression
still uses the universal root kernel, which identifies the next remaining
expansion. Different candidate prefixes are not treated as equal-size comparisons.

The isolated 96 MiB full-domain attempt stops after about 85 seconds with a
290,242,560-byte child RSS peak. It resolves more candidates than the previous
run but is not evidence of shorter total compilation time or parallel/CUDA
acceleration. No full.expr, truncated prefix or full-coordinate parity is claimed.
A broader central slab also stops at its budget; it emits no slab.expr and has
no parity result. Its failed attempt is retained to expose the expansion cliff.

## Regression and state evidence

All 38 focused integration tests pass. The general suite has 632 tests: 570 pass,
the same 15 locations fail and 47 skip. `record.py` verifies current numerical
source/backend/checkpoint identity, real artifact hashes and test counts before
appending the entry to the existing test map. Numerical states from the previous
source identity are incompatible. No historical numerical state was reused.
The intermediate relative-only evidence lives in the sibling
`direct-sympy-relative-rms` directory and is explicitly not the current compiler.
The complete coordinate, multiple-token parity and full output vector remain
pending; the next work must simplify surviving universal-kernel dependencies
without replacing them with runtime intermediates or dropping input paths.
