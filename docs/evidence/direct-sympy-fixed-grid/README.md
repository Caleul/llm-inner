# Signed-binade grid conversion and global expression sharing

`fixed_grid_conversion` admits a one-occurrence arithmetic spelling of a
normal IEEE conversion only when the branch already proves a finite,
nonzero sign and a single binary magnitude interval `[2^e, 2^(e+1)]`.
Zero crossings, subnormals, multiple binades and overflow retain the
existing elementary word kernel.

For precision p, let `q=2^(e-(p-1))` and `K=sign*2^(e-(p-1)+52)`.
Ordered F64 `(x+K)-K` rounds on q's grid. The sum stays in K's binade,
and subtraction is exact by Sterbenz. This matches the target conversion
on the certified interval, including both endpoints. There is no residual
rounding helper or mode node, no mathematical reassociation, and only one
occurrence of the original source. The existing global IEEE certificates
prevent SymPy from cancelling the rounding addition/subtraction.

The final rule preserves the original spelling when the existing kernel
already contains one source occurrence (`integer_word_exact` or
`no_odd_f32_ties`). This is a global sharing constraint, not only a local
character-count preference.

## A local improvement that was rejected

The initially unrestricted rule shortened one actual checkpoint region
from 17,712 to 14,232 characters. But after recompiling exactly the same
covered domain, distinct bodies increased from 24 to 33 and their total
size increased from 3,657,767 to 4,025,983 characters.
`initial-coalescing-comparison.json` records that rejected experiment.
Its partial state remains separately in `fixed-grid-state`; it must not
be reused under the current numerical source identity.

Preserving the existing one-occurrence kernels recovered all 24 identical
bodies and the original 3,657,767 characters. The selected model-region
expression is now byte-identical to its preceding form (17,712 characters).
`coalescing-comparison.json` and `diagnosis.json` record the corrected
comparison. There is **no claimed size or speed gain for the checkpoint's
already-covered domain**. The new primitive is admitted only where it
actually removes a source copy; its generic numerical proof is below.

This comparison measures distinct effective partial bodies, not the final
coordinate file. There are still pending domains. Single old/new timings
are diagnostics, not a controlled whole-coordinate benchmark.

## Native proofs and regression gate

`helpers/direct_sympy_fixed_grid_test.py` validates the actual emitted
expression against native target casts:

- 50,331,648 normalized F32 midpoint/neighbour/sign cases;
- 184,314 Half midpoint/neighbour/sign cases across every normal Half
  binade, also checking the original F32-then-Half storage order;
- 15,234 endpoint and midpoint cases across all 254 normal F32 binades;
- constructor rejection outside the certificate and byte preservation
  of the original one-occurrence kernels.

All comparisons passed. Exact numerical equivalence follows from the
signed-grid proof; the enumeration does not claim to enumerate all F64
patterns. The paired actual model artifacts were independently compared
with the checkpoint on 8,196 input pairs each, with zero mismatches.
Both artifact files are kept for the explicit old/new comparison.

The build and full **28-test** direct-string integration gate passed.
The gate first exposed a missing `<cmath>` in the Half sum verifier when
its new expression included constant powers. `missing-header.log` records
clang's actual `no member named 'pow'` failure. The test now includes the
required header and preserves compiler stderr in assertion diagnostics.
Its numerical parity passed on 1,050,625 cases. The TypeScript assertion
now accepts changing character counts; the Python test still asserts
that the proved interval optimization makes the expression smaller.

## Coverage and saved-state compatibility

Before changing numerical compilation, targeted same-sign covers added
2,757,250 input patterns and performed 24,588 fresh checkpoint/native
comparisons with no mismatches. Their effective expressions, exact domains
and reports are in `../direct-sympy-silu-followup/`.

The numerical change rejects the previous saved state without modifying
it (`state-compatibility.json`). Only audited geometry was reused: all
141 formerly completed leaves were reset and freshly recompiled under
`fixed-grid-reuse-state`, with current source/checkpoint/backend identity,
artifact hashes and native parity checked before publication.

The 81 successful broad compilations performed 663,898 comparisons,
with zero mismatches. Exact covered cardinality is 2,044,212,866 of
4,030,726,144 finite-Half input pairs (50.715747%). There are still
1,986,513,278 pending patterns. Pair corpora can overlap and are not
exhaustive validation of the covered rectangles.

`frontier.json`, `artifact-map.json`, `validation.json` and the appended
`fixedGridConversionValidation` test-map entry preserve the actual
artifacts and this limited proof scope. Byte-identical existing bodies
are linked once; no previous numerical result was reused as a new proof.

The complete coordinate, multiple-token last-position parity, the full
output vector and subsequent Rust artifact remain unfinished. This is
coordinate 2, position zero, one-token embedding input evidence.
