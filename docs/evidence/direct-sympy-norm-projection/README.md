# Correlated RMS projection bounds eliminate more MLP dependencies

Expressions remain SymPy-compatible mathematical strings. JSON records here
are compiler state and evidence, not expression source. This change continues
the first coordinate; it does not promote other coordinates or multiple tokens.

The previous gate/up projection bounds treated the normalization's components
as independent maxima. `norm_dot()` additionally uses the certified Euclidean
bound of the actual named RMS output. It composes concrete coefficients in
exact rational proof arithmetic, includes learned Half gamma storage, then
adds the original F32 reduction error before final Half storage. Generated
operations preserve their original order. Invalid normalization shapes and
unsupported proofs retain the independent bound; invalid projections provide
no certificate. Only the MLP projections of that actual normalization use it.

The test checkpoint's gate/up bounds decrease from 0.021270751953125 and
0.0400390625 to 0.0206298828125 and 0.032135009765625. The two final MLP bounds
decrease from 0.000006794929504394531/0.000008046627044677734 to
0.000005304813385009766/0.000006258487701416016. They now fit strictly inside
the residual's storage cells in all four tested signed quadrants, allowing
both MLP dependencies to disappear before substitution of their producers.

The new native projection proof tests 5,332,992 cases, including every finite
Half value against zero, signed zero, subnormal and extreme anchors, with
different learned gamma weights and signed projection coefficients. There are
zero bound violations. The existing original layer-update test also validates
the tighter bounds on 888,832 cases; its F32/Half boundaries are retained.

Fresh compilation emits expressions for each signed region with both input
magnitudes in [0.03125, 8]. Each region contains 67,125,249 finite Half pairs.
All four regions are exhaustively checked against the original Torch CPU
forward, separately from the bound proof. The mixed-sign expression has the
same hash as the existing artifact. `artifact-map` and emission reports refer
to that exact file rather than copying it. The wider same-sign region requires
a new expression; `same.expr` preserves the actual emitted file.
The successful mixed-sign artifact has 10,269,774 characters and five paths;
the same-sign artifact has 11,063,008 characters and six paths. No compiler alias survives.
Substitution still runs SymPy factor/simplify to stability before the next
dependency, and conditions are distributed after those simplifications.

A separate attempt to compile the entire magnitude range [0.03125, 65504]
retains the same two MLP eliminations but grows its working expression to
218,511,376 logical characters and stops at the 16 MiB flat-artifact budget.
It emits no complete artifact and admits no input pairs. That negative result
is retained rather than extending the smaller artifact's domain by assumption.
The next investigation must locate the additional conversion/normalization
expansion and simplify it before completing the remaining input regions.

The recovery resets old numerical leaves after the source change, retains
only their audited geometry, and recompiles before fresh parity admission.
Coverage in `validation.json` is disjoint certified-region coverage; it is
not an exhaustive reference comparison of every covered pair. The full
coordinate, full-coordinate parity and multiple-token parity remain incomplete.
The test map is append-only and preserves the prior failure evidence.

The recovered frontier covers 2,518,181,896 of 4,030,726,144 finite-Half input
pairs (62.4746%), adding 138,448,898 over the preceding source version;
1,512,544,248 remain pending. All 82 promotions are fresh compilations with
672,118 reference comparisons and zero mismatches. The four larger declared
regions additionally have exhaustive parity on all 268,500,996 pairs.

All 32 direct-string integration tests pass without skips; the bound helper
now runs ten tests instead of eight. The whole-project suite still has 626
tests, 570 passes, 15 failures and 41 skips. Every failing location matches
the preceding evidence; no new failure location is introduced.

A matched comparison of the same four bounded projection regions uses one
and two CPU workers under a 3 GiB admission limit. Time decreases from
11.620 to 6.468 seconds (1.797x); sampled process-group peak RSS increases
from 559,284,224 to 831,307,776 bytes. Both modes compile the same 16,842,752
input patterns, produce identical file hashes, and validate 1,044 emitted
outputs with zero mismatches. This is a regional benchmark, not a timing or
parity claim for the unfinished full coordinate.
