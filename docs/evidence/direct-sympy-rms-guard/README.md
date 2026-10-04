RMS rounding classification simplification

The Llama adapter constructs the ordered F32 inverse R32(1/R32(sqrt(s))),
where s is its actual completed positive finite F32 mean with epsilon.
For a finite Half input x, the exact dyadic predicate x*x < s*2**(-28)
can select the two Half rounding kernels. The predicate omits the much
larger inverse expression, while the evaluated product and its rounding
order remain unchanged. This applies to that constructed inverse only,
not arbitrary positive factors. The derivation is in
helpers/direct_sympy_rms_guard.py.

The mathematical boundary is allowed to disagree with the rounded-product
classification only in an interval on which both emitted kernels agree.
The native boundary test explicitly observes 39,576 such disagreements
among 634,876 cases and zero output-bit mismatches. It includes all finite
Half values, both signed zeros, extreme positive F32 means and both F32
neighbors of each nonzero Half input's classification boundary.

The complete elementary normalization expression also passes 888,832
comparisons against the ordered native F32/Float16 reference and its
legacy predicate. Its length is 17,822 -> 12,289 characters. The existing
epsilon and double-rounding counterexamples are preserved in that test.

Resume tests stop after the real mean/inverse have been saved, restore
through the normal integrity/identity checks, reconstruct the certificate
in the new compiler session and assert byte-identical normalization output.
The new numerical source is part of the savepoint identity; old compiler
identities are not rewritten. Fundamental input domains stay unchanged.

Test map additions:
- helpers/direct_sympy_rms_guard_test.py: boundary parity and resume.
- helpers/direct_sympy_checkpoint_test.py: legacy/current RMS comparison,
  existing numerical-order counterexamples and remaining checkpoint checks.
- test/direct-sympy-strings.test.ts: invokes the new Python methods alongside
  the existing parallel, SymPy, conversion, SiLU, sqrt and savepoint tests.

These prove conversion/normalization behavior, not a complete compiled
coordinate. Real checkpoint expansion and saved-producer parity are recorded
separately; added mean producers must not be counted as new architecture
progress when comparing older frontiers.

The actual A100 run saved 18 producers, including two newly explicit means.
Against the previous 8-GiB run it advanced one architecture producer, gated:0.
Its expression is 7,273,880,362 characters versus 13,497,941,778 in the earlier
larger-memory run. Posterior normalization and the gate/activation/up
expressions are about 46% smaller. Compilation took 332.25 seconds and
peaked at 48,787,267,584 bytes RSS. These runs do not form a matching speed
benchmark: their stopping points and the older larger-memory hardware differ.

The next hidden envelope would restore to 12,951,847,412 characters and was
rejected before allocating that final string under the 8-GiB character cap.
Actual saved-producer parity then completed: 60 input matrices, 360 bitwise
comparisons, zero mismatches; it covers posterior normalization and MLP
producers at position zero, not all saved means individually or the final
next-token output at varying positions. Coordinate completeness is false.

The local focused integration passed 9/9 with no skips. The broad suite
has 570 passes, 15 failures and 18 skips; the exact failure names match the
prior baseline. The remote frozen numerical sources passed boundary,
checkpoint and savepoint gates. The additional interrupted-RMS resume test
was added locally after that snapshot; it passed locally with the same
numerical sources. JSON files here contain diagnostics, not expressions.
