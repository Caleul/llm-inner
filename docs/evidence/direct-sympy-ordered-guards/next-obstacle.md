# Continue the complete coordinate

The emitted broad mixed-sign expression is complete within its stated domain.
It is not the whole coordinate. Preserve the remaining domain including both
zero signs, smaller magnitudes, the other quadrants and numerical boundaries.

Inspect a lost normalization proof: `CheckpointStrings.norm` currently publishes
norm-vector metadata only when `norm_squared_floor` is strictly positive.
For zero-admitting sources this prevents even upper and complementary-guard
proofs from reaching the subsequent projections. Existing coupled projection
exclusions already require a positive floor. Determine whether publishing
well-typed source/mean/component metadata with floor zero preserves all proof
preconditions and permits branch-local refinement; do not assert a positive floor.

For same-sign source components and projection terms with a common sign, the
actual norm-vector lower bound may also exclude impossible near-zero projection
branches. Derive an ordered F32 reduction/storage enclosure with concrete weights
before attaching it to the exact consuming producer. Opposite-sign coefficients
can cancel and must retain their paths. Verify original numerical order and
both zeros; no unconditional real-arithmetic factorization is authorized.

The subsequent requirements remain a complete first coordinate, final dispatch
parity, multiple-token inputs, and then all output coordinates. Do not reuse old
numerical results after source changes or mistake regional coverage for completion.
