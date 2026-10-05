# Remaining expansion

Actual branch proofs now propagate into V and the head projection. The O
projection occurs within a residual sum and must not replace that sum. Its
concrete ordered arithmetic can still simplify after replacing V with a constant.

The broad diagnostic yields 28 paths but 30,039,520 serialized characters.
Guards contribute 14,418,563 characters; their 211 occurrences contain only
35 distinct fully expanded strings. Inspect shared predicate prefixes and
pairwise Boolean combinations after each arm reaches its CAS fixed point.
The 28 paths have 16 distinct body strings; two bodies each occur in seven
paths. Consider combining identical bodies with prefix-factored guards, while
retaining each conjunction's original lazy order. Do not rewrite a predicate
with its own assumed truth, reuse a later branch's constants in earlier dispatch, or change lazy evaluation order without proof.

The normal/normal final arm still contains 742,153 body characters and 908,066
guard characters. Its first normalization components each expand to 3,057
characters, V to 6,246/6,243, residuals to 12,689/12,691, final mean to 358,233,
and final normalization to 371,008/371,010. Distinguish arithmetic copies in
this chain from repeated flat-dispatch guards before proposing another rewrite.

The remaining requirements are a complete input-domain expression for one
coordinate, final dispatch parity including zeros/boundaries, then multiple
tokens and the rest of the output vector. No narrow region is the full delivery.
