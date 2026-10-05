# Selected numerical identities and frozen dispatch implications

The audit found two final-RMS predicates becoming byte-identical to earlier
post-attention RMS predicates after branch-local substitution. The old search
still assigned them independent truth values, even emitting candidate contexts
that required opposite results for the identical selected predicate.

`SelectedRecipeKeys` tracks exact typed numerical recipes only within the
selected prefix. It preserves operation order, literal types, casts and zero
signs. A separate magnitude key may differ only in the sign of zero; only a
finite square consumes it as a bit-exact identity. Thus `(h+0.0)^2` and `h^2`
agree even for negative zero, while `h+0.0` itself is retained. Unsupported word
functions, unknown calls, unsupported powers and division across zero remain
barriers. Interned proof keys prevent recursive copies of dependency signatures.
These keys are compiler-only and never survive emission.

When two selected means/inverses have identical typed recipes, reuse the earlier
completed compiler literal and run factor/simplify on the replacement before
continuing. When a new dispatch predicate has the same syntax and actual fully
expanded bytes as an earlier frozen predicate, propagate its earlier truth
before opening a second decision. Different frozen definitions are not assumed
equivalent. No predicate assumes its own truth during this check.

In the bounded 96 MiB full-domain run, eight recipe identities are reused and
two duplicate decisions are implied by earlier frozen conditions. Seven paths
are completed before publication stops; the next path has a 169,710,661-character
body and 446,041,800 guard characters. This remains an incomplete coordinate.
The different traversal prevents comparing path counts as a speedup or a
percentage of the full input domain. No complete file is admitted at either
32 MiB or 96 MiB. Resource limits never authorize omitting inputs or paths.

Nineteen coherent helper tests pass, including a new native emitted-file test
on all 30,722 Half values in [-1,1] with both zero signs. The test has two
conditions on equal squares and proves they form two paths instead of four.
Identity-key tests retain differing casts, constant types and operation order.
Existing normalization and propagation native checks continue with zero errors.

Fresh emitted files retain zero Torch CPU bit differences on 66,049 balanced
input pairs and all sixteen near-zero pairs. A new mixed-scale region has four
IEEE values for X1 in [-2^-24,2^-24], and 63 positive Half values for X2 in
[2^-23,2^-18]; `mixed.expr` is fully substituted and matches all 252 pairs.
These are actual one-token coordinate-2 files, not full-domain or multiple-token
proofs. Their hashes and live compiler identities are checked by `record.py`.

`inspect.log` records the original investigation against `0ed82d3`; `after.log`
records the selected dependencies after the fix. The latter shows the final
inverse producer 34 reducing to earlier producer 17 in the zero-MLP prefix.
No runtime activation cache, checkpoint dependency or lookup is introduced.
Final expression files retain only original inputs and incorporated constants.
JSON files are evidence metadata; expressions remain mathematical strings.

The test map preserves historical records and appends this validation. The
full coordinate, multiple-token parity and full vector remain outstanding.
Next work must simplify the newly reached large projection guards and numerical
dependencies, rather than count the eliminated contradictory paths as completion.
