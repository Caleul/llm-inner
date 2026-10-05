# Bound exact guard identity reuse during symbolic compilation

OrderedGuardPruner already eliminates conditions implied by previous original
ordered arms using SymPy propositions/SAT. Investigation found repeated textual
expansion in its identity check: frozen prefix guards are reused by many leaves,
but previously each visit streamed their expanded bytes again for SHA-256 and
literal comparison. Large numerical producer bodies make this work expensive.

The compiler now reuses the established atom only when the entire expression
string and immutable definition tuple are equal. Those inputs determine every
expanded byte. Dictionary keys compare actual string/tuple values; digest
collisions or object identifiers do not establish identity. Different snapshots
retain the existing digest candidate selection plus complete expanded-byte
comparison. Reassigning a view's definitions changes its key, as verified by the
mutation regression test, including a forced digest collision. Stored digest
witnesses now retain a shallow copy of their view so reassignment cannot alter
the definitions that were originally hashed. Lazy runtime guard order and numerical bodies remain
unchanged. No numerical predicate equivalence is newly assumed.

The cache retains references, not expanded strings or copied numerical values.
Its conservative charge counts expression plus all definition characters,
including shared references, and is bounded by 8 MiB with LRU eviction. A zero
cache budget disables it; an oversized snapshot takes the original route.
The independent 512 MiB final expression/condition ledger is unchanged; aggregate
worker RSS remains separately monitored. This is a compiler-only proof cache.

The actual first 27 checkpoint candidate arms were each pruned with the cache
on and off, alternating call order. Every retained Guard tuple matched exactly.
The initial pruning elapsed sums were 14.283064s uncached and 4.372451s cached
(about 3.27x faster for this operation). The final run is repeated after freezing
identity witnesses, with its measurements recorded in current-arms.json.
The initial run had 220 hits and a 645,904-character charge under 8 MiB. This compares
one compiler operation on identical arms in the same run, not whole-compiler
CPU/GPU speed. The measured arm sizes and ordered skeletons match historical
baseline measurements exactly.

A fresh checkpoint coordinate in [1/32,5/128]^2 is byte-identical to its freshly
compiled pre-change baseline: 217,033 characters, two flat arms, zero compiler
aliases. All 66,049 Half input pairs match the original checkpoint bitwise.
This proves a regional coordinate only. It does not prove whole-domain,
multiple-token or output-vector compilation. Those require their final artifacts.

Regression evidence includes scoped native guard/coherent-path tests, actual
checkpoint integration, budget/publication-gate tests and the full portable
suite. record.py reconciles counts, source/checkpoint/backend hashes, actual
artifact bytes and the fresh bounded full-domain attempt. It appends to the test
map without replacing historical records. Expressions remain mathematical
strings; JSON here contains execution evidence. Final goal completion is unmet.


Initial observations are explicitly retained with the initial- filename prefix.
After reviewing collision/invalidation behavior, the compiler was additionally
changed to freeze prior identity witnesses. All required current measurements,
checkpoint parity and regressions are rerun; their unprefixed files bind the
final source identity. No initial numerical compiler state is reused.

## Final-source measurements and regression status

The final snapshot-safe operation comparison measured uncached
14.612588s and cached
4.394963s on the same 27 arms; retained Guards,
body sizes and ordered skeletons are identical. Cache hits were 220 and final
charged characters 645,904. initial-paths-source.py is a recovered initial
prototype whose SHA-256 is verified against initial-smoke.json; its observations
are historical and do not replace any final-source validation.

The fresh final-source whole-domain worker took
137.246154s, resolved 25 producers and generated 28 candidates
including the rejected final candidate, then stopped at the same 512 MiB cap.
It published no final coordinate and did not dispatch further logits. Observed
aggregate RSS peaked at 579,043,328 bytes under separately admitted
10,904,915,148 bytes. Elapsed time overlapped diagnostics/tests and
is not a controlled whole-compiler speed comparison.

All three affected checkpoint integrations passed. The standalone native suite
passed 21 tests and skipped its one optional checkpoint test; the wrapper exercised
that test with the checkpoint enabled. The portable suite passed 571 tests, zero
failures and 49 existing explicit skips. These final-source checks were all rerun
after freezing prior identity witnesses. Whole-domain coordinate, multiple-token
and whole-vector artifacts/parity remain incomplete.
