Consume private projection terms without changing arithmetic

The v4 continuation restored 19 completed producers, then stopped with no new
producer after 324.507 seconds and 71,297,449,984 bytes peak RSS. Its diagnostic
pinpoints projection reduction -> op -> substitute -> stabilize_budget_envelope
-> re.sub, while materializing the next hidden producer. Saved-file parity
completed on all 19 files: 60 cases, 420 comparisons, zero mismatches. This is
not a complete coordinate. The numerical source identity is unchanged.

The optional v5 backend consumes only the private terms list created by linear.
After each original lane fold, the consumed product reference is cleared. The
same four lanes are combined as (0+1)+(2+3), releasing each consumed pair.
Parentheses, substitution, factor/simplify and exact numerical boundaries run
in the same order. Public reduction callers and the parallel-budget route
retain their existing behavior. The final R16 syntax is joined in one string
allocation, avoiding a second giant parenthesized copy on Python 3.13.

Test map:
- direct_sympy_scan_backend_test.py: 13 tests. Widths 0,1,2,3,4,5,8,9,17
  have byte-identical strings and ordered operation traces. Weak references
  prove consumed products are released before the final fold. A real SymPy
  five-column projection keeps every expression, CAS/substitution event and
  numerical closure byte-identical. Nested installation restores methods.
- The allocation regression uses a 16-MiB admitted-shaped literal. It measures
  temporary string ownership, not full-model RSS or final expression size.
- test/direct-sympy-strings.test.ts: the configured ten integrations passed;
  the final scanner named integration also passed after the additional
  allocation test was added. The final join-only correction passed all 13
  Python scanner tests without numerical-source changes.
- Earlier broad regression counts remain recorded separately; no new broad
  run was necessary for the allocation-only change.

The initial frozen Colab trial halted at its allocation gate, before model
compilation. Python 3.13 retained a concatenation temporary that reduced the
micro peak by only 25%, below the required 35% improvement. The memory gate
was not weakened: single-allocation joining removes that extra temporary.
This failure and its trace remain in colab-allocation-gate-failure.json.

A corrected frozen trial uses /content/llm-inner-consuming-reduction-v2,
starting from the same 19 records and unchanged numerical source identity.
It runs scanner and Linux controller gates, then strict state restore and
bounded compilation, followed by actual saved-file parity. Limits remain
16 GiB expression characters, 72 GiB process address space and 600 seconds.
No new producer or complete-coordinate result is claimed until read back.

The corrected Colab gate passed all 13 tests. Its allocation peaks are
67,109,596 versus 33,555,001 bytes, with identical output. The Linux controller
gate also exited zero. The bounded real compiler restored 19 records and persisted hidden:1 as
record 20. It stopped after 541.385 seconds at 70,675,308,544 bytes peak RSS
while substituting a finite-Half square in final RMS normalization. The v4
run saved no new producer. These different terminal frontiers do not support
a speedup ratio or an ETA for a complete coordinate. A Colab observation lost its connection while parity was live. Re-reading
the same session confirmed terminal parity: 60 cases, 480 comparisons, zero
mismatches across all 20 saved files, including hidden:1. Both processes
are now absent. terminal-parity.json records the final readback. This
validates position-zero producers only, not a full output coordinate or
the last-token output at variable sequence lengths.

The new hidden:1 mathematical string has 12,951,847,416 characters. Its
291,891,200-byte delta archive was downloaded through the Broker. The local
archive hash, manifest integrity, unchanged 19-record prefix and entire
uncompressed expression digest were verified without materializing the
13-GB string. hidden1-backup.json records the checks. The base and earlier
hidden:0 delta archives are still required to reconstruct all 20 records.

The controller now records retained producer/region characters, last
substitution/budget envelope and Linux address-space/resident usage at
restore, persistence and failure. Shared literal counts are not independent
allocations. Numerical sources and strict savepoint identity are unchanged.
The real resume/rejection test passed locally (one Linux-only skip), both
controller tests passed on Colab Linux, the named Node integration passed
and the TypeScript build passed. Their logs are mapped here.

Next unresolved expansion is final:mean, not another output coordinate.
Any simplification changes must keep dtype/order/sign-zero proofs and the
per-substitution CAS fixed point. Reducing temporary ownership alone has
not reduced the 13-GB saved expressions. Increasing resource caps alone
does not address duplication in the final composition. The final artifact
and position-dependent next-token coordinate remain unverified.
