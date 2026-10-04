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
gate also exited zero. The bounded real compiler is now live, restoring the
19-record state; no new saved producer has yet been observed.
