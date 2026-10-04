Producer envelope allocation, without numerical changes

The direct compiler still incorporates weights, substitutes dependencies and
runs factor/simplify to a fixed point before proceeding. Bits belong only to
explicit numerical primitives needed for exact dtype/rounding semantics.
They do not replace checkpoint compilation as the objective.

A producer previously allocated '(' + the whole completed expression + ')'
before compacting already certified call literals for SymPy. At many gigabytes,
this creates a large temporary copy with no numerical benefit. The equivalent
scan backend now compacts the exact same admitted literals first, adds the
parentheses to the compact envelope, runs the existing stabilization and
restores all literals. Input-refining branch contexts retain the original path.
No aliases remain in returned expressions. Numerical identity is not rewritten.

Test map:
- direct_sympy_scan_backend_test.py: nine tests, including allocation peaks,
  byte-identical expressions, input-domain guards, original budget rejection,
  nested installation/restoration and unchanged old savepoint identity.
- test/direct-sympy-strings.test.ts: ten configured integrations passed, without
  skips, after npm run build. Log: focused-tests.log.
- The 16-MiB admitted-literal regression measured about 33.6 MB peak allocation
  for the old wrapper and 16.8 MB for the compact wrapper (tracemalloc), with
  byte-identical output. This is a temporary-allocation test, not a full-model
  RSS or expression-size improvement.

Colab: existing A100 session llm-inner-rms-guard; new frozen directory
/content/llm-inner-producer-envelope. Nine backend tests passed on Linux,
including the same allocation and byte-equivalence checks. A bounded run is
restoring the verified 19-producer state with 16 GiB of expression characters,
72 GiB process address space and 600 seconds. The original 19-record manifest
is preserved. Actual saved-producer parity runs after compilation stops.

The initial observation captures a live process and the existing 19-record
frontier, not successful restore or new architecture progress. Full-coordinate
compilation and parity remain incomplete. The previous run saved hidden:0,
then reached the memory cap; its 60-case, 420-comparison producer parity is
already separately verified. The real effect of this allocation change is
pending terminal measurements and the new parity result.

The v3 compilation then completed its bounded attempt: compatible restore of
19 records, zero new producers, 324.760 seconds, 70,676,361,216 bytes peak RSS,
and an address-space allocation failure. Saved-file parity was still running
in the retained observation. This is not an acceleration result: it starts
from 19 producers, unlike the older 18-to-19 attempt, so their total times are
not a comparable sequential/parallel speed ratio.

The v4 backend additionally releases the producer's original input string
immediately after stabilization. Only its recorded character count is needed
later. A weak-reference regression proves the old producer still retained the
source during numerical closure while the new one has released it; returned
expressions and all producer/CAS events are exactly identical. All ten focused
integrations passed again. Numeric-source identity is unchanged.

The execution controller now records at most 12 file/function/line frames and
separates compilation failure from restore/setup failure. It records no input,
weight, expression text or frame locals. The real 1-MiB budget test identified
substitute() as the final failing frame; the changed-dimension test still
rejects restore before mutation. Local controller tests pass with the existing
Linux-only test explicitly skipped on macOS. These diagnostics do not alter
model semantics or claim any new saved architecture producer.

The v3 saved-file parity then completed: 60 cases, 420 bitwise comparisons,
zero mismatches, all 19 files evaluated. Its runner is terminal with no
remaining processes. The next frozen v4 run uses the same 19-record prefix
and budgets, after backend and Linux controller gates pass.

The frozen v4 runner is live in /content/llm-inner-producer-lifetime. All ten
backend tests passed on Linux and the controller gate exited zero. The actual
compile process is restoring the same 19-record state; no new producer or
complete-coordinate parity is claimed in this observation.
