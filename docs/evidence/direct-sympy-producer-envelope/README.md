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
