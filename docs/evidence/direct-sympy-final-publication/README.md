# Exact-body grouping and final publication gate

The parallel controller previously called the ordinary per-leaf combiner
after completing its input tree. That duplicated identical region functions
and published the combined expression without checking its actual numerical
output. It now uses the existing exact-byte coalescer, keeping domain guards
ahead of every numerical guard, and validates the temporary combined file
against the CPU checkpoint reference before publishing it. Hash, numerical
source identity and manifest checks prevent publishing a changed candidate.
Emission, parity or integrity rejection preserves the prior final file and
manifest. No runtime intermediate, response lookup or numerical reassociation
is introduced.

The native validator currently covers **position zero, one token**. Its
successful corpus validation is recorded separately from `finalParity`, which
remains false until the broader contract is proved. The current real input
tree is incomplete; **no full-coordinate expression was emitted here**.

The compatible continuation ran 24 attempts with four workers and a 6 GiB
memory ceiling. Thirteen regions completed, adding 1,064,960 input patterns;
3,380 fresh native comparisons had zero mismatches. The saved tree now has
89 completed leaves covering 1,728,946,176 of 4,030,726,144 admitted Half-input
pairs (42.894%). All 76 previously completed leaves remain unchanged. The
largest measured compiler RSS was below its configured memory ceiling.

These 89 completed regions contain 21 distinct expression files. Repeating
their bodies per region accounts for 17,514,037 characters; their distinct
bodies total 3,764,162. This is **13,749,875 repeated body characters**, not a
measurement of the missing final artifact or its guard overhead. Every
actual expression is linked in [artifact-map.json](artifact-map.json); existing
expression files are referenced rather than copied again.

Build and the full 26-test integration suite passed, zero skipped. The
seven controller tests were rerun after the final reporting changes. New
cases check grouping and admission order, rejection of a wrong output using
the actual checkpoint, and candidate mutation during parity. The success
publication-order fixture mocks the checkpoint validator; it is not evidence
that the real coordinate is complete. Existing native coalescing tests cover
126,976 Half/zero-sign cases and lazy domain guards.

[record.py](record.py) checks source compatibility, unchanged old leaves,
complete partition geometry, artifact hashes, native reports and test logs,
then preserves the evidence and adds `coalescedPublicationValidation` to the
existing test map. The continuation used the preceding controller commit
`9bbdc24`; the new publication gate does not change numerical source identity.

The Colab inventory was refreshed through Access Broker operation
`5364318f-a482-4b93-9487-aa934458a28b`: no active sessions, terminal exit 0.
This continuation did not execute a remote job or measure a CUDA speedup.

Remaining requirements: finish the admitted domain for the first coordinate,
produce and validate its effective expression, extend last-token compilation
to multiple tokens, then compile the remaining output coordinates. The latest
human representation instruction uses SymPy mathematical strings; JSON here
contains saved-state and validation metadata only.
