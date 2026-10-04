Persistent continuation of the incomplete first output coordinate

The preceding goal turn made progress: af2f605 reduced the saved gated
expression to 7,273,880,362 characters and verified its position-zero MLP
producer parity. The complete coordinate remains absent.

The rejected next hidden envelope is 12,951,847,412 characters. Its five
activation/up occurrences belong to rounding conditions and separate branch
results; this accounting does not itself prove that all copies are removable.
No numerical rewrite or input-domain narrowing is introduced in this turn.

helpers/direct_sympy_checkpoint_run.py adds a controller that validates the
normal savepoint identity before reuse, persists every completed producer and
reports restored, completed and persisted counts separately. It supports a
character cap, wall-clock cap and optional process address-space cap. It
catches allocation failure as an incomplete run. The existing diagnostic
observer remains read-only when resuming. The new controller deliberately
persists newly completed producers; it is not a model runtime.

Local real-checkpoint test:
- A fresh 1-MiB run stops with 10 actual persisted producers.
- Resume with a changed output dimension is rejected without changing the
  original manifest.
- Compatible resume with 4 MiB restores those 10 records and persists the
  eleventh; the entire earlier record prefix stays identical.
- Both reports keep coordinateComplete and parityVerified false.

Test map:
- helpers/direct_sympy_checkpoint_run_test.py exercises those real process
  stops/resumes and identity rejection.
- test/direct-sympy-strings.test.ts includes this test with the existing
  numerical gates; configured integration passes 10/10 with no skips.
- Numerical sources are unchanged; the previous 603-test broad baseline
  remains separately recorded under direct-sympy-rms-guard. It was not rerun
  for this execution-controller addition.

Colab continuation:
- Session: llm-inner-rms-guard, A100 40 GB, 12 CPUs and about 83.5 GiB RAM.
- New directory: /content/llm-inner-rms-resume16.
- Numerical source files and initial state come from the verified af2f605
  snapshot; compiler identity is checked normally, never rewritten.
- The state copy initially contains the same 18 producers. Immutable objects
  are hard linked; subsequent manifest replacement is atomic in the new
  directory. The original baseline is retained.
- The frozen controller started with 16 GiB of expression characters, 72 GiB
  process address space and 600 seconds. It differs from the final controller
  only in reporting/limit-cleanup and low-cap admission details, with no numerical-source change.
- The runner invokes actual saved-producer parity after the compilation exits.
  Parity completion and any new saved hidden producer must be read back before
  reporting further architecture progress.

The observation JSON records a live process and the actual current frontier,
not a completed-coordinate artifact. Local backup from the preceding run:
artifacts/direct-sympy-rms-colab/backup.tar (18 producers, 31 verified objects).

The final controller's two Python tests also ran on Linux in a separate
Colab directory, with the same frozen numerical sources: 2/2 passed in
34.817 seconds, including low-cap admission and actual checkpoint resume.
On macOS the Linux-specific admission test is explicitly skipped; the
actual checkpoint continuation test passes. The final named Node integration
was rerun after this addition and passes 1/1; the other nine integrations
had already passed with unchanged numerical sources.

The 16-GiB continuation restored all 18 producers with compatible identity,
then actually saved model.layers.0.hidden:0 as producer 19. Its expression
has 12,951,847,412 characters and no separate persisted sign projection.
A later allocation hit the process address-space limit; the compilation
stopped after 531.094 seconds with 66,997,841,920 bytes peak RSS. This is
one further architecture producer, not a complete coordinate. Saved-producer
parity was confirmed live afterward; its result is still pending in the
observation captured here.

A lossless delta archive contains the new hidden object, fresh 19-record
frontier and frozen continuation controller. It relies on the earlier local
verified 18-record base archive. Export verifies the source identity and the
entire previous record prefix unchanged; it does not rewrite identity.

Final saved-producer parity completed after that observation: 60 matrices,
420 bitwise comparisons, zero mismatches, with all 19 actual files evaluated.
This includes hidden:0 at position zero; hidden:1, final normalization and
output are not complete. The delta archive was downloaded locally and its
new 12,951,847,412-character object was verified by streaming its entire
content through SHA-256. Source identity and the previous 18 records match
the already verified base archive. No compiler job remains live in this run.

An additional request for full final logs later hit a kernel metadata
ReadTimeout (Broker operation 20522f76-bd02-46f1-a0e4-851c37dd5781).
The terminal runner state, zero-mismatch parity summary, hidden digest and
local archives had already been read and verified in operation
20bc65c0-f403-4443-8255-db2521cdec36. The failed optional read does not
invalidate those retained observations or require recompiling the state.
