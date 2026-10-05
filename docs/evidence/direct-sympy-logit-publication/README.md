# Publish a coordinate only after exact parity and integrity checks

The controller previously wrote `logit-N.expr` before native parity finished.
Although the scheduling gate waited for parity, that final filename could expose
an unvalidated coordinate. Workers now write a pending candidate, execute native
readback against that candidate, compare the original emission/parity/current
file hashes, and recheck numerical source identity before atomic publication.
Final artifact/parity paths refer to the verified file. Failed parity, changed
bytes or a verifier crash leave no final or pending coordinate and release the
shared budget claim. Existing attempts are rejected before mutation.

Failed workers preserve producer/weight counters and actual branch statistics.
The parent retains their result record before stopping, rather than reporting
only a generic child failure. `artifactPublished` distinguishes an emitted
candidate from a validated, published output. Numerical expressions, reduction
order, branch contexts and the shared 512 MiB limit are unchanged.

The controller tests use real subprocesses/native parity for a disposable
checkpoint with four zero output rows. Fault injection then checks three
separate publication failures: bit mismatch, changed candidate after parity,
and an exception from the verifier. All three reject publication, release claims
and retain progress. This is control-flow validation, not original Llama parity.
The unmodified checkpoint's 256-byte smoke again dispatches only coordinate 2,
stops at its explicit budget, emits no final coordinate/vector and releases the
coordinate claim. A 64 KiB attempt reaches 25 producers and 3 generated candidates
(including the rejected candidate), retains actual branch statistics and likewise
emits no final file or other logit. Two targeted integration tests also exercise the original
checkpoint projection decision and its bit-exact numerical proof.

The previous 40-focused/620-portable test results remain historical entries in
the test map; this change reruns the affected integrations and real controller
paths. No whole-project result or full Llama coordinate completion is inferred
from those tests. `record.py` binds source hashes and real evidence.

## H100 allocation remains pending

After a fresh zero-session readback, Access Broker `colab_cli` requested H100
without `--high-mem`. Colab returned the same accelerator rejection for possible
quota/entitlement. A subsequent readback again showed zero sessions. The added
RAM flag was therefore not sufficient to explain the failure. No GPU/remote RAM
resources or CUDA execution are claimed. No alternate accelerator was allocated.

A fresh deterministic source archive has its own path/hash in
`source-archive.json`; the previous archive/evidence is preserved. For a later
allocated session, upload this fresh archive and manifest under the remote names
used by the earlier launch script, alongside the preparation helper. Verify and
observe the existing launch marker before any dispatch retry. The real 512 MiB
attempt, complete original output coordinate/vector and multiple-token parity
remain pending.
