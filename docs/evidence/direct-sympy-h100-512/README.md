# Shared 512 MiB output budget and memory-admitted independent logits

`direct_sympy_logits_run.py` is the Colab CPU/CUDA controller. Its default output
budget is 536,870,912 bytes, shared across expressions and conditions of every
coordinate and the final Tuple syntax. An interprocess-locked ledger reserves
before expanding an arm; failed attempts release their own claims and cannot
publish truncated expressions. Completed coordinates retain their actual sizes.
Compiler numerical expressions and dependency order are unchanged by accounting.

RAM is separate: the controller uses at most 80% of currently available RAM,
optionally reduced by `--memory-mib`. It reserves 768 MiB plus 16 times the CAS
limit per worker and measures aggregate parent/descendant RSS every 200 ms.
Compilation workers have separate CAS, proof and branch contexts. Their original
reduction trees are unchanged. A fresh output directory is required; no saved
numerical state is reused. Progress files expose generated paths, dependencies,
claimed characters, elapsed time and individual branch statistics.

The arbitrary first coordinate (default 2) must finish and pass native CPU
readback parity before remaining coordinates are dispatched concurrently. CUDA
runs only the existing bit-audited compatible numerical batch suite, with TF32
and nondeterministic algorithms disabled. `factor()` and `simplify()` remain CPU
operations. GPU numerical mismatches stop the job. Resource limits terminate
unfinished work without treating a partial coordinate/vector as successful.

## H100 allocation evidence

Access Broker discovered `colab/colab_cli` and the authenticated personal account.
The official CLI recognizes H100. Its actual allocation request with `--gpu H100
--high-mem` was rejected by the Colab backend for possible quota/entitlement.
Readback then confirmed zero active sessions, 6.48 units balance and zero current
usage. Therefore no remote CPU count, RAM, GPU/VRAM or CUDA execution is claimed.
`access-broker-observation.json` records terminal operation IDs and the error.
No alternate GPU was allocated; the requested H100 execution is pending.

`package_sources.py` produces a deterministic 300 KiB-class archive with current
Python sources and the real checkpoint. Upload it, `source-archive.json` (as
`/content/llm-inner-h100-512-source-archive.json`) and
`helpers/direct_sympy_colab_logits_prepare.py` through `colab_cli`, then execute
`launch_colab.py`. The remote preparation checks the upload hash, installs pinned
libraries, confirms the requested GPU/CPU/RAM resources, runs numerical preflight
and dispatches the controller. The launch marker prevents accidental replay;
use `observe_colab.py` to inspect real state/liveness before any retry.
These scripts were syntax-checked locally; remote preparation was not executed.

## Verified scope

Three budget tests prove atomic concurrent admission, failed-attempt isolation,
final condition accounting and byte-identical checkpoint RMS output with/without
the shared lease. Its native normalization parity remains independently checked
by all 22 coherent-path tests, including 888,832 real checkpoint comparisons.
Two controller tests check missing-H100 rejection and real subprocess scheduling.
The latter uses a disposable checkpoint with four zero output rows to test dead
dependency elimination and admission: four complete coordinates, two live workers,
zero native bit mismatches. It does not replace the original Llama fixture.
A separate 256-byte attempt against the unmodified checkpoint proves that failure
of coordinate 2 dispatches no remaining logits and publishes no final artifact.

All 40 focused integrations pass. The portable suite has 620 tests: 571 pass,
zero fail, 49 explicit skips. Historical/pinned native prerequisites remain in
the existing test map. Controller tests were repeated after failure cleanup.
`validation.json` binds source hashes, test logs and observed scope. The complete
original Llama coordinate, full output vector and multiple-token parity are still
unfinished. The actual 512 MiB production attempt and GPU work await allocation.
