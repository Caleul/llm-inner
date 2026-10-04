Unsigned-word polynomial factorization with SymPy

StringCompiler now opens U64Add/U64Mul polynomial regions after every
substitution, before the next dependency. SymPy factor -> simplify -> factor
runs on typed integer leaves. Decoding restores every leaf and reduces
integer coefficients modulo 2^64. No temporary WordFactor symbols survive.
A shorter representation is accepted; 128-operation and 1,024-estimated-term
search budgets decline larger candidates without cutting paths or changing their expression.

Proof scope: integer polynomial identities are valid in Z/(2^64), including
intermediate overflow. Masks, shifts, floating expressions and decisions
remain opaque leaves. A missing word-width proof rejects the candidate.
Floating reductions and rounding boundaries keep their original order.
The rule also simplifies guards; each branch still receives its own normal
CAS/condition context. Events distinguish actual contractions from formatting.

Tests/evidence:
- helpers/direct_sympy_word_factor_test.py: 5 tests; 278,568 exact comparisons
  across all 65,536 F16 bit patterns, F64 boundaries and seeded raw F64 inputs.
  Covers overflow, repeated symbolic coefficients, mandatory CAS calls,
  branch conditions/contexts, fixed points, unsigned proof rejection and
  unchanged floating operation trees and rejection of an exponentially large
  polynomial before calling CAS.
- The preceding ten configured integrations passed after the numeric-source
  edits, including 25,167,601 sqrt cases and full conversion/SiLU/sign gates.
  The new named integration then passed after the TypeScript build.
  existing-ten-integrations.log and new-named-integration.log are separate
  runs, not an invented single 11-test run.
- Colab Linux: both controller tests and the initial four word tests passed. The final five-test gate is recorded
  separately after adding the expansion search guard.
- helpers/direct_sympy_word_factor_cuda.py runs factor/simplify on CPU and
  modular numeric comparisons in bounded int64 CUDA batches. The test buffers
  are raw words, not checkpoint activations; CPU forward remains the parity
  reference. CUDA arithmetic proof is not complete-coordinate parity.

Real checkpoint comparison before the final no-CAS search guard (same fresh
input/checkpoint, CPU/platform,
128-MiB character and 24-GiB process-address-space caps):
- Old: 27.904810 seconds, peak RSS 1,180,028,928 bytes, 14 saved producers.
- New: 28.395179 seconds, peak RSS 1,180,176,384 bytes, 14 saved producers.
- Producer character sizes are identical. New word-factor contractions: 0.
- Both actual saved-file parity runs: 60 cases, 120 comparisons, 0 mismatches.
- Both stop before the same gate projection substitution: 177,411,604
  estimated characters exceeds the 134,217,728-character cap.
This sampled prefix shows no speed, memory or expansion gain. It does not
measure later producers or establish an ETA/full-model ratio.

Changing direct_sympy_words.py/direct_sympy_strings.py changes numerical
source identity. The new run rejected the old 20-record manifest before
loading expressions, verified no mutation, and compiled fresh. The legacy
20-record backup remains separately verified against the previous sources.
Do not rewrite its identity to silently resume it under these new sources.

The complete output coordinate/vector is still missing. The last actual
legacy continuation reached hidden:1 then exhausted address space at final
RMS squaring. The added modular rule is useful when repeated integer terms
exist but did not reduce the measured checkpoint prefix. Remaining work
must address repeated literal materialization and the composition of the
closed numerical regions while retaining fully substituted final output,
all input paths and bit-for-bit parity. No alias/DAG/runtime checkpoint or
cached activation may be delivered as the compiled artifact.
