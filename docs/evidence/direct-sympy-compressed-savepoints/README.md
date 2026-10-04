# Lossless streaming producer states

The certified product producer from the previous compilation was actually emitted and validated against Torch: 60 input matrices, 360 comparisons across 16 producer files, zero bit differences. Position-zero producer parity does not establish the complete output coordinate or variable-length last-token parity. That compilation ended on its time budget after 660.31 seconds; a captured stack showed save_expression inside disk synchronization.

The opt-in --compressed-savepoints flag writes producer strings through 1-MiB blocks with gzip level 1. SHA256 remains defined over the original UTF8 expression. Schema 4 records the encoding and all existing source/checkpoint/domain/dtype/reference-platform identities. Restore checks metadata size before opening/decompressing objects, limits decoding to the declared length, verifies content and CRC, validates every record, and mutates the model only after all checks pass.

Immutable identical strings share their storage object across producers. Each producer/session still supplies its own numerical bounds and dtype proofs. This does not introduce runtime caches, aliases, an interpreter or a checkpoint dependency in emitted expressions.

97 Python tests and 8 Node integrations passed; the build passed. Seven savepoint tests cover identical resumed expressions, full finite-Half native parity, lossless compression, cross-producer object reuse, corruption, size bounds, encoding mismatch, incompatible states and writer exclusivity. The earlier wide regression evidence remains mapped; it was not rerun for this storage-only change.

The real 1,316,872,194-byte gate producer occupies 28,011,357 compressed bytes with the identical SHA256. On local disk, save+fsync took 0.93 seconds raw and 1.80 seconds compressed. This proves reduced storage, not a local speedup. Workflow peak RSS was 2,871,820,288 bytes, including loading the large producer. Colab speedup is not established.

The validated gated expression has 16,131,686,432 characters and occupies 343,176,312 gzip bytes. Its local artifact and SHA256 are recorded under ../direct-sympy-square-composition/local-gated-artifact.json. Large producer files are kept outside Git. The mathematical expression remains large and the full coordinate remains incomplete.

The original schema-3 state is preserved. It is not silently rewritten into schema 4. Resuming with its frozen compiler retains strict original identities; the numerical helper hashes and adapter AST are checked against the current compiler before launching that continuation.
