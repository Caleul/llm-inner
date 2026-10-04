# Certified SiLU magnitude bound before dependency expansion

The compiler now tightens a stored-Half SiLU magnitude bound before
requesting the activation's dependencies. This changes proof arithmetic,
not the model's ordered operations or the emitted activation kernel.

For a finite Half bound `0 <= B <= 1/16`, the existing certified quartic
is `x * (.5 + x * (.25 - x*x/48))`, followed by F32 and Half storage.
Its inner coefficient is in `[0, 1/4]` under the original ordered F64
operations. Monotonic rounding then bounds its magnitude by
`B * (.5 + B/4)`. This endpoint is an exact F64 dyadic with at most 37
significant bits. The bound applies the original F32 **then** Half
storage. Omitting the F32 boundary is wrong at `B=2**-24`.

The proof is admitted only for exactly representable finite Half bounds.
Outside the polynomial certificate it retains the previous finite-Half
absolute-input bound. Unsupported values are rejected. No generic
real-number factorization, reassociation or activation approximation was
introduced.

`helpers/direct_sympy_layer_bounds_test.py` checks every one of the
22,530 signed Half inputs in the certified interval, and every magnitude
prefix, against both the native ordered quartic and vector PyTorch SiLU.
It checks endpoint exactness, the essential F32 boundary, invalid inputs
and fallback domains. Existing native ordered-projection and actual
checkpoint update tests remain: 3,555,328 and 888,832 cases, respectively.
The full 27-test direct-string integration gate and build passed.

The fixture's MLP update bounds changed from
`[1.341104507446289e-05, 1.5974044799804688e-05]` to
`[6.794929504394531e-06, 8.046627044677734e-06]`. Attention bounds and
the four large update-cell seed boxes are unchanged.

## Effective expression, not just expansion estimates

`diagnose.py` compares the preceding committed bound implementation with
the new implementation on exactly the same input rectangle:
`X1 in [1/32, 1]`, `X2 in [-65504, -64]`. Both discover checkpoint
properties and read weights through the normal streaming adapter.

The preceding version builds 25 producers and an estimated 6,581,987
expanded characters, then refuses to publish a complete flat expression
under the 1 MiB artifact budget. The new version builds 17 producers and
149,304 expanded characters before branch distribution, and emits four
flat arms with 171,977 characters, no compiler aliases, and only `X1`,
`X2` as input variables.

The [effective expression](../direct-sympy-tight-update-bounds/tightened.expr)
has SHA-256 `0ef677da44b5bd97219e4632f8e1003a1e16131bccd87b01256f03c18e1ae8ea`.
Its bytes already existed for a smaller admitted rectangle; they are
referenced once. Fresh compilation on the wider rectangle and 8,196
fresh checkpoint/native comparisons are recorded in `diagnosis.json`
and `witness-parity.json`: zero mismatches. The identical bytes are not
a reused numerical proof.

The single ordered old/new timing is recorded as diagnostic evidence,
not a controlled speed benchmark. No CUDA acceleration is claimed for
CPU SymPy proof arithmetic in this change.

## Compatible recovery and remaining scope

`state-compatibility.json` records an actual attempt to use the preceding
saved state: rejected as incompatible, with its manifest unchanged.
`recompile.py` resets all 100 completed leaves to pending using only
audited geometry. It recompiles under the full current numerical source,
backend and checkpoint identity; every admitted rectangle is independently
emitted, hashed and compared with the checkpoint before publication.

`validation.json`, `frontier.json` and `artifact-map.json` record the exact
new coverage, pending domain, effective expressions and fresh parity
counts. Failed broad rectangles stay pending. No limits omit inputs and
no region is counted twice. The original test map is preserved, with a
new `siluUpdateBoundValidation` entry.

The fresh run recovered all 100 formerly completed domains and admitted
174,097,408 additional input patterns. Coverage is 2,041,455,616 of
4,030,726,144 patterns (50.647341%); 1,989,270,528 remain pending.
There are 133 completed leaves and 21 distinct effective expressions.
The 78 successful fresh broad-region compilations performed 639,310
checkpoint/native comparisons with zero mismatches. Comparisons can
overlap across rectangles; this count is not unique-pair coverage.

This evidence concerns coordinate 2, position zero, one token and finite
Half embedding inputs. Native checkpoint corpora are sampled pairs;
they are not exhaustive coverage of all input pairs. Complete-domain
coordinate emission, multiple-token last-position parity, the remaining
coordinates and subsequent Rust generation are still unfinished.
