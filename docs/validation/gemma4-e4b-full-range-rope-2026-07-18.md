# Gemma 4 E4B full-range RoPE candidate

## Result

The literal Gemma 4 text program now declares and executes the complete
`Sleef_sinf4_u10advsimd` / `Sleef_cosf4_u10advsimd` binary32 argument range.
The previous fail-closed boundary at `abs(angle) >= 125` is gone. One
source-selected contract applies to all 66 compatible RoPE assignments; it is
not keyed by layer or assignment ID.

The transcript follows PyTorch source commit
`7269437d655783a26cba32aa88195b741ff496aa` and bundled SLEEF commit
`5a1d179df9cf652951b59010a2d2075372d67f68`. The artifact embeds the first 416
little-endian F32 entries needed from `Sleef_rempitabsp`; the 1,664 decoded
bytes have SHA-256
`9a623b9ff705f726ddb129e4b1c3c0311ac19b86a30667cdd395e6ea98d8c5c5`.
The declared calculation includes the fast-range threshold, full `rempif`
double-float reduction, quadrant/sign rules, ADVSIMD FMA polynomial order and
all BF16 RoPE casts.

The immutable model remains `google/gemma-4-E4B` revision
`411aa17b749aa952df1359d2dcea73917a544d9a`; `model.safetensors` SHA-256 is
`43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651` and
`config.json` SHA-256 is
`f27a045f32c39fb9cd930204920de6b0962810cf09929a8810901ffaec780f20`.

## Artifact and source-removed evidence

The regenerated artifact is
`artifacts/gemma4-e4b-dense.literal.json`, 21,327,700,723 bytes, SHA-256
`95a81a3d5a1969f62608a3a2d537d6b930de8bae9ebbef0130826186fe41efd8`.
It contains 2,130 embedded constants. With `gemma-4-E4B-dense` renamed away,
payload verification decoded all 15,992,314,836 bytes and retained literal
storage SHA-256
`e21b49734b3945d760d4f630e747e71f6510826f45bbb0159fb59abef005af98`.

The source-removed scalar view for `layer_0_q_rope[0,0,0,0]` exposes:

```text
angle = F32(position_ids[0,0] / F32(F32(10000^0) * F32(1)))
range_reduction = abs(angle) < 125 ? SLEEF_CODY_WAITE_F32(angle) :
  SLEEF_REMPIF_F32(angle, inline_f32_le_table_sha256=9a623b9f...d8c5c5)
trig_polynomial = ADVSIMD FMA binary32 in declaration order;
  coefficients_ascending=[-0.16666659712791443,0.00833307858556509,
  -0.00019810690719168633,0.0000026083159809786594]
cosine = BF16(SLEEF_COS_F32(angle))
sine = BF16(SLEEF_SIN_F32(angle))
```

The complete scalar-view report SHA-256 is
`33c08d4a20491e1c34a0f3755bc23d276eaf22ef61b7f21ad1538e3c08a0c2cd`.

## Authoritative differential validation

The installed PyTorch 2.12.1 ARM CPU vector kernels supplied bitwise F32 oracle
values at `125`, `-125`, `126`, `255`, `256`, `1000`, `4096`, `8192`,
`32768`, `65536`, `100000`, `1000000`, `1e10` and the largest tested finite
F32 magnitude. Unit tests reproduce every sine/cosine value exactly and verify
the resulting BF16 RoPE at position 125.

A fresh authoritative full-assignment trace forced the first learned RoPE pair
through `rempif`:

```bash
npm run capture:gemma4-operation-checkpoints -- \
  --source ./gemma-4-E4B-dense \
  --output /tmp/gemma4-loop46-position125-operation-checkpoints.json \
  --input-tokens 184 --position-ids 125 \
  --python ./venv/bin/python --model google/gemma-4-E4B \
  --revision 411aa17b749aa952df1359d2dcea73917a544d9a --device cpu

# after renaming ./gemma-4-E4B-dense away
node dist/src/gemma4-paged-text-operation-differential-cli.js \
  --artifact ./artifacts/gemma4-e4b-dense.literal.json \
  --trace /tmp/gemma4-loop46-position125-operation-checkpoints.json \
  --report /tmp/gemma4-loop46-position125-operation-report.json \
  --max-read-mib 16 --assert-source-unavailable ./gemma-4-E4B-dense \
  --absolute-tolerance 0 --relative-tolerance 0 --allow-unverified-fidelity
```

All 1,229 assignments, terminal logits and 24 producer KV key/value entries
passed with maximum absolute and relative error 0; cosine similarity, top-k
overlap and argmax agreement were 1; `firstDivergentOperation` was null;
`sourceCheckpointAccessed` was false. The trace/report SHA-256 values are
`943083042e9f014694173de67ff96901dd69c040aae3728d29c679d05869581f` and
`01f1d39a78e4b0f9aafa3de5404cedb69144902c46c85263fbb08c4cac2d80ff`.

## Independent review of cached text generation

This loop independently inspected loop 45's implementation and captured a new
four-step authoritative trace before replaying the regenerated artifact with
the checkpoint absent. Tokens `184`, `3910`, `531`, `974`, every selection
logit tensor, all 96 per-step producer cache snapshots, terminal logits and all
24 terminal producer caches passed at zero tolerance. `firstDivergence` was
null and `sourceCheckpointAccessed` was false. This independently accepts the
prior loop's dense-text cached-generation claim. The new trace/report SHA-256
values are
`5a117e47f463d80678e9cf701ae5d5e79572fef7e654ec82991ab690cfac6055` and
`3c45372f7eac6ee5042c723bbe4227d32d397c3589508d547eac926a0ccc3edb`.

## Acceptance boundary

This loop implemented the full-range RoPE boundary and therefore records that
new evidence as a candidate for independent review. The Gemma checkpoint
marker remains prohibited: real image, video and audio tower execution has not
been differentially compared source-removed against the authoritative runtime.
