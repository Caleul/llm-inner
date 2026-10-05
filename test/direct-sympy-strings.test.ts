import test from 'node:test';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const python=process.env.LLM_INNER_DIRECT_PYTHON;
test('Signed normal-binade conversions contain one source occurrence and preserve native IEEE boundaries',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_fixed_grid_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Fixed-grid F32 midpoint parity: cases=50331648 mismatches=0/);
    assert.match(stdout,/Fixed-grid Half midpoint parity: cases=184314 mismatches=0/);
    assert.match(stdout,/Fixed-grid scaled F32 parity: cases=\d+ mismatches=0/);
  });
test('Certified constant projections avoid dead producers and preserve signed Half outcomes',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_early_projection_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Early constant V projection: pairs=8421376 mismatches=0/);
  });
test('Tight Half update bounds preserve ordered F32 reductions and eliminate dependencies with proof',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_layer_bounds_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 10 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Correlated RMS linear bounds: cases=5332992 violations=0/);
    assert.match(stdout,/Correlated RMS projection bounds: cases=5332992 violations=0/);
    assert.match(stdout,/Composed projection bounds: cases=7110656 violations=0/);
    assert.match(stdout,/Tight projection bounds: cases=3555328 violations=0/);
    assert.match(stdout,/Tight checkpoint update bounds: cases=888832 violations=0/);
    assert.match(stdout,/Certified SiLU magnitude bounds: cases=22530 violations=0/);
  });
test('Memory-admitted parallel regions preserve full geometry, ordered numerical bodies and native parity',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_partition_parallel_test.py'],{timeout:180_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 7 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Timeout retry proof: sameDomain=true deadlines=10,30 sizeFailurePreservesFullRoot=true/);
    assert.match(stdout,/Final publication proof: coalescedBody=true actualCheckpointRejectsWrongOutput=true atomic=true/);
    assert.match(stdout,/Parallel partition proof: workers=1,2 patterns=1006505988 identicalHashes=true mismatches=0/);
  });
test('Fresh broad-region covers refine only pending intersections and preserve exact source compatibility',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_cover_regions_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 3 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Cover intersection proof: vectorPatterns=324 added=48 lost=0 overlaps=0 signedZeroPatterns=4/);
    assert.match(stdout,/Fresh central cover parity: patterns=184571904 mismatches=0 compatibleResume=true finalParity=false/);
  });
test('Native parity corpus covers mixed signed zeros without exponential growth in high dimensions',
  {skip:!python},async()=>{
    const {stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_region_parity_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
  });
test('Correlated RMS bounds eliminate proved constant components before expanding mean and inverse',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_rms_components_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 5 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Correlated RMS enclosures: cases=5332992 violations=0/);
    assert.match(stdout,/Early RMS constant component: cases=122888 mismatches=0/);
    assert.match(stdout,/Dominant RMS square: kernels=126972 normCases=\d+ mismatches=0/);
  });
test('Scoped exact algebra factors closed producers while preserving native payloads and IEEE barriers',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_exact_factor_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 7 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Exact producer factoring: cases=92166 mismatches=0/);
    assert.match(stdout,/Linear signed-zero factoring: cases=122888 mismatches=0/);
    assert.match(stdout,/Positive-zero multivariable factor: cases=552996 mismatches=0/);
    if(process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT) assert.match(stdout,/Complete composition growth regression: characters=\d+->\d+; finalArtifactEmitted=false/);
  });
test('Finite F32 cells and compile-time constants preserve native boundaries and signed zeros',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_f32_cells_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 5 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/F32 cell native parity: cases=268216 mismatches=0/);
    assert.match(stdout,/Constant cast parity: cases=69634 mismatches=0/);
    assert.match(stdout,/Unit-grid scale native parity: cases=\d+ mismatches=0/);
  });
test('Proved unsigned masks remove redundant operations without changing overflow or numeric contexts',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_known_bits_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Known word mask parity: comparisons=278544 mismatches=0/);
  });
test('Stabilized branches share decisions, preserve lazy guards and emit flat input-only expressions',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_coherent_paths_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 16 tests/);
    assert.match(stdout,/Selected numeric propagation parity: cases=30722 mismatches=0/);
    assert.match(stderr,/OK/);
    if(process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT) assert.match(stdout,/Flat checkpoint normalization: cases=888832 mismatches=0/);
  });
test('Compiler-only literal sharing uses expanded costs and emits bit-exact input-only expressions',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_streaming_literals_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 7 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Emitted streaming literal: cases=30722 mismatches=0/);
    if(process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT) {
      assert.match(stdout,/Complete compiler composition: cases=60 mismatches=0; position=0 dimension=2; logicalCharacters=\d+; emittedFinalParity=false/);
      assert.match(stdout,/Emitted checkpoint inverse: cases=888832 mismatches=0/);
    }
  });
test('SymPy factors typed word polynomials modulo 2^64 without reassociating floating operations',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_word_factor_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 5 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Modular word factor parity: comparisons=278568 mismatches=0/);
  });
test('Parallel SymPy blocks preserve ordered F32 folds, branch contexts and memory admission',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_parallel_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    if(process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT) assert.match(stdout,/cases=60 mismatches=0; finalParity=false/);
  });
test('Equivalent scan backend preserves Unicode regex results and byte-identical saved compiler states',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_scan_backend_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 13 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/"identical": true/);
    assert.match(stdout,/Producer wrapper allocation parity:/);
  });
test('SymPy savepoints preserve completed rounding frontiers and reject incompatible or corrupt state',
  {skip:!python},async()=>{
    const {stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_savepoints_test.py'],{timeout:120_000});
    assert.match(stderr,/Ran 8 tests/);
    assert.match(stderr,/OK/);
  });
test('SymPy string substitution factors and simplifies each step with isolated IEEE branch proofs',
  {skip:!python},async()=>{
    const {stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_strings_test.py'],{timeout:120_000});
    assert.match(stderr,/Ran 32 tests/);
    assert.match(stderr,/OK/);
  });

test('SymPy checkpoint working string is re-read and matches a freshly captured coordinate reference',
  {skip:!python||!process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_checkpoint_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/OK/);
    assert.match(stdout,/60 exact cases/);
    assert.match(stdout,/Closed RMS rounding parity: cases=888832 mismatches=0/);
    assert.match(stdout,/Closed gate\/up projection parity: cases=1015808 mismatches=0/);
    assert.match(stdout,/Early Half product parity: cases=888832 mismatches=0/);
    assert.match(stdout,/Expression file allocation parity:/);
    assert.match(stdout,/finalParity=false/);
  });

test('SymPy elementary conversion strings preserve native IEEE cells and composed boundaries',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_conversions_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 34 tests/);
    assert.match(stdout,/Completed selector activation parity: cases=18434 mismatches=0/);
    assert.match(stdout,/Certified frontier composition parity: cases=147472 mismatches=0/);
    assert.match(stdout,/Scoped conversion parity: cases=30722 mismatches=0/);
    assert.match(stdout,/Positive-factor sign parity: cases=507904 mismatches=0/);
    assert.match(stdout,/Compact envelope selector parity: cases=507904 mismatches=0/);
    assert.match(stdout,/Synchronized sibling native parity: cases=131072 mismatches=0/);
    assert.match(stdout,/Exact branch facts native parity: cases=196608 mismatches=0/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Integer offset tandem parity: cases=1152322 mismatches=0 integerCastLoss=0/);
    assert.match(stdout,/Half cell dependency parity: cases=327680 mismatches=0/);
    assert.match(stdout,/Native Half squares: 63488, mismatches=0/);
    assert.match(stdout,/Two-Half-square F32 certificate: pairs=503856640 mismatches=0 oddTies=0/);
    assert.match(stdout,/F32 normal-cell certificate: sqrt=2139095040 reciprocals=4219469826 mismatches=0 ties=0/);
    assert.match(stdout,/F32=100679708 F16=206870 composed=559086 prunedF32=100663300 prunedF16=3072 mismatches=0/);
  });


test('Bounded SiLU arithmetic lowering matches every certified Half scalar and vector input',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_silu_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 5 tests/);
    assert.match(stdout,/Quadratic Half threshold preimage: cases=19458 mismatches=0 sourceCopies=7->6/);
    assert.match(stdout,/Completed square tandem certificate: cases=19458 rawNegativeZeroDifferences=1 sourceCopies=6->4/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Bounded SiLU word parity: cases=22530 mismatches=0/);
    assert.match(stdout,/Bounded SiLU word parity: cases=19458 mismatches=0/);
    assert.match(stdout,/Bounded SiLU word parity: cases=21506 mismatches=0/);
  });

test('Positive F32 square-root word lowering preserves all mantissas, parities and subnormals',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_sqrt_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 2 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/F32 sqrt emitted word parity: cases=25167601 mismatches=0/);
    assert.match(stdout,/Fixed-scale sqrt parity: cases=58720257 mismatches=0/);
  });

test('IEEE sign projections preserve signed zero and reduce completed down-conversion strings',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_signs_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Finite sign projection parity: cases=3047424 mismatches=0/);
    assert.match(stdout,/Completed down projection parity: cases=147472 mismatches=0/);
    assert.match(stdout,/Seeded checkpoint down reduction parity: cases=147472 mismatches=0/);
    const diagnostic=await promisify(execFile)(python!,['helpers/direct_sympy_envelope_diagnostic_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(diagnostic.stderr,/Ran 2 tests/);
    assert.match(diagnostic.stderr,/OK/);
    const paths=await promisify(execFile)(python!,['helpers/direct_sympy_path_conversion_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(paths.stderr,/Ran 5 tests/);
    assert.match(paths.stdout,/Disconnected magnitude parity: cases=122888 mismatches=0/);
    assert.match(paths.stderr,/OK/);
    assert.match(paths.stdout,/Path Half conversion parity: cases=245776 mismatches=0/);
    assert.match(paths.stdout,/Path F32 conversion parity: cases=\d+ mismatches=0/);
    assert.match(paths.stdout,/Path tandem conversion parity: cases=19458 mismatches=0/);
    const rms=await promisify(execFile)(python!,['helpers/direct_sympy_rms_guard_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(rms.stderr,/Ran 2 tests/);
    assert.match(rms.stderr,/OK/);
    assert.match(rms.stdout,/RMS guard boundary parity: cases=634876 mismatches=0 guardDisagreements=39576/);
  });

test('Half interval precision removes redundant F32 rounding with native emitted-expression parity',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_half_quantum_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Half neighbor cell native parity: cases=\d+ mismatches=0/);
    assert.match(stdout,/Half interval grid proof: cases=63488 violations=0/);
    assert.match(stdout,/Half interval sum native parity: cases=1050625 mismatches=0/);
    assert.match(stdout,/Half interval expression: characters=\d+->\d+/);
  });

test('Disjoint input partitions preserve complete Half coverage and emitted coordinate parity within each certified region',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_input_partitions_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 8 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Partition domain proof: HalfPatterns=63488 vectorPatterns=4030726144 lost=0 overlaps=0/);
    assert.match(stdout,/Checkpoint update-cell geometry: regions=4 covered=1006505988 lost=0 overlaps=0 finalParity=false/);
    const artifact=stdout.match(/Emitted full region coordinate: cases=576 mismatches=0 characters=(\d+) position=0 dimension=2 fullInputCoverage=false/);
    assert.ok(artifact,'Native reference parity of the emitted region is required');
    assert.ok(Number(artifact[1])<21922,'Rational root lowering must reduce the previous emitted region size');
  });

test('Proved signed-zero updates preserve formats, branch isolation and native payloads',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_zero_updates_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Zero-update native parity: cases=656300 mismatches=0/);
    assert.match(stdout,/Exact-zero interval parity: cases=319488 mismatches=0/);
  });

test('Constant IEEE cells eliminate whole producers and propagate compound branch contexts',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_constant_cells_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Constant-cell native parity: cases=317440 mismatches=0/);
    assert.match(stdout,/Signed-zero cell native parity: cases=63488 mismatches=0/);
  });

test('Identical emitted coordinate bodies coalesce without losing coverage, signed zeros or lazy guards',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_partition_coalesce_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 3 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Coalesced native parity: cases=126976 mismatches=0/);
  });

test('Budgeted coordinate continuation persists actual producers and rejects incompatible saved dimensions',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_checkpoint_run_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 2 tests/);
    assert.match(stderr,/OK/);
    if(process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT){
      assert.match(stdout,/Persistent checkpoint run:/);
      assert.match(stdout,/"identityRejectionBeforeMutation": true/);
    }
  });


test('Coupled projection exclusions preserve rounded reductions across every central Half pair',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_projection_constraints_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Coupled projection proof: pairs=20980737 normViolations=0 exclusionViolations=0/);
  });

test('Cross-normalization path exclusions include every original rounding and reject foreign contexts',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_norm_correlation_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 6 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Cross norm proof: pairs=83922948 distanceViolations=0 exclusionViolations=0/);
  });

test('Directed attention bounds retain both projection roundings and prove residual cells for every mixed Half pair',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_directed_residual_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 2 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Directed residual proof: pairs=62924800 intervalViolations=0 cellViolations=0/);
  });

test('RMS subnormal branch proofs retain shared means and original Half storage cells',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_rms_branch_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 4 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/RMS subnormal branch cells: pairs=462422016 selected=44294796 violations=0/);
    if(process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT)assert.match(stdout,/Zero-admitting RMS parity: cases=3047424 mismatches=0/);
  });

test('Source changes retain only audited partition geometry and recompile numerical evidence',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_partition_replan_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 2 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Geometry replan: numericalResultsReused=false lost=0 overlaps=0/);
  });

test('Branch-local projection cells propagate constants in original dependency order without replacing residual sums',
  {skip:!python || !process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_branch_projection_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 3 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/Branch projection cells: pairs=462422016 selected=22147398 violations=0/);
  });
