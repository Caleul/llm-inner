import test from 'node:test';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const python=process.env.LLM_INNER_DIRECT_PYTHON;
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
    assert.match(stderr,/Ran 7 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/"identical": true/);
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
    assert.match(stderr,/Ran 31 tests/);
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
    assert.match(stderr,/Ran 1 test/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/F32 sqrt emitted word parity: cases=25167601 mismatches=0/);
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
    assert.match(paths.stderr,/Ran 4 tests/);
    assert.match(paths.stderr,/OK/);
    assert.match(paths.stdout,/Path Half conversion parity: cases=245776 mismatches=0/);
    assert.match(paths.stdout,/Path F32 conversion parity: cases=\d+ mismatches=0/);
    assert.match(paths.stdout,/Path tandem conversion parity: cases=19458 mismatches=0/);
    const rms=await promisify(execFile)(python!,['helpers/direct_sympy_rms_guard_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(rms.stderr,/Ran 2 tests/);
    assert.match(rms.stderr,/OK/);
    assert.match(rms.stdout,/RMS guard boundary parity: cases=634876 mismatches=0 guardDisagreements=39576/);
  });
