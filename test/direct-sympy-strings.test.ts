import test from 'node:test';
import assert from 'node:assert/strict';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const python=process.env.LLM_INNER_DIRECT_PYTHON;
test('SymPy string substitution factors and simplifies each step with isolated IEEE branch proofs',
  {skip:!python},async()=>{
    const {stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_strings_test.py'],{timeout:120_000});
    assert.match(stderr,/Ran 17 tests/);
    assert.match(stderr,/OK/);
  });

test('SymPy checkpoint working string is re-read and matches a freshly captured coordinate reference',
  {skip:!python||!process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_checkpoint_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/OK/);
    assert.match(stdout,/60 exact cases/);
    assert.match(stdout,/finalParity=false/);
  });

test('SymPy elementary conversion strings preserve native IEEE cells and composed boundaries',
  {skip:!python},async()=>{
    const {stdout,stderr}=await promisify(execFile)(python!,['helpers/direct_sympy_conversions_test.py'],{timeout:120_000,maxBuffer:1024*1024});
    assert.match(stderr,/Ran 11 tests/);
    assert.match(stderr,/OK/);
    assert.match(stdout,/F32=100679708 F16=206870 composed=559086 prunedF32=100663300 prunedF16=3072 mismatches=0/);
  });
