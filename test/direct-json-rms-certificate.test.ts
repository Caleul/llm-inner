import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {certifyJsonRmsRoot,jsonCertifiedRmsRootSteps,jsonThreeNewtonRoot} from '../src/direct-json-rms-certificate.js';
import {lowerJsonCertifiedRmsRootAsF64,lowerJsonPositiveNormalSqrtAsF64} from '../src/direct-json-sqrt.js';
import {jsonInput,jsonOperation as o,jsonConstant as c} from '../src/direct-json-expression.js';
import {lowerJsonRoundNormalF32AsF64 as r32} from '../src/direct-json-f16.js';
import {lowerJsonFiniteF16AsF64} from '../src/direct-json-half-value.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';

test('RMS consumer certificate exhausts root error cells and their finite half pair preimages',()=>{
  const certificate=certifyJsonRmsRoot(2,1e-6);
  assert.equal(certificate.steps,3);assert.equal(certificate.complete,true);
  assert.equal(certificate.normalizedMantissas,16777216);assert.equal(certificate.rootFailures,8);
  assert.equal(certificate.variances,208);assert.equal(certificate.sumCells,224);assert.equal(certificate.cases.length,121);
  for(const item of certificate.cases)assert.deepEqual(item.candidateBits,item.referenceBits);
  assert.ok(Object.isFrozen(certificate));assert.ok(Object.isFrozen(certificate.cases[0]!.inputBits));
  assert.throws(()=>jsonCertifiedRmsRootSteps({...certificate}),/Unverified/);
});

test('RMS certificate keeps four steps for unsupported widths, resource limits and invalid domains',()=>{
  for(const c of [certifyJsonRmsRoot(4,1e-6),certifyJsonRmsRoot(2,1e-6,{maxSumCells:1}),certifyJsonRmsRoot(2,1e-6,{maxPairs:1})]){
    assert.equal(c.steps,4);assert.equal(c.complete,false);assert.equal(jsonCertifiedRmsRootSteps(c),4);
  }
  assert.throws(()=>certifyJsonRmsRoot(2,0),RangeError);
  assert.throws(()=>certifyJsonRmsRoot(-1,1e-6),RangeError);
  assert.throws(()=>certifyJsonRmsRoot(2,1e-6,{maxPairs:0}),RangeError);
});

test('three-step root is restricted to the certified consumer; standalone roots retain exact rounding',()=>{
  const input=jsonInput('f64','X1'),certificate=certifyJsonRmsRoot(2,1e-6);
  const standalone=lowerJsonPositiveNormalSqrtAsF64(input),consumer=lowerJsonCertifiedRmsRootAsF64(input,certificate);
  const word=new DataView(new ArrayBuffer(4));
  for(const b of [1073002314,1073174610,1073214678,1073296793,1073736324,1073868703,1074471489,1074782274]){
    word.setUint32(0,b);const x=word.getFloat32(0),reference=Math.fround(Math.sqrt(x));
    assert.notEqual(jsonThreeNewtonRoot(x),reference);
    assert.equal(evaluate(standalone,{X1:x},{allowPendingPrimitives:false}),reference);
    assert.equal(evaluate(consumer,{X1:x},{allowPendingPrimitives:false}),jsonThreeNewtonRoot(x));
  }
});

const python=process.env.LLM_INNER_DIRECT_PYTHON;
test('certified RMS critical pairs match live PyTorch for every sign and one/eight-token shapes',
  {skip:!python},async()=>{
    const dir=await mkdtemp(join(tmpdir(),'direct-json-rms-'));
    try{
      const certificate=certifyJsonRmsRoot(2,1e-6),source=join(dir,'certificate.json'),target=join(dir,'actual.json');
      await writeFile(source,JSON.stringify(certificate));
      await promisify(execFile)(python!,[new URL('../../helpers/capture_direct_json_rms_pairs.py',import.meta.url).pathname,source,target]);
      const result=JSON.parse(await readFile(target,'utf8'));
      assert.equal(result.cases,certificate.cases.length*8);assert.deepEqual(result.mismatches,[]);
    }finally{await rm(dir,{recursive:true,force:true});}
});


test('closed JSON RMS composition preserves every critical coordinate and sign without pending primitives',()=>{
  const certificate=certifyJsonRmsRoot(2,1e-6),x=jsonInput('f64','X1'),y=jsonInput('f64','X2');
  const sum=r32(o('add','f64',o('mul','f64',x,x),o('mul','f64',y,y)));
  const variance=r32(o('add','f64',o('div','f64',sum,c('f64',2)),c('f64',Math.fround(1e-6))));
  const root=lowerJsonCertifiedRmsRootAsF64(variance,certificate),inverse=r32(o('div','f64',c('f64',1),root));
  const outputs=[x,y].map(input=>lowerJsonFiniteF16AsF64(r32(o('mul','f64',input,inverse))));
  for(const pair of certificate.cases)for(let signs=0;signs<4;signs++){
    const inputs=pair.inputBits.map((b,i)=>decodeIeeeF16ToF32(b|((signs&(1<<i))?0x8000:0)));
    for(let i=0;i<2;i++){
      const expected=decodeIeeeF16ToF32(pair.referenceBits[i]!|((signs&(1<<i))?0x8000:0));
      assert.ok(Object.is(evaluate(outputs[i]!,{X1:inputs[0]!,X2:inputs[1]!},{allowPendingPrimitives:false}),expected));
    }
  }
});
