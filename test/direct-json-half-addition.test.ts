import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {jsonInput as input,jsonOperation as o,jsonConstant as c,type JsonExpression} from '../src/direct-json-expression.js';
import {lowerJsonFiniteF16AsF64} from '../src/direct-json-half-value.js';
import {lowerJsonRoundNormalF32AsF64} from '../src/direct-json-f16.js';
import {lowerJsonF32ToF16} from '../src/direct-json-f16.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import type {JsonModelLoweringFacts} from '../src/direct-json-model.js';
const facts=():JsonModelLoweringFacts=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),
  exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap()});
const word=new DataView(new ArrayBuffer(4));
const reference=(x:number)=>{word.setFloat32(0,x);if(!Number.isFinite(x))return x;
  if(Object.is(x,-0))return -0;return decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(word.getUint32(0))));};

test('finite half sum/difference needs no intermediate F32 boundary for every magnitude pair',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'half-addition-')),run=promisify(execFile);
  try{
    const executable=join(dir,'certificate');
    await run('rustc',['--edition=2021','-O',new URL('../../helpers/certify_half_addition.rs',import.meta.url).pathname,'-o',executable]);
    const {stdout}=await run(executable,[],{timeout:120000});const proof=JSON.parse(stdout);
    assert.equal(proof.magnitudeValues,31744);assert.equal(proof.sumAndDifferenceComparisons,1007713280);
    assert.equal(proof.signedZeroComparisons,8);assert.equal(proof.mismatches,0);
  }finally{await rm(dir,{recursive:true,force:true});}
});

test('JSON half sum fusion preserves signed zero, binade gaps, cancellation and overflow across all finite half operands',()=>{
  for(const operator of ['add','sub'] as const){
    const x=input('f16','X1'),y=input('f16','X2');
    const source=o(operator,'f32',o('widen','f32',x),o('widen','f32',y));
    const target=lowerJsonF32ToF16(source),proof=facts();proof.halfSources.set(target,source);
    const fused=lowerJsonModelExpression(target,proof);
    const baseline=lowerJsonFiniteF16AsF64(lowerJsonRoundNormalF32AsF64(o(operator,'f64',input('f64','X1'),input('f64','X2'))));
    assert.ok(measureJsonExpression(fused).serializedBytes<measureJsonExpression(baseline).serializedBytes);
    for(let b=0;b<65536;b++)if((b&0x7c00)!==0x7c00){
      const a=decodeIeeeF16ToF32(b),m=b&0x7fff;
      for(const other of [b,Math.max(0,m-12*1024),Math.max(0,m-13*1024),(b^0x8000)]){
        const z=decodeIeeeF16ToF32(other),raw=operator==='add'?a+z:a-z;
        assert.ok(Object.is(evaluate(fused,{X1:a,X2:z},{allowPendingPrimitives:false}),reference(raw)),`${operator}, bits=${b}, other=${other}`);
      }
    }
  }
});

test('half addition identity is not applied to higher-precision operands or sums of half products',()=>{
  const x=input('f16','X1'),y=input('f16','X2');
  const a=o('widen','f32',x),b=o('mul','f32',o('widen','f32',y),c('f32',1+2**-13));
  const source=o('add','f32',a,b),target=lowerJsonF32ToF16(source),proof=facts();proof.halfSources.set(target,source);
  const closed=lowerJsonModelExpression(target,proof);
  const rawHalf=lowerJsonFiniteF16AsF64(o('add','f64',input('f64','X1'),o('mul','f64',input('f64','X2'),c('f64',1+2**-13))));
  assert.notEqual(evaluate(closed,{X1:1,X2:2**-11},{allowPendingPrimitives:false}),evaluate(rawHalf,{X1:1,X2:2**-11}));
  for(const [u,v] of [[1,2**-11],[-1,-(2**-11)],[0,2**-24],[65504,65504]])
    assert.ok(Object.is(evaluate(closed,{X1:u!,X2:v!},{allowPendingPrimitives:false}),reference(u!+v!*(1+2**-13))));
});


test('summing exact half products still needs the F32 boundary at an odd half tie',()=>{
  const x=input('f16','X1'),y=input('f16','X2'),coefficient=1+2**-10;
  const source=o('add','f32',o('mul','f32',o('widen','f32',x),c('f32',1)),
    o('mul','f32',o('widen','f32',y),c('f32',coefficient)));
  const target=lowerJsonF32ToF16(source),proof=facts();proof.halfSources.set(target,source);
  const closed=lowerJsonModelExpression(target,proof),a=1+2**-10,b=2**-11*(1-2**-10);
  const rawHalf=lowerJsonFiniteF16AsF64(o('add','f64',input('f64','X1'),
    o('mul','f64',input('f64','X2'),c('f64',coefficient))));
  assert.equal(evaluate(closed,{X1:a,X2:b},{allowPendingPrimitives:false}),reference(a+b*coefficient));
  assert.notEqual(evaluate(closed,{X1:a,X2:b},{allowPendingPrimitives:false}),evaluate(rawHalf,{X1:a,X2:b}));
});
