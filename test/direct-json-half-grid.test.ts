import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonOperation as o,jsonConstant as c} from '../src/direct-json-expression.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {certifyJsonHalfDifferenceExp,lowerJsonSmallNonpositiveExpAsF64,foldJsonExpPolynomial} from '../src/direct-json-exp.js';
import type {JsonModelLoweringFacts} from '../src/direct-json-model.js';
const facts=():JsonModelLoweringFacts=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap()});

test('every point of the finite half sum/difference grid within [-1,1] is exact F32',()=>{
  const limit=2**24,quantum=2**-24;
  for(let k=-limit;k<=limit;k++)assert.equal(Math.fround(k*quantum),k*quantum);
  assert.ok(Object.is(Math.fround(-0),-0));
  assert.notEqual(Math.fround(1+quantum),1+quantum);
});

test('certified bounded half sums and differences lose their F32 rounding but preserve all finite admitted operands',()=>{
  for(const operator of ['add','sub'] as const){
    const x=input('f16','X1'),y=input('f16','X2'),source=o(operator,'f32',o('widen','f32',x),o('widen','f32',y));
    const proof=facts();proof.ranges!.set(x,{minimum:-.5,maximum:.5});proof.ranges!.set(y,{minimum:-.5,maximum:.5});
    const closed=lowerJsonModelExpression(source,proof),baseline=lowerJsonModelExpression(source,facts());
    assert.ok(measureJsonExpression(closed).serializedBytes<measureJsonExpression(baseline).serializedBytes);
    for(let b=0;b<65536;b++)if((b&0x7c00)!==0x7c00){
      const u=decodeIeeeF16ToF32(b);if(Math.abs(u)>.5)continue;
      for(const v of [u,-u,.5,-.5,0,-0,2**-24,-(2**-24)]){
        const expected=Math.fround(operator==='add'?u+v:u-v);
        assert.ok(Object.is(evaluate(closed,{X1:u,X2:v},{allowPendingPrimitives:false}),expected),`${operator}, bits=${b}`);
      }
    }
  }
});

test('half-grid rounding removal requires both the half lattice and the absolute domain bound',()=>{
  const x=input('f16','X1'),y=input('f16','X2');
  const sum=o('add','f32',o('widen','f32',x),o('widen','f32',y));
  const proof=facts();proof.ranges!.set(x,{minimum:-1,maximum:1});proof.ranges!.set(y,{minimum:0,maximum:2**-24});
  assert.equal(evaluate(lowerJsonModelExpression(sum,proof),{X1:1,X2:2**-24},{allowPendingPrimitives:false}),1);
  const product=o('mul','f32',o('widen','f32',y),c('f32',2**-9));
  proof.ranges!.set(product,{minimum:0,maximum:2**-33});
  const fineSum=o('add','f32',c('f32',.5),product);
  assert.equal(evaluate(lowerJsonModelExpression(fineSum,proof),{X2:2**-24},{allowPendingPrimitives:false}),.5);
  assert.notEqual(.5+2**-33,.5);
});


test('half-difference exponential removes only its unreachable tiny dispatch and retains exact certified polynomial bits',()=>{
  const x=input('f64','X1'),certificate=certifyJsonHalfDifferenceExp(.001);
  const closed=lowerJsonSmallNonpositiveExpAsF64(x,certificate),generic=lowerJsonSmallNonpositiveExpAsF64(x);
  assert.equal(measureJsonExpression(closed).uniqueDecisions,0);
  assert.equal(measureJsonExpression(generic).uniqueDecisions,1);
  for(const u of [0,-0,-(2**-149),-(2**-126)])
    assert.equal(evaluate(generic,{X1:u},{allowPendingPrimitives:false}),1);
  for(let k=0;k<=certificate.maxGridIndex;k++){
    const u=-k*2**-24;
    assert.equal(evaluate(closed,{X1:u},{allowPendingPrimitives:false}),foldJsonExpPolynomial(u,certificate.polynomialDegree));
  }
  assert.throws(()=>lowerJsonSmallNonpositiveExpAsF64(x,{...certificate}),/Unverified/);
});
