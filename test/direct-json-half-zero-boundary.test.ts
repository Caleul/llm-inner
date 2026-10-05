import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput,jsonConstant as c,jsonOperation as o} from '../src/direct-json-expression.js';
import {lowerJsonF32ToF16} from '../src/direct-json-f16.js';
import {lowerJsonModelExpression,createJsonModelLowerer} from '../src/direct-json-lower-model.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import type {JsonModelLoweringFacts} from '../src/direct-json-model.js';
const facts=():JsonModelLoweringFacts=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap()});

test('Half reduction with zero lanes retains signed-zero arithmetic but eliminates repeated quantization',()=>{
  const input=jsonInput('f16','X1'),x=o('widen','f32',input);
  for(const [source,reference] of [
    [o('add','f32',o('add','f32',c('f32',0),x),c('f32',0)),(v:number)=>(0+v)+0],
    [o('add','f32',o('add','f32',o('add','f32',c('f32',0),x),c('f32',0)),o('add','f32',c('f32',0),c('f32',0))),(v:number)=>((0+v)+0)+(0+0)],
    [o('sub','f32',c('f32',0),o('mul','f32',x,c('f32',-1))),(v:number)=>0-v*(-1)],
    [o('div','f32',o('sub','f32',x,c('f32',-0)),c('f32',-1)),(v:number)=>(v-(-0))/(-1)],
  ] as const){
    const target=lowerJsonF32ToF16(source),proof=facts();proof.halfSources.set(target,source);
    const closed=lowerJsonModelExpression(target,proof);
    assert.equal(measureJsonExpression(closed).inputReferences,1n);
    assert.equal(measureJsonExpression(closed).uniqueDecisions,0);
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const X1=decodeIeeeF16ToF32(bits);
      assert.ok(Object.is(evaluate(closed,{X1},{allowPendingPrimitives:false}),reference(X1)),`bits=${bits}`);
    }
  }
});

test('A nonzero addition or nonunit product cannot inherit the Half lattice',()=>{
  const input=jsonInput('f16','X1'),x=o('widen','f32',input);
  for(const source of [o('add','f32',x,c('f32',2**-11)),o('mul','f32',x,c('f32',1+2**-10))]){
    const target=lowerJsonF32ToF16(source),proof=facts();proof.halfSources.set(target,source);
    const closed=lowerJsonModelExpression(target,proof);
    assert.ok(measureJsonExpression(closed).inputReferences>1n);
  }
});

test('Completed singleton-attention probability is substituted before its consumer rounding',()=>{
  const proof=facts(),one=c('f32',1),sourceProbability=o('div','f32',one,one);
  const probability=lowerJsonF32ToF16(sourceProbability);proof.halfSources.set(probability,sourceProbability);
  const lowerer=createJsonModelLowerer(proof);
  assert.equal(evaluate(lowerer.lower(probability)),1);
  const x=o('widen','f32',jsonInput('f16','X1'));
  const product=o('mul','f32',o('widen','f32',probability),x);
  const source=o('add','f32',o('add','f32',c('f32',0),product),c('f32',0));
  const target=lowerJsonF32ToF16(source);proof.halfSources.set(target,source);
  const closed=lowerer.lower(target);
  assert.equal(measureJsonExpression(closed).inputReferences,1n);
  for(const X1 of [-0,0,2**-24,-(2**-24),65504,-65504])
    assert.ok(Object.is(evaluate(closed,{X1},{allowPendingPrimitives:false}),(0+1*X1)+0));
});
