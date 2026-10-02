import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonOperation as o,jsonConstant as c} from '../src/direct-json-expression.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {lowerJsonF32ToF16} from '../src/direct-json-f16.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';
import type {JsonModelLoweringFacts} from '../src/direct-json-model.js';
const facts=():JsonModelLoweringFacts=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap()});

test('JSON zero identities preserve every finite half value and both zero signs',()=>{
  const wide=o('widen','f32',input('f16','X1'));
  for(const operator of ['add','sub'] as const)for(const zero of [0,-0]){
    const source=o(operator,'f32',wide,c('f32',zero)),closed=lowerJsonModelExpression(source,facts());
    for(let b=0;b<65536;b++)if((b&0x7c00)!==0x7c00){
      const x=decodeIeeeF16ToF32(b),expected=Math.fround(operator==='add'?x+zero:x-zero);
      assert.ok(Object.is(evaluate(closed,{X1:x},{allowPendingPrimitives:false}),expected),`${operator}, zero=${Object.is(zero,-0)?'-0':'+0'}, bits=${b}`);
    }
  }
});

test('JSON reduction trailing zero is removed only after its accumulator excludes negative zero',()=>{
  const x=o('widen','f32',input('f16','X1')),y=o('widen','f32',input('f16','X2'));
  const a=o('mul','f32',x,c('f32',.03125)),b=o('mul','f32',y,c('f32',.0625));
  const sum=o('add','f32',o('add','f32',c('f32',0),a),o('add','f32',c('f32',0),b));
  const source=o('add','f32',sum,o('add','f32',c('f32',0),c('f32',0)));
  const closed=lowerJsonModelExpression(source,facts()),withoutTrailing=lowerJsonModelExpression(sum,facts());
  assert.equal(measureJsonExpression(closed).serializedBytes,measureJsonExpression(withoutTrailing).serializedBytes);
  const proof=facts(),target=lowerJsonF32ToF16(source);proof.halfSources.set(target,source);
  const half=lowerJsonModelExpression(target,proof),word=new DataView(new ArrayBuffer(4));
  let state=0x918a49;
  for(let i=0;i<20000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;const ab=state&0xffff;
    state=(Math.imul(state,1664525)+1013904223)>>>0;const bb=state&0xffff;
    if((ab&0x7c00)===0x7c00||(bb&0x7c00)===0x7c00)continue;
    const u=decodeIeeeF16ToF32(ab),v=decodeIeeeF16ToF32(bb);
    const expected=Math.fround(Math.fround(0+u*.03125)+Math.fround(0+v*.0625));
    assert.ok(Object.is(evaluate(closed,{X1:u,X2:v},{allowPendingPrimitives:false}),expected));
    word.setFloat32(0,expected);const expectedHalf=decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(word.getUint32(0))));
    assert.ok(Object.is(evaluate(half,{X1:u,X2:v},{allowPendingPrimitives:false}),expectedHalf));
  }
});

test('JSON branch zero-sign proof requires both arms and does not assume half underflow is positive',()=>{
  const x=o('widen','f32',input('f16','X1'));
  const choice=o('if','f32',o('lt','bool',x,c('f32',0)),c('f32',-0),c('f32',0));
  const closed=lowerJsonModelExpression(o('add','f32',choice,c('f32',0)),facts());
  for(const u of [-1,-0,0,1])assert.ok(Object.is(evaluate(closed,{X1:u},{allowPendingPrimitives:false}),0));
  const small=o('mul','f32',x,c('f32',2**-20)),half=lowerJsonF32ToF16(small),proof=facts();proof.halfSources.set(half,small);
  const widened=o('widen','f32',half),canonical=o('add','f32',widened,c('f32',0));
  assert.ok(Object.is(evaluate(lowerJsonModelExpression(widened,proof),{X1:-(2**-24)},{allowPendingPrimitives:false}),-0));
  assert.ok(Object.is(evaluate(lowerJsonModelExpression(canonical,proof),{X1:-(2**-24)},{allowPendingPrimitives:false}),0));
});
