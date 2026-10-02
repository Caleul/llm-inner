import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o} from '../src/direct-json-expression.js';
import {createJsonModelSignProof} from '../src/direct-json-sign.js';
import {lowerJsonFiniteF16AsF64,lowerJsonFiniteF32ThenF16AsF64} from '../src/direct-json-half-value.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {simplifyJsonFixedPoint,sameJsonExpression} from '../src/direct-json-simplify.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';

const facts=()=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()});
test('finite product and quotient sign provenance removes positive-root consumers while preserving all half signs',()=>{
  const proof=facts(),x=input('f16','X1'),root=o('pending-sqrt','f32',c('f32',1));
  proof.positiveNormalRoots.add(root);
  const sign=createJsonModelSignProof(proof,node=>{
    assert.equal(node,x);return input('f64','X1');
  },()=>undefined);
  const expected=o('and','u64',o('reinterpret','u64',input('f64','X1')),c('u64',0x8000000000000000n));
  for(const operation of ['mul','div'] as const){
    const candidate=sign(o(operation,'f32',o('widen','f32',x),root),input('f64','Ignored'));
    assert.ok(sameJsonExpression(candidate,expected));
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00)
      assert.equal(evaluate(candidate,{X1:decodeIeeeF16ToF32(bits)}),bits&0x8000?0x8000000000000000n:0n);
  }
});
test('product and quotient sign proofs match actual F32 arithmetic over all finite half values and coefficient signs',()=>{
  const x=input('f16','X1'),proof=facts(),wide=o('widen','f32',x),word=new DataView(new ArrayBuffer(4));
  const sign=createJsonModelSignProof(proof,()=>input('f64','X1'),()=>undefined);
  for(const coefficient of [1,3,-1,-3,2**-24,-(2**-24),2**16,-(2**16)]){
    for(const operation of ['mul','div'] as const){
      const candidate=sign(o(operation,'f32',wide,c('f32',coefficient)),input('f64','Ignored'));
      for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
        const value=decodeIeeeF16ToF32(bits);
        word.setFloat32(0,operation==='mul'?value*coefficient:value/coefficient);
        assert.equal(evaluate(candidate,{X1:value}),word.getUint32(0)&0x80000000?0x8000000000000000n:0n);
      }
    }
  }
});
test('sign proofs do not classify intervals containing zero and prove finite squares positive even at negative zero',()=>{
  const x=input('f16','X1'),wide=o('widen','f32',x),proof=facts();
  const sign=createJsonModelSignProof(proof,()=>input('f64','X1'),()=>({minimum:0,maximum:1}));
  const square=sign(o('mul','f32',wide,wide),input('f64','Ignored'));
  assert.deepEqual(square,c('u64',0n));
  assert.equal(evaluate(sign(x,input('f64','X1')),{X1:-0}),0x8000000000000000n);
});
test('certified source signs preserve direct and double-rounded half cells without repeating the positive divisor',()=>{
  const x=input('f64','X1'),source=o('div','f64',x,c('f64',3));
  const sign=o('and','u64',o('reinterpret','u64',x),c('u64',0x8000000000000000n));
  const interval={minimum:-.001,maximum:.001};
  for(const lower of [lowerJsonFiniteF16AsF64,lowerJsonFiniteF32ThenF16AsF64]){
    const baseline=lower(source,interval),candidate=simplifyJsonFixedPoint(lower(source,interval,sign)).expression;
    assert.ok(measureJsonExpression(candidate).serializedBytes<measureJsonExpression(baseline).serializedBytes);
    // Test every finite half near the subnormal/normal boundary and signed
    // zero, then all neighboring F64 words at selected half midpoints.
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const value=decodeIeeeF16ToF32(bits);if(Math.abs(value)>interval.maximum)continue;
      assert.ok(Object.is(evaluate(candidate,{X1:value}),evaluate(baseline,{X1:value})));
    }
    const word=new DataView(new ArrayBuffer(8));
    for(const midpoint of [2**-25,3*2**-25,2**-14-2**-25,2**-14+2**-25]){
      word.setFloat64(0,3*midpoint);const center=word.getBigUint64(0);
      for(const delta of [-1n,0n,1n])for(const signBit of [0n,0x8000000000000000n]){
        word.setBigUint64(0,(center+delta)|signBit);const value=word.getFloat64(0);
        assert.ok(Object.is(evaluate(candidate,{X1:value}),evaluate(baseline,{X1:value})));
      }
    }
  }
});
