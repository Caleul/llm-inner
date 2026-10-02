import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o,type JsonExpression} from '../src/direct-json-expression.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {simplifyJsonBitPrecision,type JsonPrecisionFacts} from '../src/direct-json-precision.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
const facts=()=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()});

test('F32 widening and zero identities retain the stronger completed F16 precision proof',()=>{
  const x=input('f16','X1'),wide=o('widen','f32',x);
  for(const root of [wide,o('add','f32',wide,c('f32',-0)),o('sub','f32',wide,c('f32',0))]){
    const precision:JsonPrecisionFacts=new WeakMap();
    const closed=lowerJsonModelExpression(root,facts(),precision);
    assert.equal(precision.get(closed),42);
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const value=decodeIeeeF16ToF32(bits);
      assert.ok(Object.is(evaluate(closed,{X1:value}),value));
    }
  }
});

test('removing a redundant F32 bitword rounding never weakens its F16 producer precision',()=>{
  const x=input('u64','Bits'),u=(n:bigint)=>c('u64',n),precision:JsonPrecisionFacts=new WeakMap();
  const odd=o('and','u64',o('shr','u64',x,u(29n)),u(1n));
  const rounded=o('and','u64',o('add','u64',x,o('add','u64',u(0xfffffffn),odd)),u(0xffffffffe0000000n));
  precision.set(x,42);precision.set(rounded,29);
  const closed=simplifyJsonBitPrecision(rounded,precision);
  assert.equal(closed,x);assert.equal(precision.get(closed),42);
  const word=new DataView(new ArrayBuffer(8));
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const value=decodeIeeeF16ToF32(bits);
    word.setFloat64(0,value);const Bits=word.getBigUint64(0);
    assert.equal(evaluate(closed,{Bits}),evaluate(rounded,{Bits}));
  }
});

test('stronger precision is retained only when proved; arbitrary F32 constants do not acquire F16 precision',()=>{
  const precision:JsonPrecisionFacts=new WeakMap();
  const closed=lowerJsonModelExpression(c('f32',1+2**-23),facts(),precision);
  assert.equal(precision.get(closed),29);
  const arbitrary:JsonExpression=input('f64','X1');
  assert.equal(lowerJsonModelExpression(arbitrary,facts(),precision),arbitrary);
  assert.equal(precision.get(arbitrary),undefined);
});
