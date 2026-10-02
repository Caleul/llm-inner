import test from 'node:test';
import assert from 'node:assert/strict';
import {constantJsonF32Cell as cell} from '../src/direct-json-rounding-cell.js';
import {jsonConstant as c,jsonInput as input,jsonOperation as o} from '../src/direct-json-expression.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {evaluateJsonExpression} from '../src/direct-json-evaluator.js';
import {addDyadic,multiplyDyadic,f32BitsToDyadic,roundDyadicToF32IfElse} from '../src/fixed-f16-projection.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';

test('constant F32 cells preserve exact dyadic results for every admitted half correction and both signs',()=>{
  const word=new DataView(new ArrayBuffer(4));
  const bits=(x:number)=>{word.setFloat32(0,x);return word.getUint32(0);};
  for(const sign of [1,-1])for(const op of ['add','sub','mul']){
    const bound=op==='mul'?{minimum:1-2**-26,maximum:1+2**-26}:{minimum:-(2**-26),maximum:2**-26};
    const result=cell(op,{minimum:sign,maximum:sign},bound);assert.equal(result,sign);
    for(let code=0;code<65536;code++)if((code&0x7c00)!==0x7c00){
      const correction=decodeIeeeF16ToF32(code)*2**-16;
      const x=op==='mul'?1+correction:correction;if(x<bound.minimum||x>bound.maximum)continue;
      const a=f32BitsToDyadic(bits(sign)),delta=multiplyDyadic(f32BitsToDyadic(bits(decodeIeeeF16ToF32(code))),{coefficient:1n,exponent:-16});
      const b=op==='mul'?addDyadic({coefficient:1n,exponent:0},delta):delta;
      const exact=op==='mul'?multiplyDyadic(a,b):addDyadic(a,{...b,coefficient:b.coefficient*(op==='sub'?-1n:1n)});
      assert.equal(roundDyadicToF32IfElse(exact),bits(result!));
    }
  }
});
test('constant-cell folding rejects zeros, ties, overflow, unknown operations and invalid bounds',()=>{
  assert.equal(cell('add',{minimum:1,maximum:1},{minimum:0,maximum:2**-24}),undefined);
  assert.equal(cell('add',{minimum:0,maximum:0},{minimum:0,maximum:0}),undefined);
  assert.equal(cell('mul',{minimum:3.4e38,maximum:3.4e38},{minimum:2,maximum:2}),undefined);
  assert.equal(cell('div',{minimum:1,maximum:1},{minimum:1,maximum:1}),undefined);
  assert.equal(cell('add',{minimum:2,maximum:1},{minimum:0,maximum:0}),undefined);
});
test('a certified constant cell cuts its producer before lowering and folds the downstream root',()=>{
  const x=input('f32','X1'),sum=o('add','f32',c('f32',1),x),root=o('pending-sqrt','f32',sum);
  const facts={halfSources:new WeakMap(),positiveNormalRoots:new WeakSet([root]),exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap([[x,{minimum:0,maximum:2**-26}]])};
  // X1 is not an admitted half input. It can disappear only through the cell proof.
  const closed=lowerJsonModelExpression(root,facts);
  assert.equal(closed[0],'constant');assert.equal(evaluateJsonExpression(closed,{}, {allowPendingPrimitives:false}),1);
});
