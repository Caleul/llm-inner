import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o,type JsonExpression} from '../src/direct-json-expression.js';
import {jsonResidualCellThreshold,lowerJsonStableHalfResidual} from '../src/direct-json-residual-cell.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {lowerJsonF32ToF16} from '../src/direct-json-f16.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';

const word=new DataView(new ArrayBuffer(4));
function halfBits(value:number):number {
  if(Object.is(value,-0))return 0x8000;
  word.setFloat32(0,value);return roundDyadicToF16IfElse(f32BitsToDyadic(word.getUint32(0)));
}
test('residual cell guards retain every finite half at both correction endpoints with a strict F32 margin',()=>{
  const base=input('f16','X1');
  for(const bound of [0,2**-26,2**-25*(1-2**-12),2**-14,.004,1,7.99]){
    let endpoint=halfBits(bound);if(decodeIeeeF16ToF32(endpoint)>bound)endpoint--;
    const delta:JsonExpression=['constant','f16','0x'+endpoint.toString(16).padStart(4,'0')];
    const source=o('add','f32',o('widen','f32',base),o('widen','f32',delta)),sum=lowerJsonF32ToF16(source);
    const facts={halfSources:new WeakMap([[sum,source]]),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()};
    const guarded=lowerJsonStableHalfResidual(base,delta,sum,bound),closed=lowerJsonModelExpression(guarded,facts);
    assert.equal(closed[0],'if');const guard=closed[2] as JsonExpression;
    let kept=0;
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const value=decodeIeeeF16ToF32(bits),expected=Math.abs(value)>=jsonResidualCellThreshold(bound)!;
      assert.equal(evaluate(guard,{X1:value},{allowPendingPrimitives:false}),expected);
      if(expected){
        kept++;
        for(const correction of [-decodeIeeeF16ToF32(endpoint),decodeIeeeF16ToF32(endpoint)])
          assert.equal(halfBits(Math.fround(value+correction)),bits);
      }
    }
    assert.ok(kept>0);
  }
});
test('residual cells never remove signed-zero canonicalization or claim a non-strict or invalid bound',()=>{
  const base=input('f16','X1'),delta:JsonExpression=['constant','f16','0x0000'];
  const source=o('add','f32',o('widen','f32',base),o('widen','f32',delta)),sum=lowerJsonF32ToF16(source);
  const facts={halfSources:new WeakMap([[sum,source]]),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()};
  const closed=lowerJsonModelExpression(lowerJsonStableHalfResidual(base,delta,sum,0),facts);
  assert.ok(Object.is(evaluate(closed,{X1:-0},{allowPendingPrimitives:false}),0));
  assert.ok(Object.is(evaluate(closed,{X1:0},{allowPendingPrimitives:false}),0));
  for(const bound of [8,Infinity,NaN,-1]){
    assert.equal(jsonResidualCellThreshold(bound),undefined);
    assert.equal(lowerJsonStableHalfResidual(base,delta,sum,bound),sum);
  }
});
