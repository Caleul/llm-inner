import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonConstant as c,jsonInput as input,jsonOperation as o} from '../src/direct-json-expression.js';
import {jsonModelMagnitudeAnalysis} from '../src/direct-json-magnitude.js';
import {lowerJsonF32ToF16} from '../src/direct-json-f16.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';

const facts=()=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()});
test('scoped unsigned magnitude proofs enclose all finite half inputs through F32 arithmetic and conversion',()=>{
  const x=input('f16','X1'),wide=o('widen','f32',x),square=o('mul','f32',wide,wide);
  const quotient=o('div','f32',wide,c('f32',3)),half=lowerJsonF32ToF16(quotient),word=new DataView(new ArrayBuffer(4));
  for(const bound of [{minimum:16,maximum:65504},{minimum:0,maximum:16},{minimum:2**-24,maximum:2**-14},{minimum:0,maximum:0}]){
    const proof={...facts(),inputMagnitudeBounds:new Map([['X1',bound]])};proof.halfSources.set(half,quotient);
    const range=jsonModelMagnitudeAnalysis(proof),sq=range(square)!,q=range(quotient)!,h=range(half)!;
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const value=decodeIeeeF16ToF32(bits),magnitude=Math.abs(value);if(magnitude<bound.minimum||magnitude>bound.maximum)continue;
      const squared=Math.fround(value*value),divided=Math.abs(Math.fround(value/3));
      word.setFloat32(0,divided);const rounded=decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(word.getUint32(0))));
      assert.ok(sq.minimum<=squared&&squared<=sq.maximum);assert.ok(q.minimum<=divided&&divided<=q.maximum);
      assert.ok(h.minimum<=rounded&&rounded<=h.maximum);
    }
  }
});
test('a scoped lower bound removes only proved residual changes and keeps the signed complement',()=>{
  const x=input('f16','X1'),delta:['constant','f16',string]=['constant','f16','0x1400'];
  const source=o('add','f32',o('widen','f32',x),o('widen','f32',delta)),half=lowerJsonF32ToF16(source),proof=facts();
  proof.halfSources.set(half,source);
  const baseline=lowerJsonModelExpression(half,proof);
  const bound={minimum:16,maximum:65504},scoped=lowerJsonModelExpression(half,{...proof,inputMagnitudeBounds:new Map([['X1',bound]])});
  assert.deepEqual(scoped,input('f64','X1'));assert.ok(measureJsonExpression(scoped).serializedBytes<measureJsonExpression(baseline).serializedBytes);
  for(const value of [16,-16,16.015625,-16.015625,65504,-65504])
    assert.ok(Object.is(evaluate(scoped,{X1:value},{allowPendingPrimitives:false}),evaluate(baseline,{X1:value},{allowPendingPrimitives:false})));
  assert.ok(Object.is(evaluate(lowerJsonModelExpression(half,{...proof,inputMagnitudeBounds:new Map([['X1',{minimum:0,maximum:16}]])}),{X1:-0},{allowPendingPrimitives:false}),decodeIeeeF16ToF32(0x1400)));
});
test('invalid magnitude proofs are rejected rather than silently shrinking the input domain',()=>{
  for(const bound of [{minimum:-1,maximum:16},{minimum:32,maximum:16},{minimum:0,maximum:Infinity},{minimum:0,maximum:65536}])
    assert.throws(()=>jsonModelMagnitudeAnalysis({...facts(),inputMagnitudeBounds:new Map([['X1',bound]])}),/Invalid/);
});


test('scoped normal half conversion removes subnormal decisions for both signs and preserves every admitted half input',()=>{
  const x=input('f16','X1'),source=o('div','f32',o('widen','f32',x),c('f32',65504)),half=lowerJsonF32ToF16(source),proof=facts();
  proof.halfSources.set(half,source);
  const baseline=lowerJsonModelExpression(half,proof),scoped=lowerJsonModelExpression(half,{...proof,inputMagnitudeBounds:new Map([['X1',{minimum:16,maximum:65504}]])});
  assert.ok(measureJsonExpression(scoped).serializedBytes<measureJsonExpression(baseline).serializedBytes);
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const value=decodeIeeeF16ToF32(bits);if(Math.abs(value)<16)continue;
    assert.ok(Object.is(evaluate(scoped,{X1:value},{allowPendingPrimitives:false}),evaluate(baseline,{X1:value},{allowPendingPrimitives:false})));
  }
});
