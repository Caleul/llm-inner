import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonConstant as c,jsonInput as input,jsonOperation as o} from '../src/direct-json-expression.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {measureJsonExpression as measure} from '../src/direct-json-measure.js';

test('Inverse bitcasts cancel within their existing arms and retain guards and signed zeros',()=>{
  const x=input('u64','X1'),y=input('u64','X2'),a=input('bool','X3'),b=input('bool','X4');
  const source=o('reinterpret','u64',o('if','f64',a,
    o('reinterpret','f64',x),o('if','f64',b,o('reinterpret','f64',y),c('f64',-0))));
  const result=simplifyJsonFixedPoint(source).expression;
  assert.ok(measure(result).serializedBytes<measure(source).serializedBytes);
  const words=[0n,0x8000000000000000n,1n,0x000fffffffffffffn,0x0010000000000000n,
    0x3ff0000000000000n,0x7fefffffffffffffn,0x7ff0000000000000n,0xfff0000000000000n,
    0x7ff8000000000042n,0xfff8000000000042n];
  for(const X1 of words)for(const X2 of words)for(const X3 of [false,true])for(const X4 of [false,true]){
    const bindings={X1,X2,X3,X4};
    assert.equal(evaluate(result,bindings),evaluate(source,bindings));
  }
  assert.throws(()=>evaluate(result,{X1:0n,X2:0n,X4:false}),/Missing input X3/);
});

test('Conditional bitcasts keep an undefined unselected arm lazy and do not distribute arithmetic',()=>{
  const x=input('u64','X1'),condition=input('bool','X2');
  const bad=o('div','u64',c('u64',1n),c('u64',0n));
  const source=o('reinterpret','u64',o('if','f64',condition,
    o('reinterpret','f64',x),o('reinterpret','f64',bad)));
  const result=simplifyJsonFixedPoint(source).expression;
  assert.equal(evaluate(result,{X1:42n,X2:true}),42n);
  assert.throws(()=>evaluate(result,{X1:42n,X2:false}),/zero/);
  const arithmetic=o('reinterpret','u64',o('add','f64',
    o('if','f64',condition,o('reinterpret','f64',x),c('f64',1)),c('f64',2)));
  assert.equal(simplifyJsonFixedPoint(arithmetic).expression[0],'reinterpret');
});
