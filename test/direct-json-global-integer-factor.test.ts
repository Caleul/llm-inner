import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o,jsonWidths,type JsonExpression,type JsonDtype} from '../src/direct-json-expression.js';
import {simplifyJsonFixedPoint,jsonExpressionIsTotal} from '../src/direct-json-simplify.js';
import {factorJsonIntegerSum} from '../src/direct-json-global-integer-factor.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';

test('global integer factors collect repeated nonadjacent inputs and preserve all u16 values',()=>{
  const x=input('u16','X1'),y=input('u16','X2');
  const shared=o('add','u16',o('mul','u16',c('u16',2n),x),o('mul','u16',c('u16',5n),y));
  const original=o('add','u16',o('mul','u16',c('u16',3n),shared),
    o('add','u16',c('u16',11n),o('mul','u16',c('u16',7n),shared)));
  const result=simplifyJsonFixedPoint(original).expression;
  assert.ok(measureJsonExpression(result).serializedBytes<measureJsonExpression(original).serializedBytes);
  for(let bits=0;bits<65536;bits++){
    const values={X1:BigInt(bits),X2:BigInt((bits*997+32769)&65535)};
    assert.equal(evaluate(result,values),evaluate(original,values));
  }
  assert.equal(simplifyJsonFixedPoint(result).expression,result);
});

test('global modular cancellation and common factors preserve overflow at every integer width',()=>{
  let cases=0;
  for(const type of ['u16','u32','u64'] as JsonDtype[]){
    const x=input(type,'X1'),y=input(type,'X2');
    const original=o('sub',type,o('add',type,o('mul',type,c(type,14n),x),
      o('add',type,o('mul',type,c(type,6n),y),o('mul',type,c(type,8n),x))),
      o('add',type,o('mul',type,c(type,22n),x),o('mul',type,c(type,2n),y)));
    const result=simplifyJsonFixedPoint(original).expression;
    assert.deepEqual(result,o('mul',type,c(type,4n),y));
    for(let i=0n;i<2048n;i++){
      const values={X1:BigInt.asUintN(jsonWidths[type],i*0xffffffffffffffffn),
        X2:BigInt.asUintN(jsonWidths[type],i*0x8000000000000001n)};
      assert.equal(evaluate(result,values),evaluate(original,values));cases++;
    }
  }
  console.log(`Global modular JSON proof: cases=${cases+65536} floatReassociation=false`);
});

test('global collection keeps partial operations and IEEE arithmetic barriers',()=>{
  const x=input('u16','X1'),bad=o('div','u16',x,c('u16',0n));
  const partial=o('sub','u16',o('add','u16',bad,x),bad);
  const result=simplifyJsonFixedPoint(partial).expression;
  assert.throws(()=>evaluate(result,{X1:7n}));
  const f=input('f64','X1'),original=o('sub','f64',o('add','f64',f,c('f64',1)),f);
  const stable=simplifyJsonFixedPoint(original).expression;
  assert.deepEqual(stable,original);
  assert.equal(evaluate(stable,{X1:1e16}),0);
  assert.ok(Object.is(evaluate(input('f64','X2'),{X2:-0}),-0));
});

test('bounded global collection declines oversized searches without dropping terms',()=>{
  const x=input('u32','X1');let original:JsonExpression=x;
  for(let i=1;i<400;i++)original=o('add','u32',original,input('u32',`X${i+1}`));
  const values=Object.fromEntries(Array.from({length:400},(_,i)=>[`X${i+1}`,BigInt(i+1)]));
  assert.equal(evaluate(simplifyJsonFixedPoint(original).expression,values),evaluate(original,values));
  let atom:JsonExpression=o('div','u32',x,c('u32',3n));
  for(let i=0;i<30;i++)atom=o('div','u32',o('add','u32',atom,atom),c('u32',3n));
  const repeated=o('add','u32',o('mul','u32',c('u32',3n),atom),
    o('add','u32',c('u32',5n),o('mul','u32',c('u32',7n),atom)));
  const ids=new WeakMap<object,number>();let next=0;
  const reduced=factorJsonIntegerSum(repeated,node=>{
    let id=ids.get(node);if(id===undefined){id=next++;ids.set(node,id);}return id;
  },node=>jsonExpressionIsTotal(node));
  assert.ok(measureJsonExpression(repeated).serializedBytes>1_000_000_000n);
  assert.ok(measureJsonExpression(reduced).serializedBytes<measureJsonExpression(repeated).serializedBytes);
  for(const value of [0n,1n,0x80000000n,0xffffffffn])
    assert.equal(evaluate(reduced,{X1:value}),evaluate(repeated,{X1:value}));
});
