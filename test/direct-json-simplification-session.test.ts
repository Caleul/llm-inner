import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonConstant as c,jsonInput as input,jsonOperation as o} from '../src/direct-json-expression.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {JsonSimplificationSession,simplifyJsonExpression,simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';

test('Completed producer proofs are reused only under the same relevant branch context',()=>{
  const session=new JsonSimplificationSession(),x=input('u32','X1'),guard=o('lt','bool',x,c('u32',8));
  const root=o('if','u32',guard,o('add','u32',x,c('u32',1)),o('sub','u32',x,c('u32',1)));
  const first=simplifyJsonFixedPoint(root,32,1_000_000,session).expression;
  const misses=session.stats.misses,hits=session.stats.hits;
  assert.equal(simplifyJsonFixedPoint(first,32,1_000_000,session).expression,first);
  assert.equal(session.stats.misses,misses);assert.ok(session.stats.hits>hits);
  const yes=simplifyJsonExpression(root,undefined,[[guard,true]],1_000_000,session);
  const no=simplifyJsonExpression(root,undefined,[[guard,false]],1_000_000,session);
  for(let X1=0n;X1<16n;X1++){
    assert.equal(evaluate(first,{X1}),evaluate(root,{X1}));
    if(X1<8n)assert.equal(evaluate(yes,{X1}),evaluate(root,{X1}));
    else assert.equal(evaluate(no,{X1}),evaluate(root,{X1}));
  }
  assert.notDeepEqual(yes,no);
});

test('Reusing numerical proof nodes cannot erase a reachable undefined branch',()=>{
  const session=new JsonSimplificationSession(),guard=input('bool','X1');
  const bad=o('div','u32',c('u32',1),c('u32',0));
  const root=o('if','u32',guard,c('u32',17),bad);
  const safe=simplifyJsonExpression(root,undefined,[[guard,true]],1_000_000,session);
  assert.equal(evaluate(safe),17n);
  const general=simplifyJsonFixedPoint(root,32,1_000_000,session).expression;
  assert.equal(evaluate(general,{X1:true}),17n);
  assert.throws(()=>evaluate(general,{X1:false}),/zero/i);
  assert.throws(()=>evaluate(simplifyJsonExpression(root,undefined,[[guard,false]],1_000_000,session)),/zero/i);
});
