import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o,type JsonExpression} from '../src/direct-json-expression.js';
import {shareJsonExpression} from '../src/direct-json-share.js';
import {sameJsonExpression} from '../src/direct-json-simplify.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {simplifyJsonSharedConditions} from '../src/direct-json-cofactor.js';

test('compiler structural sharing preserves the entire literal tree and floating evaluation order',()=>{
  const common=()=>o('add','f64',o('mul','f64',input('f64','X1'),c('f64',3)),c('f64',-0));
  const root=o('sub','f64',common(),common()),shared=shareJsonExpression(root);
  assert.ok(shared.mergedNodes>0);assert.equal(shared.expression[2],shared.expression[3]);
  assert.ok(sameJsonExpression(root,shared.expression));assert.equal(JSON.stringify(root),JSON.stringify(shared.expression));
  const before=measureJsonExpression(root),after=measureJsonExpression(shared.expression);
  assert.ok(after.uniqueNodes<before.uniqueNodes);
  for(const key of ['occurrences','decisions','inputReferences','serializedBytes','depth'] as const)
    assert.equal(after[key],before[key]);
  for(const X1 of [-Infinity,-1e30,-1,-0,0,1,1e30,Infinity,NaN])
    assert.ok(Object.is(evaluate(root,{X1}),evaluate(shared.expression,{X1})));
});

test('structural sharing distinguishes IEEE payloads, signed zeros, types and input names',()=>{
  const first=['constant','f64','0x7ff8000000000001'] as const;
  const second=['constant','f64','0x7ff8000000000002'] as const;
  const roots=[o('if','f64',input('bool','Gate'),first,second),
    o('if','f64',input('bool','Gate'),c('f64',0),c('f64',-0)),
    o('add','u64',o('convert','u64',input('u32','X1')),input('u64','X1')),
    o('add','f64',input('f64','X1'),input('f64','X10'))];
  for(const root of roots){
    const shared=shareJsonExpression(root).expression;
    assert.ok(sameJsonExpression(root,shared));assert.equal(JSON.stringify(root),JSON.stringify(shared));
    const offset=shared[0]==='if'?3:2;assert.notEqual(shared[offset],shared[offset+1]);
  }
});

test('structural sharing keeps lazy failures and rejects cycles or exhausted compiler budgets',()=>{
  const invalid=()=>o('div','u32',c('u32',1),input('u32','Missing'));
  const root=o('if','u32',input('bool','Gate'),o('add','u32',invalid(),invalid()),c('u32',0));
  const shared=shareJsonExpression(root).expression;
  assert.equal(evaluate(shared,{Gate:false}),0n);
  assert.throws(()=>evaluate(shared,{Gate:true,Missing:0n}),/zero/i);
  assert.throws(()=>shareJsonExpression(root,1),/budget/);
  assert.throws(()=>shareJsonExpression(root,0),/budget/);
  const cyclic:unknown[]=['add','f64',c('f64',1)];cyclic.push(cyclic);
  assert.throws(()=>shareJsonExpression(cyclic as unknown as JsonExpression),/Cyclic/);
});

test('equivalent condition objects count as one repeated candidate instead of hiding a reduction',()=>{
  const condition=()=>o('eq','bool',input('u32','X1'),c('u32',10));
  const root=o('add','u32',o('if','u32',condition(),c('u32',3),c('u32',7)),
    o('if','u32',condition(),c('u32',4),c('u32',8)));
  const result=simplifyJsonSharedConditions(root,{maxCandidates:1});
  assert.equal(result.stats.candidates,1);assert.equal(result.stats.accepted,1);
  assert.equal(result.stats.converged,true);assert.equal(result.expression[0],'if');
  for(const X1 of [0n,10n,11n])assert.equal(evaluate(result.expression,{X1}),evaluate(root,{X1}));
});
