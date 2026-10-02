import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonOperation as o,jsonConstant as c} from '../src/direct-json-expression.js';
import {simplifyJsonSharedConditions} from '../src/direct-json-cofactor.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
const example=()=>{
  const p=input('bool','P'),q=input('bool','Q'),branch=(condition:typeof p,a:number,b:number)=>o('if','u32',condition,c('u32',a),c('u32',b));
  let repeated=branch(p,3,7);
  for(let i=1;i<8;i++)repeated=o('add','u32',repeated,branch(p,i+3,i+7));
  return o('add','u32',repeated,o('add','u32',branch(q,9,1),branch(q,2,3)));
};
test('shared condition reduction continues past one promotion to its own fixed point',()=>{
  const root=example(),first=simplifyJsonSharedConditions(root,{maxRounds:1}),full=simplifyJsonSharedConditions(root);
  assert.equal(first.stats.accepted,1);assert.equal(first.stats.converged,false);assert.equal(first.stats.stopReason,'round-budget');
  assert.ok(full.stats.accepted>=2);assert.ok(full.stats.afterBytes<first.stats.afterBytes);
  assert.equal(full.stats.converged,true);assert.equal(full.stats.stopReason,'fixed-point');
  for(const P of [true,false])for(const Q of [true,false])assert.equal(evaluate(full.expression,{P,Q}),evaluate(root,{P,Q}));
  const stable=simplifyJsonSharedConditions(full.expression);
  assert.equal(stable.stats.accepted,0);assert.equal(stable.stats.converged,true);assert.equal(stable.stats.beforeBytes,stable.stats.afterBytes);
});
test('condition candidate and round budgets never masquerade as convergence',()=>{
  const root=example(),result=simplifyJsonSharedConditions(root,{maxCandidates:0});
  assert.equal(result.stats.converged,false);assert.equal(result.stats.stopReason,'candidate-budget');assert.equal(result.stats.accepted,0);
  assert.throws(()=>simplifyJsonSharedConditions(root,{maxRounds:0}),/budget/);
  assert.throws(()=>simplifyJsonSharedConditions(root,{maxRounds:1.5}),/budget/);
  for(const P of [true,false])for(const Q of [true,false])assert.equal(evaluate(result.expression,{P,Q}),evaluate(root,{P,Q}));
});


test('later resource exhaustion preserves the already-admitted reduction without claiming convergence',()=>{
  const root=example();let reads=0;
  const result=simplifyJsonSharedConditions(root,{get maxUniqueNodes(){return reads++===0?100000:1;}});
  assert.equal(result.stats.accepted,1);assert.ok(result.stats.afterBytes<result.stats.beforeBytes);
  assert.equal(result.stats.converged,false);assert.equal(result.stats.stopReason,'resource-budget');
  assert.equal(result.stats.roundBudgetFailures,1);
  for(const P of [true,false])for(const Q of [true,false])assert.equal(evaluate(result.expression,{P,Q}),evaluate(root,{P,Q}));
});
