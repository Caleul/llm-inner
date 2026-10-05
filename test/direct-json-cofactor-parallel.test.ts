import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonOperation as o,jsonConstant as c} from '../src/direct-json-expression.js';
import {sameJsonExpression} from '../src/direct-json-simplify.js';
import {simplifyJsonSharedConditions} from '../src/direct-json-cofactor.js';
import {simplifyJsonSharedConditionsParallel} from '../src/direct-json-cofactor-parallel.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
const example=()=>{
  const p=input('bool','P'),q=input('bool','Q'),branch=(condition:typeof p,a:number,b:number)=>o('if','u32',condition,c('u32',a),c('u32',b));
  let repeated=branch(p,3,7);for(let i=1;i<8;i++)repeated=o('add','u32',repeated,branch(p,i+3,i+7));
  return o('add','u32',repeated,o('add','u32',branch(q,9,1),branch(q,2,3)));
};
test('parallel condition workers preserve alias facts, stable candidate choice and serial convergence',async()=>{
  const root=example(),serial=simplifyJsonSharedConditions(root);
  for(const workers of [1,2,4]){
    const result=await simplifyJsonSharedConditionsParallel(root,{workers});
    assert.deepEqual(result.stats,serial.stats);assert.ok(sameJsonExpression(result.expression,serial.expression));
    for(const P of [true,false])for(const Q of [true,false])assert.equal(evaluate(result.expression,{P,Q}),evaluate(root,{P,Q}));
  }
});
test('parallel conditions retain lazy missing inputs, undefined integer operations and signed floating paths',async()=>{
  const missing=o('eq','bool',input('u32','Missing'),c('u32',0));
  const lazy=o('if','u32',input('bool','Gate'),o('add','u32',o('if','u32',missing,c('u32',3),c('u32',7)),o('if','u32',missing,c('u32',4),c('u32',8))),c('u32',0));
  const x=input('f64','X1'),p=o('lt','bool',o('mul','f64',x,c('f64',3)),c('f64',7));
  const floating=o('add','f64',o('if','f64',p,c('f64',-0),c('f64',1e30)),o('if','f64',p,c('f64',-0),c('f64',-1e30)));
  const invalid=o('eq','bool',o('div','u32',c('u32',1),input('u32','Z')),c('u32',0));
  const undefinedRoot=o('add','u32',o('if','u32',invalid,c('u32',3),c('u32',7)),o('if','u32',invalid,c('u32',4),c('u32',8)));
  for(const root of [lazy,floating,undefinedRoot]){
    const serial=simplifyJsonSharedConditions(root),parallel=await simplifyJsonSharedConditionsParallel(root,{workers:3});
    assert.deepEqual(parallel.stats,serial.stats);assert.ok(sameJsonExpression(parallel.expression,serial.expression));
    if(root===lazy)assert.equal(evaluate(parallel.expression,{Gate:false}),0n);
    else if(root===undefinedRoot)assert.throws(()=>evaluate(parallel.expression,{Z:0n}),/zero/i);
    else for(const value of [-Infinity,-1,-0,0,1,3,Infinity,NaN])assert.ok(Object.is(evaluate(parallel.expression,{X1:value}),evaluate(root,{X1:value})));
  }
});
test('parallel condition budgets match serial incomplete results and retain earlier progress',async()=>{
  const root=example();
  for(const options of [{maxCandidates:0},{maxRounds:1}]){
    const serial=simplifyJsonSharedConditions(root,options),parallel=await simplifyJsonSharedConditionsParallel(root,{...options,workers:2});
    assert.deepEqual(parallel.stats,serial.stats);assert.ok(sameJsonExpression(parallel.expression,serial.expression));
  }
  let reads=0;const result=await simplifyJsonSharedConditionsParallel(root,{workers:2,get maxUniqueNodes(){return reads++===0?100000:1;}});
  assert.equal(result.stats.accepted,1);assert.equal(result.stats.roundBudgetFailures,1);assert.equal(result.stats.stopReason,'resource-budget');
  for(const P of [true,false])for(const Q of [true,false])assert.equal(evaluate(result.expression,{P,Q}),evaluate(root,{P,Q}));
  await assert.rejects(simplifyJsonSharedConditionsParallel(root,{workers:0}),/budget/);
});

test('ineligible lazy decisions do not consume the eligible candidate budget',async()=>{
  const gate=input('bool','Gate'),missing=input('bool','Missing');
  const branch=(condition:typeof gate,a:number,b:number)=>o('if','u32',condition,c('u32',a),c('u32',b));
  let lazy=branch(missing,3,7);
  for(let i=0;i<12;i++)lazy=o('add','u32',lazy,branch(missing,i+4,i+8));
  const gated=(offset:number)=>o('if','u32',gate,o('add','u32',lazy,c('u32',offset)),c('u32',offset+40));
  const root=o('add','u32',gated(1),o('add','u32',gated(2),gated(3)));
  const serial=simplifyJsonSharedConditions(root,{maxCandidates:1});
  assert.ok(serial.stats.unsafe>0);assert.ok(serial.stats.accepted>0);
  assert.equal(serial.stats.testedCandidates,1);
  assert.equal(serial.stats.candidates,serial.stats.testedCandidates+serial.stats.unsafe);
  assert.equal(serial.stats.converged,true);assert.equal(serial.stats.stopReason,'fixed-point');
  for(const workers of [2,4]){
    const parallel=await simplifyJsonSharedConditionsParallel(root,{maxCandidates:1,workers});
    assert.deepEqual(parallel.stats,serial.stats);assert.ok(sameJsonExpression(parallel.expression,serial.expression));
    for(const Gate of [false,true])for(const Missing of [false,true]){
      const values=Gate?{Gate,Missing}:{Gate};
      assert.equal(evaluate(parallel.expression,values),evaluate(root,values));
    }
  }
  // No eligible repeated decisions remain: a zero transformation budget is
  // sufficient to prove closure, and absent inputs stay absent.
  const unsafeOnly=o('if','u32',gate,lazy,c('u32',0));
  const zero=simplifyJsonSharedConditions(unsafeOnly,{maxCandidates:0});
  assert.equal(zero.stats.converged,true);assert.ok(zero.stats.unsafe>0);
  assert.equal(zero.stats.testedCandidates,0);
  assert.equal(evaluate(zero.expression,{Gate:false}),0n);
  assert.deepEqual((await simplifyJsonSharedConditionsParallel(unsafeOnly,{workers:2,maxCandidates:0})).stats,zero.stats);
});
