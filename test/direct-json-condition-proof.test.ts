import test from 'node:test';
import assert from 'node:assert/strict';
import {proveJsonFiniteComparison as prove} from '../src/direct-json-condition-proof.js';
import {jsonConstant as c,jsonInput as input,jsonOperation as o} from '../src/direct-json-expression.js';
import {createJsonModelLowerer} from '../src/direct-json-lower-model.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
const facts=()=>({halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()});

test('finite comparison proofs preserve endpoints, overlap, strictness and signed-zero equality',()=>{
  for(const operation of ['lt','le','eq'])for(let lo=-3;lo<=3;lo++)for(let hi=lo;hi<=3;hi++)
    for(let blo=-3;blo<=3;blo++)for(let bhi=blo;bhi<=3;bhi++){
      const result=prove(operation,{minimum:lo,maximum:hi},{minimum:blo,maximum:bhi});
      if(result!==undefined)for(let a=lo;a<=hi;a++)for(let b=blo;b<=bhi;b++)
        assert.equal(result,operation==='lt'?a<b:operation==='le'?a<=b:a===b);
    }
  assert.equal(prove('eq',{minimum:-0,maximum:0},{minimum:0,maximum:-0}),true);
  assert.equal(prove('lt',{minimum:-Infinity,maximum:0},{minimum:1,maximum:1}),undefined);
  assert.equal(prove('eq',undefined,{minimum:1,maximum:1}),undefined);
  assert.equal(prove('lt',{minimum:2,maximum:1},{minimum:1,maximum:1}),undefined);
});
test('a proved guard cuts unreachable numerical producers before lowering for every finite half input',()=>{
  const x=input('f16','X1'),guard=o('lt','bool',o('widen','f32',x),c('f32',70000));
  const dead=o('pending-exp','f32',c('f32',1)),root=o('if','f32',guard,o('widen','f32',x),dead),events:string[]=[];
  const result=createJsonModelLowerer(facts(),new WeakMap(),{incremental:true,onSubstitution:e=>events.push(e.operation)}).lower(root);
  assert.ok(!events.includes('pending-exp'));assert.deepEqual(result,input('f64','X1'));
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const value=decodeIeeeF16ToF32(bits);assert.ok(Object.is(evaluate(result,{X1:value},{allowPendingPrimitives:false}),value));
  }
});
test('a constant discovered during condition substitution selects its branch before unsupported producers',()=>{
  const x=input('f16','X1'),wide=o('widen','f32',x),zero=o('sub','f32',wide,wide);
  const guard=o('eq','bool',zero,c('f32',0)),dead=o('pending-exp','f32',c('f32',1));
  const result=createJsonModelLowerer(facts(),new WeakMap(),{incremental:true}).lower(o('if','f32',guard,c('f32',7),dead));
  assert.deepEqual(result,c('f64',7));
  assert.throws(()=>createJsonModelLowerer(facts()).lower(o('if','f32',o('lt','bool',wide,c('f32',0)),c('f32',7),dead)),/not proved/);
});
