import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o,type JsonExpression} from '../src/direct-json-expression.js';
import {simplifyJsonAffine,type JsonAffineDomains} from '../src/direct-json-affine.js';
import {sameJsonExpression,simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {createJsonModelLowerer} from '../src/direct-json-lower-model.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
const add=(a:JsonExpression,b:JsonExpression)=>o('add','f64',a,b),mul=(a:JsonExpression,k:number)=>o('mul','f64',a,c('f64',k));
function domains(nodes:JsonExpression[],bound=.5):JsonAffineDomains {
 const facts:JsonAffineDomains=new WeakMap();for(const n of nodes)facts.set(n,{minimum:-bound,maximum:bound,quantumExponent:-24,excludesNegativeZero:false});return facts;
}
test('affine substitution distributes coefficients and gathers repeated fundamental variables bitwise',()=>{
 const [x,y,z]=['X1','X2','X3'].map(n=>input('f64',n)) as [JsonExpression,JsonExpression,JsonExpression];
 const producer=add(add(mul(x,132),mul(y,3)),mul(z,4));
 const root=add(c('f64',0),add(add(mul(producer,14),mul(y,Math.fround(32.8))),mul(z,54)));
 const facts=domains([x,y,z]),stats={visited:0,rewrites:0,barriers:0};
 const result=simplifyJsonAffine(root,facts,stats);
 const expected=add(add(add(c('f64',0),mul(x,1848)),mul(y,42+Math.fround(32.8))),mul(z,110));
 assert.ok(sameJsonExpression(simplifyJsonFixedPoint(result).expression,expected));
 assert.ok(stats.rewrites>0);assert.ok(measureJsonExpression(result).serializedBytes<measureJsonExpression(root).serializedBytes);
 assert.equal(simplifyJsonAffine(result,facts),result);
 for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
   const a=decodeIeeeF16ToF32(bits);if(Math.abs(a)>.5)continue;
   const b=decodeIeeeF16ToF32((bits*17)&0xffff),d=decodeIeeeF16ToF32((bits*31)&0xffff);
   const values={X1:a,X2:Number.isFinite(b)&&Math.abs(b)<=.5?b:0,X3:Number.isFinite(d)&&Math.abs(d)<=.5?d:-0};
   assert.ok(Object.is(evaluate(root,values),evaluate(result,values)));
 }
});
test('floating rounding and signed-zero barriers survive affine analysis',()=>{
 const x=input('f64','X1'),f=domains([x],65504);
 const source=add(add(x,c('f64',2**53)),c('f64',-(2**53)));
 const result=simplifyJsonAffine(source,f);assert.ok(Object.is(evaluate(source,{X1:1}),evaluate(result,{X1:1})));
 assert.equal(evaluate(result,{X1:1}),0);
 const negativeZero=mul(add(x,x),3);assert.ok(Object.is(evaluate(simplifyJsonAffine(negativeZero,f),{X1:-0}),-0));
 const unproved=add(mul(x,2),mul(x,3));assert.equal(simplifyJsonAffine(unproved),unproved);
 const bits=o('reinterpret','u64',x),masked=o('and','u64',bits,c('u64',0xffffffffe0000000n));
 const rounded=o('reinterpret','f64',masked);assert.equal(simplifyJsonAffine(rounded,f),rounded);
});
test('every operator and guard is visited and branch bounds never leak to the complement',()=>{
 const x=input('f64','X1'),f=domains([x],65504);f.set(x,{minimum:0,maximum:65504,quantumExponent:-24,excludesNegativeZero:false});
 const body=add(c('f64',0),mul(add(mul(x,Math.fround(32.8)),mul(x,3)),14));
 const guard=o('le','bool',x,c('f64',.5)),root=o('if','f64',guard,body,body),stats={visited:0,rewrites:0,barriers:0};
 const result=simplifyJsonAffine(root,f,stats);assert.ok(stats.visited>=8);
 assert.ok(measureJsonExpression(result[3] as JsonExpression).serializedBytes<measureJsonExpression(body).serializedBytes);
 assert.equal(result[4],body);
 for(const value of [-0,0,2**-24,.5,1,65504])assert.ok(Object.is(evaluate(root,{X1:value}),evaluate(result,{X1:value})));
});

test('incremental F32 substitution composes exact affine producers before bitwise rounding',()=>{
 const x=input('f16','X1'),y=input('f16','X2'),z=input('f16','X3');
 const wide=(n:JsonExpression)=>o('widen','f32',n),sum=(a:JsonExpression,b:JsonExpression)=>o('add','f32',a,b);
 const scale=(a:JsonExpression,b:number)=>o('mul','f32',a,c('f32',b));
 const producer=sum(sum(scale(wide(x),132),scale(wide(y),3)),scale(wide(z),4));
 const root=sum(c('f32',0),scale(producer,14));
 const facts={halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),
   inputMagnitudeBounds:new Map(['X1','X2','X3'].map(n=>[n,{minimum:0,maximum:2**-12}]))};
 const closed=createJsonModelLowerer(facts).lower(root);
 const inspect=(node:JsonExpression):void=>{assert.notEqual(node[0],'reinterpret');if(node[0]!=='constant'&&node[0]!=='input')for(const arg of node.slice(2))inspect(arg as JsonExpression);};inspect(closed);
 for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
   const value=decodeIeeeF16ToF32(bits);if(Math.abs(value)>2**-12)continue;
   const values={X1:value,X2:-value,X3:value};assert.ok(Object.is(evaluate(root,values),evaluate(closed,values,{allowPendingPrimitives:false})));
 }
});

test('invalid affine certificates fail instead of narrowing a numerical domain',()=>{
 const x=input('f64','X1'),f=domains([x]);f.set(x,{minimum:1,maximum:0,quantumExponent:-24,excludesNegativeZero:false});
 assert.throws(()=>simplifyJsonAffine(add(x,c('f64',1)),f),/Invalid certified/);
});

test('affine forms reuse completed shared producers without enumerating literal copies',()=>{
 const x=input('f64','X1'),f=domains([x],0);let producer=x;
 for(let step=0;step<80;step++)producer=add(producer,producer);
 const root=add(c('f64',0),producer),stats={visited:0,rewrites:0,barriers:0};
 const result=simplifyJsonAffine(root,f,stats);
 assert.ok(stats.visited<1000);assert.ok(measureJsonExpression(result).serializedBytes<1000n);
 for(const value of [0,-0])assert.ok(Object.is(evaluate(root,{X1:value}),evaluate(result,{X1:value})));
});

test('an identity cannot weaken the half lattice of a completed producer reused by later affine consumers',()=>{
 const x=input('f16','X1'),wide=o('widen','f32',x),identity=o('add','f32',wide,c('f32',-0));
 const shared=o('widen','f64',identity),other=o('widen','f64',x);
 const root=add(c('f64',0),mul(add(mul(shared,132),mul(other,3)),14));
 const facts={halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()};
 const closed=createJsonModelLowerer(facts).lower(root),expected=add(c('f64',0),mul(input('f64','X1'),1890));
 assert.ok(sameJsonExpression(closed,expected));
 for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
   const values={X1:decodeIeeeF16ToF32(bits)};assert.ok(Object.is(evaluate(root,values),evaluate(closed,values,{allowPendingPrimitives:false})));
 }
});
