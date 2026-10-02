import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonInput as input,jsonConstant as c,jsonOperation as o,type JsonExpression} from '../src/direct-json-expression.js';
import {jsonIntegerRange} from '../src/direct-json-integer-range.js';
import {sameJsonExpression,simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';

test('unsigned interval propagation encloses every u16 operand through wrap, masks, shifts and conversion',()=>{
  const x=input('u16','X1'),u=(n:number)=>c('u16',n);
  const expressions=[o('add','u16',x,u(65530)),o('sub','u16',x,u(40)),o('mul','u16',x,u(3)),
    o('div','u16',x,u(7)),o('and','u16',x,u(127)),o('or','u16',x,u(128)),o('xor','u16',x,u(1024)),
    o('shl','u16',x,u(12)),o('shr','u16',x,u(6)),o('convert','u32',x)];
  for(const facts of [[],[[o('lt','bool',x,u(32)),true] as const],[[o('lt','bool',x,u(32)),false] as const]]){
    const allowed=jsonIntegerRange(x,facts,sameJsonExpression)!;
    for(const expression of expressions){
      const interval=jsonIntegerRange(expression,facts,sameJsonExpression)!;
      for(let X1=allowed[0];X1<=allowed[1];X1++){
        const value=evaluate(expression,{X1}) as bigint;
        assert.ok(value>=interval[0]&&value<=interval[1],`${expression[0]}, ${X1}, ${value}`);
      }
    }
  }
  assert.deepEqual(jsonIntegerRange(o('sub','u16',x,u(40)),[[o('lt','bool',x,u(32)),true]],sameJsonExpression),[65496n,65527n]);
  assert.throws(()=>jsonIntegerRange(x,[],sameJsonExpression,0),/budget/);
  assert.throws(()=>jsonIntegerRange(expressions[0]!,[],sameJsonExpression,1),/budget/);
});

test('derived integer comparisons inherit path bounds without losing wrap or lazy failure semantics',()=>{
  const x=input('u16','X1'),u=(n:number)=>c('u16',n),gate=o('lt','bool',x,u(32));
  const derived=o('add','u16',o('shl','u16',x,u(1)),u(3));
  const root=o('if','u16',gate,o('if','u16',o('lt','bool',derived,u(70)),u(3),u(4)),u(5));
  const closed=simplifyJsonFixedPoint(root).expression;
  assert.equal(measureJsonExpression(closed).uniqueDecisions,1);
  for(let X1=0n;X1<=65535n;X1++)assert.equal(evaluate(closed,{X1}),evaluate(root,{X1}));
  const invalid=o('div','u16',u(1),x),unsafe=o('if','u16',gate,
    o('if','u16',o('lt','bool',invalid,u(65535)),u(3),u(4)),u(5));
  const kept=simplifyJsonFixedPoint(unsafe).expression;
  assert.throws(()=>evaluate(kept,{X1:0n}),/zero/i);assert.equal(evaluate(kept,{X1:32n}),5n);
});

test('positive IEEE magnitude classification equals unsigned word ordering including zeros and nonfinite payloads',()=>{
  const bits=input('u64','Bits'),mag=o('and','u64',bits,c('u64',0x7fffffffffffffffn));
  const word=new DataView(new ArrayBuffer(8));
  for(const threshold of [2**-14,65520]){
    word.setFloat64(0,threshold);const boundary=word.getBigUint64(0);
    const floating=o('lt','bool',o('reinterpret','f64',mag),c('f64',threshold));
    const integer=o('lt','bool',mag,c('u64',boundary));
    const check=(Bits:bigint)=>assert.equal(evaluate(integer,{Bits}),evaluate(floating,{Bits}));
    for(let b=0;b<65536;b++){word.setFloat64(0,decodeIeeeF16ToF32(b));check(word.getBigUint64(0));}
    for(const value of [boundary-1n,boundary,boundary+1n,0x7ff0000000000000n,0x7ff0000000000001n,
      0x7ff8000000000001n,0xffffffffffffffffn])check(value);
    let state=0x123456789abcdef0n;
    for(let i=0;i<4096;i++){state=BigInt.asUintN(64,state*6364136223846793005n+1442695040888963407n);check(state);}
  }
});
