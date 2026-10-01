import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonConstant as c,jsonInput as input,jsonOperation as op,type JsonExpression} from '../src/direct-json-expression.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {simplifyJsonExpression as simplify,sameJsonExpression,simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {lowerJsonF32ToF16,lowerJsonF64ToF32} from '../src/direct-json-f16.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {mkdtemp,readFile,rm,writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {auditJsonExpression,writeJsonScalarUnits,type JsonScalarHeader} from '../src/direct-json-stream.js';

test('JSON exact constants survive serialization including signed zero and all finite F16 bits',()=>{
  assert.ok(Object.is(evaluate(JSON.parse(JSON.stringify(c('f64',-0)))),-0));
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const expression:JsonExpression=['constant','f16','0x'+bits.toString(16).padStart(4,'0')];
    assert.ok(Object.is(evaluate(JSON.parse(JSON.stringify(expression))),decodeIeeeF16ToF32(bits)));
  }
});
test('JSON conditional facts remove repeated decisions without distributing branches',()=>{
  const x=input('u32','X1'),condition=op('lt','bool',x,c('u32',9));
  const expression=op('if','u32',condition,
    op('if','u32',condition,c('u32',3),c('u32',99)),
    op('if','u32',condition,c('u32',99),c('u32',4)));
  const reduced=simplify(expression);
  assert.ok(sameJsonExpression(reduced,op('if','u32',condition,c('u32',3),c('u32',4))));
  for(let n=0;n<20;n++)assert.equal(evaluate(reduced,{X1:BigInt(n)}),evaluate(expression,{X1:BigInt(n)}));
});
test('JSON fixed point propagates integer bounds to remove implied and contradictory conditions',()=>{
  const x=input('u32','X1');
  const expression=op('if','u32',op('lt','bool',x,c('u32',5)),
    op('if','u32',op('le','bool',x,c('u32',6)),c('u32',11),c('u32',99)),
    op('if','u32',op('lt','bool',x,c('u32',3)),c('u32',99),c('u32',12)));
  const {expression:reduced}=simplifyJsonFixedPoint(expression);
  assert.ok(sameJsonExpression(reduced,op('if','u32',op('lt','bool',x,c('u32',5)),c('u32',11),c('u32',12))));
  for(let i=0;i<20;i++)assert.equal(evaluate(expression,{X1:BigInt(i)}),evaluate(reduced,{X1:BigInt(i)}));
  assert.throws(()=>simplifyJsonFixedPoint(expression,1),/fixed point/);
});
test('JSON factoring is allowed for modular integers and withheld across float rounding',()=>{
  for(const type of ['u32','u64'] as const){
    const x=input(type,'X1'),a=c(type,0xffffffff),b=c(type,2);
    const expression=op('add',type,op('mul',type,x,a),op('mul',type,x,b));
    const reduced=simplify(expression);
    assert.equal(reduced[0],'mul');
    for(const n of [0n,1n,2n,0xffffffffn])assert.equal(evaluate(reduced,{X1:n}),evaluate(expression,{X1:n}));
  }
  const x=input('f32','X1'),a=c('f32',16777216),b=c('f32',-16777215);
  const expression=op('add','f32',op('mul','f32',x,a),op('mul','f32',x,b));
  assert.ok(sameJsonExpression(expression,simplify(expression)));
  const factored=op('mul','f32',x,op('add','f32',a,b));
  assert.notEqual(evaluate(expression,{X1:1.5}),evaluate(factored,{X1:1.5}));
});
test('JSON simplification preserves -0 and does not hide undefined integer operations',()=>{
  const x=input('f64','X1'),expression=op('add','f64',x,c('f64',0));
  assert.ok(Object.is(evaluate(simplify(expression),{X1:-0}),0));
  assert.ok(sameJsonExpression(expression,simplify(expression)));
  const invalid=op('div','u32',c('u32',1),c('u32',0));
  assert.throws(()=>evaluate(simplify(op('mul','u32',invalid,c('u32',0)))));
  const invalidCondition=op('eq','bool',invalid,c('u32',0));
  assert.throws(()=>evaluate(simplify(op('if','u32',invalidCondition,c('u32',2),c('u32',2)))));
  const invalidBound=op('lt','bool',invalid,c('u32',0));
  assert.throws(()=>evaluate(simplify(op('if','u32',invalidBound,c('u32',2),c('u32',3)))));
});
test('JSON rejects hidden floating conversions and malformed operation types',()=>{
  assert.throws(()=>op('convert','f16',input('f32','X1')),/Unlowered/);
  assert.throws(()=>op('and','f32',c('f32',1),c('f32',2)),/unsigned/);
  assert.throws(()=>op('reinterpret','f64',c('u32',1)),/width/);
  assert.throws(()=>evaluate(['unknown','f64'] as unknown as JsonExpression),/Unknown/);
});
test('JSON F32 to F16 bitwise conversion preserves every finite F16 value including signed zeros',()=>{
  const expression=lowerJsonF32ToF16(input('f32','X1'));
  assert.doesNotMatch(JSON.stringify(expression),/nearest|round|sqrt|exp/);
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const x=decodeIeeeF16ToF32(bits);
    assert.ok(Object.is(evaluate(expression,{X1:x}),x),`half bits ${bits.toString(16)}`);
  }
});
test('JSON F32 to F16 explicit decisions agree with independent dyadic rounding at ties and sampled F32 bits',()=>{
  const expression=lowerJsonF32ToF16(input('f32','X1'));
  const buffer=new DataView(new ArrayBuffer(4));
  const values=[-0,0,2**-25,2**-24,2**-14,1+2**-11,1+3*2**-11,65504,65520];
  let state=0xabcde123;
  for(let i=0;i<12000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    buffer.setUint32(0,state);const x=buffer.getFloat32(0);
    if(Number.isFinite(x))values.push(x);
  }
  for(const x of values){
    buffer.setFloat32(0,x);
    const bits=Object.is(x,-0)?0x8000:roundDyadicToF16IfElse(f32BitsToDyadic(buffer.getUint32(0)));
    assert.ok(Object.is(evaluate(expression,{X1:x}),decodeIeeeF16ToF32(bits)),`F32 ${x}`);
  }
});
test('JSON F64 to F32 bitwise conversion agrees with IEEE rounding at boundaries and sampled binary64 values',()=>{
  const expression=lowerJsonF64ToF32(input('f64','X1'));
  assert.doesNotMatch(JSON.stringify(expression),/nearest|round|sqrt|exp/);
  const values=[-0,0,Infinity,-Infinity,2**-150,-(2**-150),2**-149,2**-126,
    1+2**-24,1+3*2**-24,3.4028234663852886e38,3.4028235677973366e38];
  let state=0xabcd1234;
  for(let i=0;i<12000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    values.push((state/0xffffffff-.5)*2**((i%2200)-1100));
  }
  for(const x of values)if(!Number.isNaN(x))assert.ok(Object.is(evaluate(expression,{X1:x}),Math.fround(x)),`F64 ${x}`);
});
test('JSON admission distinguishes shared decisions from expanded occurrences and rejects residual implicit rounding',()=>{
  const x=input('f16','X1'),condition=op('lt','bool',x,['constant','f16','0x3c00']);
  const branch=op('if','f16',condition,x,['constant','f16','0x0000']);
  const expression=op('if','f16',condition,branch,branch);
  const bindings={X1:{dtype:'f16' as const,tokenPosition:0,coordinate:0}};
  const audit=auditJsonExpression(expression,bindings);
  assert.equal(audit.decisions,3);assert.equal(audit.uniqueDecisions,2);
  assert.throws(()=>auditJsonExpression(expression,bindings,2),/budget/);
  assert.throws(()=>auditJsonExpression(input('f16','X2'),bindings),/Undeclared/);
  assert.throws(()=>auditJsonExpression(op('add','f32',c('f32',1),c('f32',2)),{}),/Implicit/);
});
test('JSONL stream emits all scalar coordinates with exact syntax and refuses to publish incomplete vectors',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-json-'));
  const path=join(dir,'model.jsonl');
  const header:JsonScalarHeader={schema:'direct-scalar-json-v1',inputWidth:1,context:1,outputWidth:2,
    inputs:{X1:{dtype:'f16',tokenPosition:0,coordinate:0}}};
  try{
    const expression=op('if','f16',op('lt','bool',input('f16','X1'),['constant','f16','0x0000']),
      ['constant','f16','0x0000'],input('f16','X1'));
    async function* complete(){for(let dimension=0;dimension<2;dimension++)yield {position:0,dimension,expression};}
    const result=await writeJsonScalarUnits(path,header,complete());assert.equal(result.units,2);
    const original=await readFile(path,'utf8'),records=original.trim().split('\n').map(x=>JSON.parse(x));
    assert.deepEqual(records[1].expression,expression);assert.equal(records[3].finalParity,false);
    async function* incomplete(){yield {position:0,dimension:0,expression};}
    await assert.rejects(writeJsonScalarUnits(path,header,incomplete()),/Incomplete/);
    assert.equal(await readFile(path,'utf8'),original);
    await assert.rejects(writeJsonScalarUnits(path,header,complete(),{maxBytes:10}),/budget/);
    assert.equal(await readFile(path,'utf8'),original);
    await writeFile(path+'.compile.lock','do not modify');
  }finally{await rm(dir,{recursive:true,force:true});}
});
