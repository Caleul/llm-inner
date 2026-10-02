import test from 'node:test';
import assert from 'node:assert/strict';
import {jsonConstant as c,jsonInput as input,jsonOperation as op,type JsonExpression} from '../src/direct-json-expression.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {simplifyJsonExpression as simplify,sameJsonExpression,simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {lowerJsonF32ToF16,lowerJsonF64ToF32,lowerJsonRoundNormalF32AsF64} from '../src/direct-json-f16.js';
import {lowerJsonWiden} from '../src/direct-json-widen.js';
import {lowerJsonPositiveNormalSqrt,lowerJsonPositiveNormalSqrtAsF64} from '../src/direct-json-sqrt.js';
import {certifyJsonSmallSilu,lowerJsonSmallSilu,lowerJsonSmallSiluAsF64} from '../src/direct-json-silu.js';
import {lowerJsonSmallNonpositiveExp} from '../src/direct-json-exp.js';
import {lowerJsonFiniteF16AsF64,lowerJsonNormalF32ThenF16AsF64} from '../src/direct-json-half-value.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {simplifyJsonBitPrecision,type JsonPrecisionFacts} from '../src/direct-json-precision.js';
import {jsonModelRangeAnalysis} from '../src/direct-json-range.js';
import {simplifyJsonSharedConditions} from '../src/direct-json-cofactor.js';
import {foldDeclaredCpuF32Exponential} from '../src/direct-rust-numeric.js';
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
test('JSON shared conditions combine repeated queries after fixed point without changing floating path arithmetic',()=>{
  const x=input('f64','X1'),p=op('lt','bool',op('mul','f64',x,c('f64',3)),c('f64',7));
  const expression=op('add','f64',op('if','f64',p,c('f64',-0),c('f64',1e30)),
    op('if','f64',p,c('f64',-0),c('f64',-1e30)));
  const result=simplifyJsonSharedConditions(expression);
  assert.ok(result.stats.accepted>0);assert.ok(result.stats.afterBytes<result.stats.beforeBytes);
  for(const value of [-Infinity,-1,-0,0,1,3,Infinity,NaN])
    assert.ok(Object.is(evaluate(expression,{X1:value}),evaluate(result.expression,{X1:value})));
  const b=input('bool','P'),repeated=op('add','u32',op('if','u32',b,c('u32',3),c('u32',7)),
    op('if','u32',b,c('u32',4),c('u32',8)));
  const folded=simplifyJsonSharedConditions(repeated).expression;
  assert.ok(measureJsonExpression(folded).serializedBytes<measureJsonExpression(repeated).serializedBytes);
  for(const P of [true,false])assert.equal(evaluate(folded,{P}),evaluate(repeated,{P}));
});
test('JSON shared condition promotion does not expose lazy missing inputs or undefined integer operations',()=>{
  const p=op('eq','bool',input('u32','Missing'),c('u32',0));
  const inner=op('add','u32',op('if','u32',p,c('u32',3),c('u32',7)),
    op('if','u32',p,c('u32',4),c('u32',8)));
  const expression=op('if','u32',input('bool','Gate'),inner,c('u32',0));
  const result=simplifyJsonSharedConditions(expression);
  assert.ok(result.stats.unsafe>0);assert.equal(evaluate(result.expression,{Gate:false}),0n);
  const invalid=op('eq','bool',op('div','u32',c('u32',1),input('u32','Z')),c('u32',0));
  const undefinedRoot=op('add','u32',op('if','u32',invalid,c('u32',3),c('u32',7)),
    op('if','u32',invalid,c('u32',4),c('u32',8)));
  const unchanged=simplifyJsonSharedConditions(undefinedRoot);
  assert.equal(unchanged.stats.accepted,0);assert.ok(unchanged.stats.unsafe>0);
  assert.throws(()=>evaluate(unchanged.expression,{Z:0n}));
  assert.throws(()=>simplifyJsonSharedConditions(inner,{maxCandidates:-1}),/budget/);
  assert.throws(()=>simplifyJsonSharedConditions(inner,{maxUniqueNodes:2}),/budget/);
  assert.throws(()=>simplify(inner,undefined,[],1),/budget/);
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
test('JSON precision cancels redundant bit rounding only with a proved lattice and preserves signed zeros',()=>{
  const x=input('f64','X1'),rounded=lowerJsonRoundNormalF32AsF64(x);
  const facts:JsonPrecisionFacts=new WeakMap([[x,42]]),stats={redundantMasks:0,redundantRoundings:0};
  const reduced=simplify(simplifyJsonBitPrecision(rounded,facts,stats));
  assert.ok(sameJsonExpression(reduced,x));assert.equal(stats.redundantRoundings,1);
  // Without a certificate, an arbitrary F64 input must retain its rounding.
  assert.ok(sameJsonExpression(simplifyJsonBitPrecision(rounded),rounded));
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const value=decodeIeeeF16ToF32(bits);
    assert.ok(Object.is(evaluate(rounded,{X1:value}),evaluate(reduced,{X1:value})));
  }
  const twice=lowerJsonRoundNormalF32AsF64(rounded);
  const once=simplify(simplifyJsonBitPrecision(twice));
  assert.ok(sameJsonExpression(once,rounded));
  for(const value of [-0,0,1+2**-25,1+3*2**-24,-123.456])
    assert.ok(Object.is(evaluate(once,{X1:value}),evaluate(twice,{X1:value})));
  // A redundant mask retains an undefined operand rather than hiding it.
  const invalid=op('shl','u64',c('u64',1),c('u64',64));
  assert.throws(()=>evaluate(simplifyJsonBitPrecision(op('and','u64',invalid,c('u64',0xffffffffffffffffn)))));
});
test('JSON certified F16 source ranges eliminate only unreachable sign, subnormal and overflow arms',()=>{
  const x=input('f64','X1'),generic=lowerJsonFiniteF16AsF64(x);
  const ranges=[{minimum:-.1,maximum:.1},{minimum:2**-25,maximum:2**-15},
    {minimum:-2,maximum:-1},{minimum:-65504,maximum:65504},{minimum:0,maximum:0}];
  for(const range of ranges){
    const reduced=lowerJsonFiniteF16AsF64(x,range);
    assert.ok(measureJsonExpression(reduced).uniqueDecisions<measureJsonExpression(generic).uniqueDecisions);
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const value=decodeIeeeF16ToF32(bits);
      if(value>=range.minimum&&value<=range.maximum)
        assert.ok(Object.is(evaluate(reduced,{X1:value}),evaluate(generic,{X1:value})));
    }
    // Non-F16 F32 values exercise rounding inside the certified intervals.
    for(let i=0;i<1000;i++){
      const value=Math.fround(range.minimum+(range.maximum-range.minimum)*i/999);
      if(value>=range.minimum&&value<=range.maximum)
        assert.ok(Object.is(evaluate(reduced,{X1:value}),evaluate(generic,{X1:value})));
    }
  }
  assert.throws(()=>lowerJsonFiniteF16AsF64(x,{minimum:1,maximum:0}),/interval/);
});
test('JSON working range analysis encloses F32 endpoint rounding and refuses unknown or zero-crossing divisors',()=>{
  const facts={halfSources:new WeakMap(),positiveNormalRoots:new WeakSet<JsonExpression>(),
    exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap()};
  const range=jsonModelRangeAnalysis(facts),x=input('f16','X1'),wide=op('widen','f32',x);
  const square=op('mul','f32',wide,wide),r=range(square)!;
  assert.equal(r.minimum,0);assert.ok(r.maximum>=65504**2);
  const sum=op('add','f32',square,c('f32',2**-126)),s=range(sum)!;
  assert.ok(s.minimum>0);assert.ok(s.maximum>=r.maximum);
  assert.equal(range(op('div','f32',c('f32',1),wide)),undefined);
  assert.equal(range(input('f64','Unknown')),undefined);
  assert.equal(range(c('bool',true)),undefined);
  const source=op('mul','f32',wide,c('f32',.0001)),half=lowerJsonF32ToF16(source);
  facts.halfSources.set(half,source);
  const h=range(half)!;
  for(const value of [-65504,-(2**-24),-0,0,2**-24,65504]){
    const actual=Number(evaluate(half,{X1:value}));assert.ok(actual>=h.minimum&&actual<=h.maximum);
  }
});
test('JSON normal double rounding is fused with both widened even and odd tie cells preserved',()=>{
  const x=input('f64','X1'),fused=lowerJsonNormalF32ThenF16AsF64(x);
  const separate=lowerJsonFiniteF16AsF64(lowerJsonRoundNormalF32AsF64(x));
  const buffer=new DataView(new ArrayBuffer(8));
  const f32=new DataView(new ArrayBuffer(4));
  let comparisons=0;
  // Each positive normal half midpoint, both F32 tie-cell edges and the
  // adjacent binary64 values. Repeat with both signs; exclude overflow.
  for(let bits=0x0400;bits<0x7bff;bits++){
    const a=decodeIeeeF16ToF32(bits),b=decodeIeeeF16ToF32(bits+1),midpoint=(a+b)/2;
    buffer.setFloat64(0,midpoint);const middle=buffer.getBigUint64(0);
    for(const offset of [-(1n<<28n),1n<<28n])for(const adjacent of [-1n,0n,1n]){
      buffer.setBigUint64(0,middle+offset+adjacent);const value=buffer.getFloat64(0);
      for(const sign of [-1,1]){
        const actual=evaluate(fused,{X1:sign*value});
        assert.ok(Object.is(actual,evaluate(separate,{X1:sign*value})),
          `half midpoint ${bits.toString(16)}, edge ${offset}, adjacent ${adjacent}, sign ${sign}`);
        f32.setFloat32(0,sign*value);
        const expected=decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(f32.getUint32(0))));
        assert.ok(Object.is(actual,expected),'independent IEEE F32 and dyadic F16 oracle');
        comparisons++;
      }
    }
  }
  assert.equal(comparisons,368628);
  // Ordinary direct F64->F16 rounding differs at these double-rounding cells.
  const value=1+2**-11+2**-25;
  assert.equal(evaluate(fused,{X1:value}),1);
});
test('JSON rejects hidden floating conversions and malformed operation types',()=>{
  assert.throws(()=>op('convert','f16',input('f32','X1')),/Unlowered/);
  assert.throws(()=>op('and','f32',c('f32',1),c('f32',2)),/unsigned/);
  assert.throws(()=>op('reinterpret','f64',c('u32',1)),/width/);
  assert.throws(()=>evaluate(['unknown','f64'] as unknown as JsonExpression),/Unknown/);
  assert.throws(()=>evaluate(op('pending-exp','f32',c('f32',0)),{}, {allowPendingPrimitives:false}),/closed/);
  assert.throws(()=>evaluate(op('add','f32',c('f32',1),c('f32',2)),{}, {allowPendingPrimitives:false}),/closed/);
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
test('JSON exact widening covers every finite F16 bit pattern and sampled F32 normal/subnormal values',()=>{
  for(const target of ['f32','f64'] as const){
    const expression=lowerJsonWiden(input('f16','X1'),target);
    for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
      const x=decodeIeeeF16ToF32(bits);assert.ok(Object.is(evaluate(expression,{X1:x}),x));
    }
  }
  const expression=lowerJsonWiden(input('f32','X1'),'f64'),buffer=new DataView(new ArrayBuffer(4));
  let state=0x12345678;
  for(let i=0;i<12000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;buffer.setUint32(0,state);const x=buffer.getFloat32(0);
    if(Number.isFinite(x))assert.ok(Object.is(evaluate(expression,{X1:x}),x));
  }
});
test('JSON positive normal F32 square root is lowered without native sqrt and agrees bitwise on exponent and mantissa boundaries',()=>{
  const expression=lowerJsonPositiveNormalSqrt(input('f32','X1'));
  const composed=lowerJsonPositiveNormalSqrtAsF64(input('f64','X1'));
  assert.equal(measureJsonExpression(composed).decisions,0n);
  assert.doesNotMatch(JSON.stringify(expression),/pending|sqrt|round|widen/);
  const buffer=new DataView(new ArrayBuffer(4));
  const patterns:number[]=[];
  for(let exponent=1;exponent<255;exponent++)for(const fraction of [0,1,2,0x3fffff,0x400000,0x7ffffe,0x7fffff])
    patterns.push(exponent*2**23+fraction);
  let state=0x12345678;
  for(let i=0;i<12000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    const bits=state&0x7fffffff;if(bits>=0x00800000&&bits<0x7f800000)patterns.push(bits);
  }
  for(const bits of patterns){buffer.setUint32(0,bits);const x=buffer.getFloat32(0);
    assert.ok(Object.is(evaluate(expression,{X1:x}),Math.fround(Math.sqrt(x))),`sqrt bits ${bits.toString(16)}`);
    assert.ok(Object.is(evaluate(composed,{X1:x}),Math.fround(Math.sqrt(x))),`composed sqrt ${bits.toString(16)}`);
  }
});
test('JSON factorized SiLU polynomial is certified against every finite F16 point in its declared domain',()=>{
  const certificate=certifyJsonSmallSilu(.1);assert.equal(certificate.checkedPoints,23758);
  assert.equal(certificate.polynomialDegree,4);
  assert.equal(certifyJsonSmallSilu(.03).polynomialDegree,2);
  // -0.031005859375 is a real counterexample to the quadratic candidate.
  assert.equal(certifyJsonSmallSilu(.0311).polynomialDegree,4);
  assert.match(certificate.profileSha256,/^[a-f0-9]{64}$/);
  const expression=lowerJsonSmallSilu(input('f16','X1'));
  assert.doesNotMatch(JSON.stringify(expression),/pending|silu|exp|widen|round/);
  assert.ok(Object.is(evaluate(expression,{X1:-0}),-0));
  const closed=lowerJsonSmallSiluAsF64(input('f64','X1'));
  assert.ok(Object.is(evaluate(closed,{X1:-0}),-0));
  assert.equal(evaluate(closed,{X1:2**-24}),0);
  assert.ok(Object.is(evaluate(closed,{X1:-(2**-24)}),-0));
  assert.ok(measureJsonExpression(closed).occurrences<150n);
  assert.throws(()=>certifyJsonSmallSilu(.100001),/proven/);
});
test('JSON direct F64 to F16 composition matches an independent dyadic oracle at every half midpoint',()=>{
  const expression=lowerJsonFiniteF16AsF64(input('f64','X1')),word=new DataView(new ArrayBuffer(8));
  function check(value:number){
    word.setFloat64(0,value);const bits=word.getBigUint64(0),e=Number((bits>>52n)&0x7ffn);
    const coefficient=((e?1n<<52n:0n)|(bits&0xfffffffffffffn))*(bits>>63n?-1n:1n);
    const expected=Object.is(value,-0)?-0:decodeIeeeF16ToF32(roundDyadicToF16IfElse({coefficient,exponent:e?e-1075:-1074}));
    assert.ok(Object.is(evaluate(expression,{X1:value}),expected),`direct F64 bits ${bits.toString(16)}`);
  }
  for(let bits=0;bits<0x7bff;bits++){
    const middle=(decodeIeeeF16ToF32(bits)+decodeIeeeF16ToF32(bits+1))/2;
    word.setFloat64(0,middle);const raw=word.getBigUint64(0);
    for(const adjacent of [-1n,0n,1n]){
      word.setBigUint64(0,raw+adjacent);const value=word.getFloat64(0);check(value);check(-value);
    }
  }
  for(const value of [-0,0,Number.MIN_VALUE,-Number.MIN_VALUE,65520,-65520,Number.MAX_VALUE,-Number.MAX_VALUE])check(value);
});
test('JSON exponential eliminates constant range reduction and reproduces the pinned F32 polynomial in its proved domain',()=>{
  const expression=lowerJsonSmallNonpositiveExp(input('f32','X1'));
  assert.doesNotMatch(JSON.stringify(expression),/pending|exp|widen|round/);
  const samples=[-0,0,Math.fround(-.34),-(2**-25),-(2**-24),-(2**-149),-(2**-126)];
  let state=0x12345678;
  for(let i=0;i<12000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;samples.push(Math.fround(-.34*state/0xffffffff));
  }
  for(const x of samples)assert.ok(Object.is(evaluate(expression,{X1:x}),foldDeclaredCpuF32Exponential(x)),`exp ${x}`);
});
test('JSON normal rounding followed by exact widening collapses to one bitword rounding expression',()=>{
  const expression=lowerJsonRoundNormalF32AsF64(input('f64','X1'));
  for(const x of [-0,0])assert.ok(Object.is(evaluate(expression,{X1:x}),x));
  for(const sign of [-1,1])for(let exponent=-120;exponent<120;exponent++)
    for(const fraction of [1,1+2**-24,1+3*2**-24,1.99999999999999]){
      const x=sign*fraction*2**exponent;
      assert.ok(Object.is(evaluate(expression,{X1:x}),Math.fround(x)));
    }
});
test('JSON composed F16 conversion and widening preserves all F16 values and sampled F32 rounding boundaries',()=>{
  const expression=lowerJsonFiniteF16AsF64(input('f64','X1'));
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const x=decodeIeeeF16ToF32(bits);assert.ok(Object.is(evaluate(expression,{X1:x}),x));
  }
  const buffer=new DataView(new ArrayBuffer(4)),samples=[-0,0,2**-25,2**-24,2**-14,1+2**-11,65520,-65520];
  let state=0x12345678;
  for(let i=0;i<12000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;buffer.setUint32(0,state);const x=buffer.getFloat32(0);
    if(Number.isFinite(x))samples.push(x);
  }
  for(const x of samples){buffer.setFloat32(0,x);
    const half=Object.is(x,-0)?0x8000:roundDyadicToF16IfElse(f32BitsToDyadic(buffer.getUint32(0)));
    assert.ok(Object.is(evaluate(expression,{X1:x}),decodeIeeeF16ToF32(half)),`composed F16 ${x}`);
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
  assert.equal(audit.inputReferences,5);
  assert.throws(()=>auditJsonExpression(expression,bindings,2),/budget/);
  assert.throws(()=>auditJsonExpression(input('f16','X2'),bindings),/Undeclared/);
  assert.throws(()=>auditJsonExpression(op('add','f32',c('f32',1),c('f32',2)),{}),/Implicit/);
});
test('JSON admission rejects exponential duplication and longest shared paths without expanding the tree',()=>{
  const x=input('f64','X1'),bindings={X1:{dtype:'f64' as const,inputDtype:'f16' as const,tokenPosition:0,coordinate:0}};
  let duplicated=x;for(let i=0;i<100;i++)duplicated=op('add','f64',duplicated,duplicated);
  assert.equal(measureJsonExpression(duplicated).inputReferences,1n<<100n);
  assert.throws(()=>auditJsonExpression(duplicated,bindings),/Expanded expression budget/);
  let shared=x;for(let i=0;i<300;i++)shared=op('add','f64',shared,c('f64',0));
  let deep=shared;for(let i=0;i<300;i++)deep=op('add','f64',deep,c('f64',0));
  // The shared tail is seen first on the shorter path; memoization cannot
  // conceal the longest path through it from the depth budget.
  const root=op('add','f64',shared,deep);
  assert.equal(measureJsonExpression(root).depth,602);
  assert.throws(()=>auditJsonExpression(root,bindings),/depth budget/);
  assert.throws(()=>measureJsonExpression(root,1_000_000,128),/depth budget/);
});
test('JSON measurement predicts serialization and huge duplication without rendering expanded expressions',()=>{
  const x=input('f64','X1'),condition=op('lt','bool',x,c('f64',0));
  const expression=op('if','f64',condition,x,op('mul','f64',x,c('f64',2)));
  const measured=measureJsonExpression(expression);
  assert.equal(measured.serializedBytes,BigInt(Buffer.byteLength(JSON.stringify(expression))));
  let duplicated=x;
  for(let i=0;i<100;i++)duplicated=op('add','f64',duplicated,duplicated);
  const huge=measureJsonExpression(duplicated);
  assert.equal(huge.uniqueNodes,101);assert.equal(huge.occurrences,(1n<<101n)-1n);
  assert.throws(()=>measureJsonExpression(duplicated,10),/budget/);
  const simplified=simplifyJsonFixedPoint(duplicated);
  assert.equal(measureJsonExpression(simplified.expression).uniqueNodes,101);
  assert.ok(sameJsonExpression(simplified.expression,duplicated));
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
    let huge:JsonExpression=input('f16','X1');
    const condition=op('lt','bool',input('f16','X1'),['constant','f16','0x0000']);
    for(let i=0;i<100;i++)huge=op('if','f16',condition,huge,huge);
    async function* oversized(){yield {position:0,dimension:0,expression:huge};}
    await assert.rejects(writeJsonScalarUnits(path,header,oversized()),/before expression emission/);
    const status=JSON.parse(await readFile(path+'.status.json','utf8'));
    assert.equal(status.units,0);assert.equal(status.finalParity,false);
    assert.equal(status.rejectedCoordinate.dimension,0);
    assert.ok(BigInt(status.rejectedCoordinate.predictedOccurrences)>1_000_000n);
    assert.ok((await readFile(path+'.draft')).length<65536);
    assert.equal(await readFile(path,'utf8'),original);
    await writeFile(path+'.compile.lock','do not modify');
  }finally{await rm(dir,{recursive:true,force:true});}
});
