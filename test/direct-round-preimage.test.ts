import assert from 'node:assert/strict';
import {test} from 'node:test';
import {exactNumberRational,normalizeFiniteAffineRunComparison,normalizePositiveReciprocalComparison,normalizePositiveSqrtComparison,normalizeReciprocalRootAffineComparison,normalizeRoundedAffineComparison,type DirectPreimage} from '../src/direct-round-preimage.js';
import {rational} from '../src/direct-branch-domain.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';

const ieee=new DataView(new ArrayBuffer(8));
const f32=new DataView(new ArrayBuffer(4));
function roundedHalf(x:number){
  f32.setFloat32(0,x,true);const bits=f32.getUint32(0,true);
  if((bits&0x7fffffff)===0)return x;
  return decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(bits)));
}
function satisfies(x:number,p:DirectPreimage|boolean){
  if(typeof p==='boolean')return p;
  const a=exactNumberRational(x),delta=a.numerator*p.value.denominator-p.value.numerator*a.denominator;
  return p.op==='<'?delta<0n:p.op==='<='?delta<=0n:p.op==='>'?delta>0n:delta>=0n;
}
test('affine numerical runs invert emitted binary64 operations over the discrete producer',()=>{
  for(const [scale,offset] of [[1/3,0.1],[-1/7,2**40],[0,0.5]] as const){
    for(const rhs of [-1,0,0.5,1,2**40])for(const op of ['<','<=','>','>='] as const){
      const p=normalizeFiniteAffineRunComparison('f16',scale,offset,op,exactNumberRational(rhs));
      for(let bits=0;bits<65536;bits+=17){
        if((bits&0x7c00)===0x7c00)continue;
        const x=decodeIeeeF16ToF32(bits),y=scale*x+offset;
        const reference=op==='<'?y<rhs:op==='<='?y<=rhs:op==='>'?y>rhs:y>=rhs;
        assert.equal(satisfies(x,p),reference,`${scale} ${offset} ${op} ${rhs} ${x}`);
      }
    }
  }
});
test('positive reciprocal comparisons propagate before the root producer is expanded further',()=>{
  for(const numerator of [1,0.1,2**40])for(const rhs of [-1,0,0.1,1/3,1,100,2**50]){
    for(const op of ['<','<=','>','>='] as const){
      const p=normalizePositiveReciprocalComparison('f16',numerator,op,exactNumberRational(rhs));
      for(let bits=1;bits<31744;bits+=13){
        const x=decodeIeeeF16ToF32(bits),y=numerator/x;
        assert.equal(satisfies(x,p),op==='<'?y<rhs:op==='<='?y<=rhs:op==='>'?y>rhs:y>=rhs);
      }
    }
  }
});
test('direct corrected-root comparisons invert to the original positive F32 input',()=>{
  for(const rhs of [2**-74,0.001,1,1.00001,2,1e10,1e20])for(const op of ['<','<=','>','>='] as const){
    const p=normalizePositiveSqrtComparison(op,exactNumberRational(rhs));
    assert.notEqual(typeof p,'boolean');
    const cut=(p as DirectPreimage).value;
    f32.setFloat32(0,Number(cut.numerator)/Number(cut.denominator),true);
    const center=f32.getUint32(0,true);
    for(let code=Math.max(1,center-2);code<=Math.min(0x7f7fffff,center+2);code++){
      f32.setUint32(0,code,true);const x=f32.getFloat32(0,true),y=Math.fround(Math.sqrt(x));
      assert.equal(satisfies(x,p),op==='<'?y<rhs:op==='<='?y<=rhs:op==='>'?y>rhs:y>=rhs);
    }
  }
});
test('rounded affine conditions invert both rounding stages and tie ownership',()=>{
  for(let code=1;code<31743;code+=31){
    for(const sign of [1,-1]){
    const rhs=sign*decodeIeeeF16ToF32(code);
    for(const op of ['<','<=','>','>='] as const){
      const p=normalizeRoundedAffineComparison('f32-f16',rational(1n),rational(0n),op,exactNumberRational(rhs));
      assert.notEqual(typeof p,'boolean');const cut=(p as DirectPreimage).value;
      const x=Number(cut.numerator)/Number(cut.denominator);
      ieee.setFloat64(0,x,true);const bits=ieee.getBigUint64(0,true);
      for(const nearby of [bits-1n,bits,bits+1n]){
        ieee.setBigUint64(0,nearby,true);const input=ieee.getFloat64(0,true),y=roundedHalf(input);
        const reference=op==='<'?y<rhs:op==='<='?y<=rhs:op==='>'?y>rhs:y>=rhs;
        assert.equal(satisfies(input,p),reference,`${code} ${op} ${input}`);
      }
    }
    }
  }
});
test('backward rounded condition remains contradictory after exact affine substitution',()=>{
  const p=normalizeRoundedAffineComparison('f32',rational(4n),rational(0n),'<',rational(5n));
  assert.notEqual(typeof p,'boolean');
  assert.equal(satisfies(3.0001,p),false);
  assert.equal(satisfies(1,p),true);
  const negative=normalizeRoundedAffineComparison('f32',rational(-4n),rational(0n),'<',rational(5n));
  assert.equal(satisfies(-3.0001,negative),false);
  assert.equal(satisfies(-1,negative),true);
});

test('reciprocal-root consumer preimages preserve ordered CPU rounding',()=>{
  for(const rhs of [0.01,0.1,0.8,1,1.25,10,100,1e20]){
    for(const op of ['<','<=','>','>='] as const){
      const p=normalizeReciprocalRootAffineComparison(rational(1n),rational(0n),op,exactNumberRational(rhs));
      if(typeof p==='boolean')continue;
      const cut=Number(p.value.numerator)/Number(p.value.denominator);
      if(!(cut>0&&Number.isFinite(Math.fround(cut))))continue;
      f32.setFloat32(0,cut,true);const code=f32.getUint32(0,true);
      for(let nearby=Math.max(1,code-2);nearby<=Math.min(0x7f7fffff,code+2);nearby++){
        f32.setUint32(0,nearby,true);const input=f32.getFloat32(0,true);
        const y=Math.fround(1/Math.fround(Math.sqrt(input)));
        const reference=op==='<'?y<rhs:op==='<='?y<=rhs:op==='>'?y>rhs:y>=rhs;
        assert.equal(satisfies(input,p),reference,`${op} ${rhs} ${input}`);
      }
    }
  }
});
