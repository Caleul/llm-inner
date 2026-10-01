import assert from 'node:assert/strict';
import {test} from 'node:test';
import {exactNumberRational,finiteIeeeValue} from '../src/direct-round-preimage.js';
import {rational} from '../src/direct-branch-domain.js';
test('power-of-two boundary normalization matches independent generic GCD decoding',()=>{
 const bits=new DataView(new ArrayBuffer(8));
 const reference=(x:number)=>{
  if(x===0)return rational(0n);
  bits.setFloat64(0,x,false);const hi=bits.getUint32(0),lo=bits.getUint32(4),e=(hi>>>20)&2047;
  const mantissa=(BigInt(hi&0xfffff)<<32n)|BigInt(lo);
  const sig=(e===0?mantissa:(1n<<52n)|mantissa)*(hi>>>31?-1n:1n);
  const power=(e===0?-1022:e-1023)-52;
  return power>=0?rational(sig<<BigInt(power)):rational(sig,1n<<BigInt(-power));
 };
 for(let i=0;i<=63486;i++){const x=finiteIeeeValue('f16',i);assert.deepEqual(exactNumberRational(x),reference(x));}
 let seed=97;
 for(let i=0;i<20000;i++){
  seed=(Math.imul(seed,1664525)+1013904223)>>>0;bits.setUint32(0,seed);
  seed=(Math.imul(seed,1664525)+1013904223)>>>0;bits.setUint32(4,seed);
  const x=bits.getFloat64(0);if(Number.isFinite(x))assert.deepEqual(exactNumberRational(x),reference(x));
 }
 for(const x of [-0,Number.MIN_VALUE,-Number.MIN_VALUE,Number.MAX_VALUE,-Number.MAX_VALUE,2**-1022,2**52])assert.deepEqual(exactNumberRational(x),reference(x));
 assert.throws(()=>exactNumberRational(Infinity));assert.throws(()=>exactNumberRational(NaN));
});
