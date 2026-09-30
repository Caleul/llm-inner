import { type Comparison, type Rational, rational, normalizeExactAffineComparison } from './direct-branch-domain.js';
import { decodeIeeeF16ToF32 } from './utils.js';

/** Backward condition reduction. These are path boundaries, not a model IR.
 * Invert the actual nearest-even cells before normalizing an exact affine
 * producer. A rounded transform must never be treated as real arithmetic.
 */
export type DirectRoundedFormat='f16'|'f32'|'f32-f16';
export interface DirectPreimage {op:Comparison;value:Rational}
export function exactNumberRational(x:number):Rational{
  if(!Number.isFinite(x))throw new Error('A finite condition boundary is required');
  if(x===0)return rational(0n);
  const data=new DataView(new ArrayBuffer(8));data.setFloat64(0,x,false);
  const hi=data.getUint32(0,false),lo=data.getUint32(4,false),e=(hi>>>20)&2047;
  const magnitude=(BigInt(hi&0xfffff)<<32n)|BigInt(lo);
  const significand=e===0?magnitude:(1n<<52n)|magnitude;
  const exponent=(e===0?-1022:e-1023)-52;
  const signed=hi>>>31?-significand:significand;
  return exponent>=0?rational(signed<<BigInt(exponent)):rational(signed,1n<<BigInt(-exponent));
}
function roundedPreimage(kind:'f16'|'f32',op:Comparison,rhs:Rational):DirectPreimage{
  const largest=kind==='f16'?31743:0x7f7fffff,sign=kind==='f16'?32768:0x80000000;
  const last=2*largest+2,data=new DataView(new ArrayBuffer(4));
  const code=(index:number)=>index-1<largest?sign+largest-(index-1):index-1-largest;
  const value=(index:number)=>{
    if(index===0)return -Infinity;if(index===last)return Infinity;
    if(kind==='f16')return decodeIeeeF16ToF32(code(index));
    data.setUint32(0,code(index),true);return data.getFloat32(0,true);
  };
  const compare=(index:number)=>{
    if(index===0)return -1;if(index===last)return 1;
    const x=exactNumberRational(value(index));
    const delta=x.numerator*rhs.denominator-rhs.numerator*x.denominator;
    return delta<0n?-1:delta>0n?1:0;
  };
  // For >=/<, find the first representable output >= rhs. For >/<=,
  // find the first output > rhs. Infinity is included as an IEEE output cell.
  const strict=op==='>'||op==='<=';
  let lo=0,hi=last;
  while(lo<hi){const mid=Math.floor((lo+hi)/2),c=compare(mid);
    if(c>0||(!strict&&c===0))hi=mid;else lo=mid+1;
  }
  const upper=lo,lower=upper-1;
  if(lower<0||upper>last)throw new Error('Undefined rounded comparison boundary');
  const overflow=kind==='f16'?65520:2**128-2**103;
  const midpoint=lower===0?-overflow:upper===last?overflow:(value(lower)+value(upper))/2;
  const upperEven=upper===last||(code(upper)&1)===0;
  const high=op==='>'||op==='>=';
  // At the midpoint the even output owns the equality, including the
  // maximum-finite to infinity transition and both signs.
  const normalized:Comparison=high?(upperEven?'>=':'>'):(upperEven?'<':'<=');
  return {op:normalized,value:exactNumberRational(midpoint)};
}
export function normalizeRoundedAffineComparison(format:DirectRoundedFormat,scale:Rational,offset:Rational,
  op:Comparison,rhs:Rational):DirectPreimage|boolean{
  let boundary=roundedPreimage(format==='f32'?'f32':'f16',op,rhs);
  if(format==='f32-f16')boundary=roundedPreimage('f32',boundary.op,boundary.value);
  return normalizeExactAffineComparison(scale,offset,boundary.op,boundary.value);
}

/** Invert the CPU policy F32(1 / F32(sqrt(F32(a*x+b)))) for positive
 * finite variances. Each target cell is inverted before reaching its producer;
 * sqrt therefore disappears from a consumer comparison instead of expanding
 * millions of unrelated output cells. The affine producer must be exact.
 */
export function normalizeReciprocalRootAffineComparison(scale:Rational,offset:Rational,
  op:Comparison,rhs:Rational):DirectPreimage|boolean{
  if(rhs.numerator<=0n)return op==='>'||op==='>=';
  const inverse=roundedPreimage('f32',op,rhs);
  if(inverse.value.numerator<=0n)return inverse.op==='>'||inverse.op==='>=';
  const reversed:Record<Comparison,Comparison>={'<':'>','<=':'>=','>':'<','>=':'<='};
  const rootLimit=rational(inverse.value.denominator,inverse.value.numerator);
  const root=roundedPreimage('f32',reversed[inverse.op],rootLimit);
  if(root.value.numerator<0n)return root.op==='>'||root.op==='>=';
  const square=rational(root.value.numerator*root.value.numerator,root.value.denominator*root.value.denominator);
  const variance=roundedPreimage('f32',root.op,square);
  return normalizeExactAffineComparison(scale,offset,variance.op,variance.value);
}
