import { type Comparison, type Rational, rational, normalizeExactAffineComparison } from './direct-branch-domain.js';
import { decodeIeeeF16ToF32 } from './utils.js';

/** Backward condition reduction. These are path boundaries, not a model IR.
 * Invert the actual nearest-even cells before normalizing an exact affine
 * producer. A rounded transform must never be treated as real arithmetic.
 */
export type DirectRoundedFormat='f16'|'f32'|'f32-f16';
export interface DirectPreimage {op:Comparison;value:Rational}
export function finiteIeeeValue(kind:'f16'|'f32',index:number):number{
  const largest=kind==='f16'?31743:0x7f7fffff;
  if(index<0||index>2*largest||!Number.isInteger(index))throw new RangeError('Finite IEEE index');
  const rank=index-largest,code=rank<0?(kind==='f16'?32768:0x80000000)-rank:rank;
  if(kind==='f16')return decodeIeeeF16ToF32(code);
  const data=new DataView(new ArrayBuffer(4));data.setUint32(0,code,true);return data.getFloat32(0,true);
}
/** Actual binary64 operation on a discrete IEEE producer. Binary search
 * evaluates its emitted arithmetic, including rounding; it does not assume
 * real-number affine identities. No values or source fragments are cached.
 */
export function normalizeFiniteArithmeticComparison(kind:'f16'|'f32',operation:'+'|'-'|'*'|'/',
  constant:number,op:Comparison,rhs:Rational):DirectPreimage|boolean{
  if(!Number.isFinite(constant)||(operation==='/'&&constant===0))throw new Error('Undefined finite arithmetic preimage');
  const largest=kind==='f16'?31743:0x7f7fffff,last=2*largest;
  const negative=(operation==='*'||operation==='/')&&constant<0;
  const target=negative?rational(-rhs.numerator,rhs.denominator):rhs;
  const reverse:Record<Comparison,Comparison>={'<':'>','<=':'>=','>':'<','>=':'<='};
  const comparison=negative?reverse[op]:op;
  const strict=comparison==='>'||comparison==='<=';
  const qualifies=(index:number)=>{
    const x=finiteIeeeValue(kind,index);
    const value=operation==='+'?x+constant:operation==='-'?x-constant:operation==='*'?x*constant:x/constant;
    const y=negative?-value:value;
    if(y===Infinity)return true;if(y===-Infinity)return false;
    const exact=exactNumberRational(y),delta=exact.numerator*target.denominator-target.numerator*exact.denominator;
    return delta>0n||(!strict&&delta===0n);
  };
  let lo=0,hi=last+1;
  while(lo<hi){const mid=Math.floor((lo+hi)/2);if(mid<=last&&qualifies(mid))hi=mid;else lo=mid+1;}
  const high=comparison==='>'||comparison==='>=';
  if(lo===0)return high;if(lo===last+1)return !high;
  return {op:high?'>=':'<',value:exactNumberRational(finiteIeeeValue(kind,lo))};
}
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
/** Invert the actual two binary64 operations of an emitted numerical run.
 * The discrete source format settles ties without real-number division.
 */
export function normalizeFiniteAffineRunComparison(kind:'f16'|'f32',scale:number,offset:number,
  op:Comparison,rhs:Rational):DirectPreimage|boolean{
  if(!Number.isFinite(scale)||!Number.isFinite(offset))throw new Error('Nonfinite affine run');
  if(scale===0){const x=exactNumberRational(offset),delta=x.numerator*rhs.denominator-rhs.numerator*x.denominator;
    return op==='<'?delta<0n:op==='<='?delta<=0n:op==='>'?delta>0n:delta>=0n;}
  const last=2*(kind==='f16'?31743:0x7f7fffff),negative=scale<0;
  const reverse:Record<Comparison,Comparison>={'<':'>','<=':'>=','>':'<','>=':'<='};
  const comparison=negative?reverse[op]:op,target=negative?rational(-rhs.numerator,rhs.denominator):rhs;
  const strict=comparison==='>'||comparison==='<=';
  const qualifies=(index:number)=>{
    let y=scale*finiteIeeeValue(kind,index)+offset;if(negative)y=-y;
    if(y===Infinity)return true;if(y===-Infinity)return false;
    const x=exactNumberRational(y),delta=x.numerator*target.denominator-target.numerator*x.denominator;
    return delta>0n||(!strict&&delta===0n);
  };
  let lo=0,hi=last+1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(mid<=last&&qualifies(mid))hi=mid;else lo=mid+1;}
  const high=comparison==='>'||comparison==='>=';
  if(lo===0)return high;if(lo===last+1)return !high;
  return {op:high?'>=':'<',value:exactNumberRational(finiteIeeeValue(kind,lo))};
}
/** Binary64 reciprocal of a positive finite IEEE producer. Restricting the
 * search to positive values avoids crossing the reciprocal discontinuity.
 */
export function normalizePositiveReciprocalComparison(kind:'f16'|'f32',numerator:number,
  op:Comparison,rhs:Rational):DirectPreimage|boolean{
  if(!(numerator>0&&Number.isFinite(numerator)))throw new Error('Positive reciprocal numerator required');
  const largest=kind==='f16'?31743:0x7f7fffff,first=largest+1,last=2*largest;
  const strict=op==='<'||op==='>=';
  const qualifies=(index:number)=>{
    const y=numerator/finiteIeeeValue(kind,index);
    if(y===Infinity)return false;
    const x=exactNumberRational(y),delta=x.numerator*rhs.denominator-rhs.numerator*x.denominator;
    return delta<0n||(!strict&&delta===0n);
  };
  let lo=first,hi=last+1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(mid<=last&&qualifies(mid))hi=mid;else lo=mid+1;}
  const high=op==='<'||op==='<=';
  if(lo===first)return high;if(lo===last+1)return !high;
  return {op:high?'>=':'<',value:exactNumberRational(finiteIeeeValue(kind,lo))};
}
export function normalizePositiveSqrtComparison(op:Comparison,rhs:Rational):DirectPreimage|boolean{
  if(rhs.numerator<=0n)return op==='>'||op==='>=';
  const boundary=roundedPreimage('f32',op,rhs);
  if(boundary.value.numerator<=0n)return boundary.op==='>'||boundary.op==='>=';
  return {op:boundary.op,value:rational(boundary.value.numerator*boundary.value.numerator,
    boundary.value.denominator*boundary.value.denominator)};
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
