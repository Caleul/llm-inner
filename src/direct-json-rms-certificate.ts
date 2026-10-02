import {decodeIeeeF16ToF32} from './utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from './fixed-f16-projection.js';
export interface JsonRmsCriticalCase {readonly inputBits:readonly number[];readonly referenceBits:readonly number[];readonly candidateBits:readonly number[]}
export interface JsonRmsRootCertificate {
  readonly width:number;readonly epsilon:number;readonly steps:3|4;readonly complete:boolean;readonly reason:string;
  readonly normalizedMantissas:number;readonly rootFailures:number;readonly variances:number;readonly sumCells:number;
  readonly cases:readonly JsonRmsCriticalCase[];
}
const word=new DataView(new ArrayBuffer(4));
const bits=(x:number)=>{word.setFloat32(0,x);return word.getUint32(0);};
const value=(b:number)=>{word.setUint32(0,b);return word.getFloat32(0);};
/** Numeric diagnostic kernel; only a certified RMS consumer may use this root. */
export function jsonThreeNewtonRoot(x:number):number {
  let y=value((bits(x)>>>1)+0x1fc00000);
  for(let i=0;i<3;i++)y=.5*(y+x/y);
  return Math.fround(y);
}
let failures:number[]|undefined;
function normalizedFailures():number[]{
  if(failures)return failures;failures=[];
  for(let b=0x3f800000;b<0x40800000;b++){
    const x=value(b),candidate=jsonThreeNewtonRoot(x),reference=Math.fround(Math.sqrt(x));
    if(candidate!==reference){
      if(bits(candidate)!==bits(reference)+1)throw new Error('Unexpected three-Newton error cell');
      failures.push(b);
    }
  }
  return failures;
}
// Exact dyadics in units 2^-150. Even the smallest F32 midpoint is integral.
const scaled=(b:number)=>{const e=b>>>23,m=b&0x7fffff;return e?BigInt(m+0x800000)<<BigInt(e):BigInt(m)<<1n;};
const admitted=new WeakSet<object>(),cache=new Map<string,JsonRmsRootCertificate>();
export function jsonCertifiedRmsRootSteps(c:JsonRmsRootCertificate):3|4 {
  if(!admitted.has(c))throw new Error('Unverified RMS root consumer certificate');return c.steps;
}
/** For width two, enumerate all root-error cells over every normal exponent.
 * Dyadic midpoint preimages invert R32(R32(a²+b²)/2 + epsilon) exactly.
 * Two monotone pointers then enumerate EVERY unordered finite-half magnitude
 * pair in those preimages. Both first-half-normalized coordinates must match.
 * Signs are symmetric, including signed zero. Other variances have identical
 * roots (or reciprocals), so their consumer is unchanged. Exact binary scaling
 * extends the exhaustive [1,4) mantissa check to all normal F32 exponents.
 * Budget exhaustion or any counterexample keeps four steps; neither is a proof
 * that further simplification is impossible. These cases are diagnostic only.
 */
export function certifyJsonRmsRoot(width:number,epsilon:number,
  limits:{maxSumCells?:number;maxPairs?:number}={}):JsonRmsRootCertificate {
  const eps=Math.fround(epsilon),maxSumCells=limits.maxSumCells??10000,maxPairs=limits.maxPairs??100000;
  if(!Number.isSafeInteger(width)||width<1||!Number.isFinite(eps)||eps<2**-126||
    [maxSumCells,maxPairs].some(x=>!Number.isSafeInteger(x)||x<1))throw new RangeError('Invalid RMS consumer domain or budget');
  const key=`${width}:${bits(eps)}:${maxSumCells}:${maxPairs}`,hit=cache.get(key);if(hit)return hit;
  let steps:3|4=4,complete=false,reason='width-not-supported',variances=0,sumCells=0;
  const cases:JsonRmsCriticalCase[]=[];
  function finish():JsonRmsRootCertificate {
    const c=Object.freeze({width,epsilon:eps,steps,complete,reason,variances,sumCells,
      normalizedMantissas:width===2?16777216:0,rootFailures:width===2?failures!.length:0,cases:Object.freeze(cases)});
    admitted.add(c);if(cache.size>=8)cache.delete(cache.keys().next().value!);cache.set(key,c);return c;
  }
  if(width!==2)return finish();
  const profile=normalizedFailures(),values:number[]=[],squares:number[]=[];
  for(let b=0;b<=0x7bff;b++){const x=decodeIeeeF16ToF32(b);values.push(x);squares.push(x*x);}
  const n=values.length,minSumBits=bits(squares[1]!),maxSumBits=bits(2*squares[n-1]!),maxVariance=Math.fround(squares[n-1]!+eps);
  const half=(x:number)=>roundDyadicToF16IfElse(f32BitsToDyadic(bits(x)));
  function firstIndex(predicate:(b:number)=>boolean):number {
    let low=minSumBits,high=maxSumBits+1;
    while(low<high){const mid=Math.floor((low+high)/2);if(predicate(mid))high=mid;else low=mid+1;}return low;
  }
  for(const f of profile)for(let e=1;e<255;e++){
    if((e&1)!==((f>>>23)&1))continue;
    const vb=(e<<23)|(f&0x7fffff),variance=value(vb);
    if(variance<eps||variance>maxVariance)continue;variances++;
    const candidateInverse=Math.fround(1/jsonThreeNewtonRoot(variance)),referenceInverse=Math.fround(1/Math.fround(Math.sqrt(variance)));
    if(candidateInverse===referenceInverse)continue;
    const lo=scaled(vb)+scaled(vb-1)-2n*scaled(bits(eps)),hi=scaled(vb)+scaled(vb+1)-2n*scaled(bits(eps));
    const inclusive=(vb&1)===0,first=firstIndex(b=>inclusive?scaled(b)>=lo:scaled(b)>lo),end=firstIndex(b=>inclusive?scaled(b)>hi:scaled(b)>=hi);
    const cells:number[]=[];
    if(inclusive?0n>=lo&&0n<=hi:0n>lo&&0n<hi)cells.push(0);
    if(sumCells+end-first+cells.length>maxSumCells){reason='sum-cell-budget';return finish();}
    for(let b=first;b<end;b++)cells.push(b);
    for(const sb of cells){
      const sum=value(sb);sumCells++;
      if(Math.fround(sum*.5+eps)!==variance)throw new Error('RMS midpoint preimage inversion failed');
      let lower=n-1,upper=n-1;
      for(let a=0;a<n;a++){
        while(lower>=0&&Math.fround(squares[a]!+squares[lower]!)>=sum)lower--;
        while(upper>=0&&Math.fround(squares[a]!+squares[upper]!)>sum)upper--;
        if(upper<a)break;
        for(let b=Math.max(a,lower+1);b<=upper;b++){
          if(cases.length>=maxPairs){reason='pair-budget';return finish();}
          const candidateBits=[half(Math.fround(values[a]!*candidateInverse)),half(Math.fround(values[b]!*candidateInverse))];
          const referenceBits=[half(Math.fround(values[a]!*referenceInverse)),half(Math.fround(values[b]!*referenceInverse))];
          cases.push(Object.freeze({inputBits:Object.freeze([a,b]),candidateBits:Object.freeze(candidateBits),referenceBits:Object.freeze(referenceBits)}));
          if(candidateBits[0]!==referenceBits[0]||candidateBits[1]!==referenceBits[1]){reason='consumer-counterexample';return finish();}
        }
      }
    }
  }
  steps=3;complete=true;reason='all-critical-consumers-equal';return finish();
}
