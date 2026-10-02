import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonWidenNormal} from './direct-json-widen.js';
import {lowerJsonF64ToNormalF32,lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';
import {decodeIeeeF16ToF32} from './utils.js';
import {foldDeclaredCpuF32Exponential} from './direct-rust-numeric.js';

/** Exact expansion of the declared CPU F32 polynomial on [-0.34,0]. The range
 * reduction exponent is identically zero in this domain. The tiny arm rounds
 * to exactly one and avoids classifying subnormal operands. No exp primitive,
 * Euler constant or runtime table remains. */
export function lowerJsonSmallNonpositiveExp(input:JsonExpression):JsonExpression {
  if(input[1]!=='f32')throw new TypeError('F32 exponential operand required');
  const x=lowerJsonWidenNormal(input,'f64');
  let polynomial=c('f64',Math.fround(0.000198527617612853646278381));
  for(const coefficient of [0.00139304355252534151077271,0.00833336077630519866943359,
    0.0416664853692054748535156,0.166666671633720397949219,0.5]){
    const product=o('mul','f64',polynomial,x);
    polynomial=lowerJsonRoundNormalF32AsF64(o('add','f64',product,c('f64',Math.fround(coefficient))));
  }
  const squared=lowerJsonRoundNormalF32AsF64(o('mul','f64',x,x));
  const tail=lowerJsonRoundNormalF32AsF64(o('add','f64',o('mul','f64',squared,polynomial),x));
  const result=lowerJsonF64ToNormalF32(o('add','f64',c('f64',1),tail));
  return o('if','f32',o('lt','bool',input,c('f32',-(2**-25))),result,c('f32',1));
}

/** The caller proves the argument domain. A truncation certificate additionally
 * requires x to be a difference of two half scores within its score bound. */
export function lowerJsonSmallNonpositiveExpAsF64(x:JsonExpression,certificate?:JsonHalfDifferenceExpCertificate):JsonExpression {
  if(x[1]!=='f64')throw new TypeError('Exactly widened F32 input required');
  if(certificate&&!admittedCertificates.has(certificate))throw new Error('Unverified exponential certificate');
  const degree=certificate?.polynomialDegree??7,coefficients=expCoefficients.slice(7-degree);
  let polynomial=c('f64',coefficients[0]!);
  for(const coefficient of coefficients.slice(1))
    polynomial=lowerJsonRoundNormalF32AsF64(o('add','f64',o('mul','f64',polynomial,x),c('f64',coefficient)));
  const squared=lowerJsonRoundNormalF32AsF64(o('mul','f64',x,x));
  const tail=lowerJsonRoundNormalF32AsF64(o('add','f64',o('mul','f64',squared,polynomial),x));
  const result=lowerJsonRoundNormalF32AsF64(o('add','f64',c('f64',1),tail));
  return o('if','f64',o('lt','bool',x,c('f64',-(2**-25))),result,c('f64',1));
}

const expCoefficients=[0.000198527617612853646278381,0.00139304355252534151077271,
  0.00833336077630519866943359,0.0416664853692054748535156,0.166666671633720397949219,0.5].map(Math.fround);
export type JsonExpPolynomialDegree=2|3|4|5|6|7;
/** Compiler diagnostic kernel corresponding to the explicit bit-rounded JSON.
 * Candidate results are never stored as a runtime lookup. */
export function foldJsonExpPolynomial(x:number,degree:JsonExpPolynomialDegree):number {
  if(![2,3,4,5,6,7].includes(degree)||!Number.isFinite(x)||x<-.34||x>0||Math.fround(x)!==x)
    throw new RangeError('Invalid small exponential candidate domain');
  if(x>=-(2**-25))return 1;
  let p=expCoefficients[7-degree]!;
  for(let i=8-degree;i<expCoefficients.length;i++)p=Math.fround(p*x+expCoefficients[i]!);
  return Math.fround(1+Math.fround(Math.fround(x*x)*p+x));
}
export interface JsonHalfDifferenceExpCertificate {
  readonly scoreBound:number;readonly halfScoreValues:number;readonly maxGridIndex:number;
  readonly checkedGridPoints:number;readonly polynomialDegree:JsonExpPolynomialDegree;
  readonly unreachableDifferences:readonly number[];
  readonly policy:'declared-cpu-f32-exp-half-differences';
}
const certificates=new Map<number,JsonHalfDifferenceExpCertificate>();
const admittedCertificates=new WeakSet<object>();
/** Scores and their maximum are finite F16 values within the proved bound.
 * Every value is an integer multiple of 2^-24. Since 2*bound<=.34, differences
 * have <24 significant integer bits and are EXACT F32 values on that lattice.
 * Enumerate the containing lattice; whenever a candidate differs, establish
 * whether ANY admitted pair of half scores can produce that difference.
 * Reject a degree on its first reachable discrepancy, not on a sampled corpus.
 * This certificate removes polynomial operations only, never embeds answers.
 */
export function certifyJsonHalfDifferenceExp(scoreBound:number):JsonHalfDifferenceExpCertificate {
  if(!Number.isFinite(scoreBound)||scoreBound<0||scoreBound>.17)
    throw new RangeError('Half-score exponential certificate requires magnitude <= 0.17');
  const hit=certificates.get(scoreBound);if(hit)return hit;
  const scale=2**24,scores=new Set<number>();let largest=0;
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00){
    const x=decodeIeeeF16ToF32(bits);if(Math.abs(x)>scoreBound)continue;
    const integer=x*scale;scores.add(integer);largest=Math.max(largest,Math.abs(integer));
  }
  const maxGridIndex=2*largest;
  let polynomialDegree:JsonExpPolynomialDegree=7,checkedGridPoints=0,unreachableDifferences:number[]=[];
  for(const degree of [2,3,4,5,6] as const){
    let rejected=false,checked=0;const excluded:number[]=[];
    for(let index=0;index<=maxGridIndex;index++){
      const x=-index/scale;checked++;
      if(foldJsonExpPolynomial(x,degree)===foldDeclaredCpuF32Exponential(x))continue;
      // x=a-b. All values and the lookup key here are exact safe integers.
      let reachable=false;for(const a of scores)if(scores.has(a+index)){reachable=true;break;}
      if(reachable){rejected=true;break;}
      excluded.push(x);
    }
    if(!rejected){polynomialDegree=degree;checkedGridPoints=checked;unreachableDifferences=excluded;break;}
  }
  // The full polynomial is the reference construction, so it remains the
  // fallback when every proposed truncation has a reachable counterexample.
  const certificate:JsonHalfDifferenceExpCertificate=Object.freeze({scoreBound,halfScoreValues:scores.size,
    maxGridIndex,checkedGridPoints,polynomialDegree,unreachableDifferences:Object.freeze(unreachableDifferences),
    policy:'declared-cpu-f32-exp-half-differences' as const});
  admittedCertificates.add(certificate);
  if(certificates.size>=16)certificates.delete(certificates.keys().next().value!);
  certificates.set(scoreBound,certificate);return certificate;
}
