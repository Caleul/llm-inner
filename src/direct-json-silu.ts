import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {jsonConstant as c,jsonInput,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonWiden} from './direct-json-widen.js';
import {lowerJsonF64ToNormalF32,lowerJsonF32ToF16} from './direct-json-f16.js';
import {lowerJsonFiniteF16AsF64} from './direct-json-half-value.js';
import {evaluateJsonExpression} from './direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from './utils.js';

/** Factorized small-domain polynomial. It is admitted only after checking every
 * finite F16 point in the checkpoint-derived range against the backend profile.
 * No profile, output lookup or SiLU primitive is embedded in the expression. */
export function lowerJsonSmallSilu(input:JsonExpression):JsonExpression {
  if(input[1]!=='f16')throw new TypeError('F16 SiLU operand required');
  const x=lowerJsonWiden(input,'f64'),square=o('mul','f64',x,x);
  let polynomial=c('f64',-17/80640);
  for(const coefficient of [1/480,-1/48,1/4])polynomial=o('add','f64',c('f64',coefficient),o('mul','f64',square,polynomial));
  const result=o('add','f64',o('mul','f64',c('f64',.5),x),o('mul','f64',square,polynomial));
  const rounded=lowerJsonF32ToF16(lowerJsonF64ToNormalF32(result));
  return o('if','f16',o('eq','bool',input,['constant','f16','0x0000']),input,rounded);
}
export function lowerJsonSmallSiluAsF64(x:JsonExpression,degree:2|4=4):JsonExpression {
  if(x[1]!=='f64')throw new TypeError('Exactly widened F16 SiLU operand required');
  if(degree!==2&&degree!==4)throw new RangeError('Unsupported small SiLU candidate degree');
  // This smaller polynomial is a candidate implementation of the primitive,
  // not an unrestricted algebraic rewrite across backend F32 boundaries. Its
  // complete F16 input domain must pass certifyJsonSmallSilu before admission.
  const square=o('mul','f64',x,x);
  const polynomial=degree===2?c('f64',1/4):o('add','f64',c('f64',1/4),o('mul','f64',square,c('f64',-1/48)));
  const result=o('add','f64',o('mul','f64',c('f64',.5),x),o('mul','f64',square,polynomial));
  const bits=o('reinterpret','u64',x),magnitude=o('and','u64',bits,c('u64',0x7fffffffffffffffn));
  // At |x|<=2^-24 the declared F32 sigmoid/product equals x/2; final F16
  // ties-to-even underflows to signed zero. Keeping this cell explicit also
  // preserves -0; direct real-polynomial rounding would differ at +2^-24.
  const signWord=o('and','u64',bits,c('u64',0x8000000000000000n));
  const signedZero=o('reinterpret','f64',signWord);
  return o('if','f64',o('le','bool',magnitude,c('u64',0x3e70000000000000n)),signedZero,
    // Outside the tiny cell, .5*x dominates the nonnegative correction:
    // 0 <= x²*(.25 - x²/48) <= .025*|x| for |x| <= .1.
    // Thus both admitted polynomials retain x's sign. F64 rounding cannot
    // bridge the >=.475*|x| separation from zero. Reuse x's sign word rather
    // than substituting and re-evaluating the polynomial to extract it.
    lowerJsonFiniteF16AsF64(result,{minimum:-.1,maximum:.1},signWord));
}
export interface JsonSiluCertificate {bound:number;checkedPoints:number;maximumMagnitude:number;polynomialDegree:2|4;profileSha256:string;policy:'pytorch-2.12.1-cpu-f16';}
const certificates=new Map<number,JsonSiluCertificate>();
export function certifyJsonSmallSilu(bound:number):JsonSiluCertificate {
  if(!Number.isFinite(bound)||bound<0||bound>.1)throw new RangeError('Small SiLU polynomial requires a proven magnitude <= 0.1');
  const cached=certificates.get(bound);if(cached)return cached;
  const expression=lowerJsonSmallSilu(jsonInput('f16','X1'));
  const composed=lowerJsonSmallSiluAsF64(jsonInput('f64','X1'));
  const quadratic=lowerJsonSmallSiluAsF64(jsonInput('f64','X1'),2);
  const profile=readFileSync(new URL('../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin',import.meta.url));
  if(profile.length!==131072)throw new Error('Invalid backend SiLU certificate profile');
  let checkedPoints=0,maximumMagnitude=0,quadraticExact=true;
  for(let bits=0;bits<65536;bits++){
    if((bits&0x7c00)===0x7c00)continue;
    const value=decodeIeeeF16ToF32(bits);if(Math.abs(value)>bound)continue;
    const expected=decodeIeeeF16ToF32(profile.readUInt16LE(2*bits));
    const actual=evaluateJsonExpression(expression,{X1:value});
    if(!Object.is(actual,expected)||!Object.is(evaluateJsonExpression(composed,{X1:value}),expected))
      throw new Error(`SiLU polynomial certificate failed at F16 bits ${bits.toString(16)}`);
    if(quadraticExact&&!Object.is(evaluateJsonExpression(quadratic,{X1:value}),expected))quadraticExact=false;
    checkedPoints++;maximumMagnitude=Math.max(maximumMagnitude,Math.abs(expected));
  }
  const certificate:JsonSiluCertificate={bound,checkedPoints,maximumMagnitude,polynomialDegree:quadraticExact?2:4,
    profileSha256:createHash('sha256').update(profile).digest('hex'),policy:'pytorch-2.12.1-cpu-f16'};
  if(certificates.size>=16)certificates.delete(certificates.keys().next().value!);
  certificates.set(bound,certificate);return certificate;
}
