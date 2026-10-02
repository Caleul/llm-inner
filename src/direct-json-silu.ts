import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {jsonConstant as c,jsonInput,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonWiden} from './direct-json-widen.js';
import {lowerJsonF64ToNormalF32,lowerJsonF32ToF16,lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';
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
export function lowerJsonSmallSiluAsF64(x:JsonExpression):JsonExpression {
  if(x[1]!=='f64')throw new TypeError('Exactly widened F16 SiLU operand required');
  const square=o('mul','f64',x,x);let polynomial=c('f64',-17/80640);
  for(const coefficient of [1/480,-1/48,1/4])polynomial=o('add','f64',c('f64',coefficient),o('mul','f64',square,polynomial));
  const result=o('add','f64',o('mul','f64',c('f64',.5),x),o('mul','f64',square,polynomial));
  return o('if','f64',o('eq','bool',x,c('f64',0)),x,
    lowerJsonFiniteF16AsF64(lowerJsonRoundNormalF32AsF64(result)));
}
export interface JsonSiluCertificate {bound:number;checkedPoints:number;maximumMagnitude:number;profileSha256:string;policy:'pytorch-2.12.1-cpu-f16';}
const certificates=new Map<number,JsonSiluCertificate>();
export function certifyJsonSmallSilu(bound:number):JsonSiluCertificate {
  if(!Number.isFinite(bound)||bound<0||bound>.1)throw new RangeError('Small SiLU polynomial requires a proven magnitude <= 0.1');
  const cached=certificates.get(bound);if(cached)return cached;
  const expression=lowerJsonSmallSilu(jsonInput('f16','X1'));
  const composed=lowerJsonSmallSiluAsF64(jsonInput('f64','X1'));
  const profile=readFileSync(new URL('../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin',import.meta.url));
  if(profile.length!==131072)throw new Error('Invalid backend SiLU certificate profile');
  let checkedPoints=0,maximumMagnitude=0;
  for(let bits=0;bits<65536;bits++){
    if((bits&0x7c00)===0x7c00)continue;
    const value=decodeIeeeF16ToF32(bits);if(Math.abs(value)>bound)continue;
    const expected=decodeIeeeF16ToF32(profile.readUInt16LE(2*bits));
    const actual=evaluateJsonExpression(expression,{X1:value});
    if(!Object.is(actual,expected)||!Object.is(evaluateJsonExpression(composed,{X1:value}),expected))
      throw new Error(`SiLU polynomial certificate failed at F16 bits ${bits.toString(16)}`);
    checkedPoints++;maximumMagnitude=Math.max(maximumMagnitude,Math.abs(expected));
  }
  const certificate:JsonSiluCertificate={bound,checkedPoints,maximumMagnitude,profileSha256:createHash('sha256').update(profile).digest('hex'),policy:'pytorch-2.12.1-cpu-f16'};
  if(certificates.size>=16)certificates.delete(certificates.keys().next().value!);
  certificates.set(bound,certificate);return certificate;
}
