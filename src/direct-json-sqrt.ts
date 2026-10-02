import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonWidenNormal} from './direct-json-widen.js';
import {lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';
import {jsonCertifiedRmsRootSteps,type JsonRmsRootCertificate} from './direct-json-rms-certificate.js';

/** Requires positive, finite, normal F32 input. Root and Newton iterates are
 * finite normal numbers. A bit seed has relative error <6.1%; three F64 Newton
 * steps put the candidate within one F32 value. Exact F64 midpoint squares
 * select the correct adjacent F32 value, without exp/sqrt/round nodes.
 * A positive F32 input cannot equal these odd 49/50-bit midpoint squares. */
export function lowerJsonPositiveNormalSqrt(input:JsonExpression):JsonExpression {
  if(input[1]!=='f32')throw new TypeError('F32 square-root operand required');
  const u=(n:number|bigint)=>c('u64',n),v=(n:number)=>c('u32',n);
  const raw=o('reinterpret','u32',input);
  const seed=o('reinterpret','f32',o('add','u32',o('shr','u32',raw,v(1)),v(0x1fc00000)));
  const x=lowerJsonWidenNormal(input,'f64');let y=lowerJsonWidenNormal(seed,'f64');
  for(let i=0;i<3;i++)y=o('mul','f64',c('f64',0.5),o('add','f64',y,o('div','f64',x,y)));
  // Candidate is in [2^-63,2^64], so conversion cannot under/overflow and needs
  // no magnitude branches. Retained mantissa low bit handles ties branchlessly.
  const bits=o('reinterpret','u64',y);
  const fraction=o('and','u64',bits,u(0xfffffffffffffn));
  const exponent=o('sub','u64',o('shr','u64',bits,u(52)),u(896));
  const quotient=o('shr','u64',fraction,u(29));
  const rounded=o('shr','u64',o('add','u64',fraction,o('add','u64',u(0xfffffff),o('and','u64',quotient,u(1)))),u(29));
  const candidateBits=o('convert','u32',o('add','u64',o('shl','u64',exponent,u(23)),rounded));
  const candidate=o('reinterpret','f32',candidateBits);
  const lower=o('reinterpret','f32',o('sub','u32',candidateBits,v(1)));
  const upper=o('reinterpret','f32',o('add','u32',candidateBits,v(1)));
  const value=lowerJsonWidenNormal(candidate,'f64');
  const midpoint=(other:JsonExpression)=>o('mul','f64',c('f64',0.5),o('add','f64',value,lowerJsonWidenNormal(other,'f64')));
  const lowMid=midpoint(lower),highMid=midpoint(upper);
  return o('if','f32',o('lt','bool',x,o('mul','f64',lowMid,lowMid)),lower,
    o('if','f32',o('lt','bool',o('mul','f64',highMid,highMid),x),upper,candidate));
}

/** Same certified root, keeping the F32 result exactly widened throughout.
 * This composes directly with surrounding expressions instead of repeatedly
 * encoding and decoding candidate/neighbor F32 values. */
export function lowerJsonPositiveNormalSqrtAsF64(x:JsonExpression):JsonExpression {
  return lowerNewtonRoot(x,4);
}
/** This intermediate may differ in F32, but the certified first-half RMS
 * consumer is identical. Do not use it as a standalone square-root result. */
export function lowerJsonCertifiedRmsRootAsF64(x:JsonExpression,certificate:JsonRmsRootCertificate):JsonExpression {
  return lowerNewtonRoot(x,jsonCertifiedRmsRootSteps(certificate));
}
function lowerNewtonRoot(x:JsonExpression,steps:3|4):JsonExpression {
  if(x[1]!=='f64')throw new TypeError('Exactly widened positive normal F32 input required');
  const u=(n:bigint)=>c('u64',n),raw=o('reinterpret','u64',x);
  // Compose F64->F32 encoding, the bit seed, and exact widening algebraically.
  // Only one occurrence of x is needed to construct the seed.
  const halfBits=o('shr','u64',o('sub','u64',o('shr','u64',raw,u(29n)),u(0x1c0000000n)),u(1n));
  let y=o('reinterpret','f64',o('shl','u64',o('add','u64',halfBits,u(0x1dfc00000n)),u(29n)));
  for(let i=0;i<steps;i++)y=o('mul','f64',c('f64',.5),o('add','f64',y,o('div','f64',x,y)));
  // After three steps |relative error|<1.2e-12. The fourth has error <1.51u,
  // u=2^-53. An F32 input is at least 2^-51 in relative root distance from
  // any F32 rounding midpoint (odd midpoint square). Therefore no correction
  // branch is required with four steps; the two roundings cannot cross a midpoint.
  // Three steps require the separate proof of the complete RMS F16 consumer.
  return lowerJsonRoundNormalF32AsF64(y);
}
