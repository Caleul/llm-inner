import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';

/** Positive finite normal F32 operand, exactly widened to F64. The normalized
 * rational seed and two Newton steps reproduce all 2^24 normalized F32 points
 * bitwise. Exact power-of-two covariance covers every normal F32 exponent.
 * See docs/experiments/direct-json-rational-sqrt-proof.cpp and the certificate
 * test which compiles this actual JSON expression, not a parallel formula.
 * No root, floating conversion, lookup, or compiler reference is emitted. */
export function lowerJsonRationalPositiveNormalSqrtAsF64(x:JsonExpression):JsonExpression {
  if(x[1]!=='f64')throw new TypeError('Exactly widened positive normal F32 input required');
  const u=(n:bigint)=>c('u64',n),raw=o('reinterpret','u64',x);
  const m=o('reinterpret','f64',o('or','u64',
    o('and','u64',raw,u(0xfffffffffffffn)),u(0x3ff0000000000000n)));
  const exponent=o('shr','u64',raw,u(52n));
  // e=E-1023, k=floor(e/2). The biased exponent of 2^k is (E+1023)>>1.
  // E even means e odd; its scale contains the rounded sqrt(2) fraction.
  const scaleBits=o('or','u64',o('shl','u64',
    o('shr','u64',o('add','u64',exponent,u(1023n)),u(1n)),u(52n)),
    o('mul','u64',o('xor','u64',o('and','u64',exponent,u(1n)),u(1n)),u(0x6a09e667f3bcdn)));
  const a:JsonExpression=['constant','f64','0x3ffb8b124936d913'];
  const b:JsonExpression=['constant','f64','0x400d07b36ff85ce5'];
  const d:JsonExpression=['constant','f64','0x401166c4d8faa3bc'];
  let y=o('div','f64',o('add','f64',a,o('mul','f64',b,m)),o('add','f64',d,m));
  for(let step=0;step<2;step++)y=o('mul','f64',c('f64',.5),o('add','f64',y,o('div','f64',m,y)));
  return lowerJsonRoundNormalF32AsF64(o('mul','f64',y,o('reinterpret','f64',scaleBits)));
}
