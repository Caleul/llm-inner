import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';

/** Caller certifies x in [2^e,2^(e+1)] for e=-1 or 0. The F64 addition
 * implements every F32 lower/tie/upper outcome at the fixed quantum 2^(e-23).
 * No outcome is dropped; subtraction is exact by Sterbenz. The offset is
 * an even number of quanta, preserving tie ownership. Keep this boundary. */
export function lowerJsonFixedBinadeF32AsF64(x:JsonExpression,e:-1|0):JsonExpression {
  if(x[1]!=='f64'||(e!==-1&&e!==0))throw new TypeError('Certified positive F64 significand required');
  const offset=c('f64',2**(e+29));
  return o('sub','f64',o('add','f64',x,offset),offset);
}

/** Positive finite normal F32 operand, exactly widened to F64. The five-term
 * rational partial fraction evaluates the normalized root using five source
 * occurrences instead of recursively substituting Newton refinements. Its
 * rounded result is exhaustively checked for all 2^24 exponent-parity points
 * by compiling this actual JSON, including the composed reciprocal boundary.
 * Binary scaling then covers every normal F32 exponent. No root, conversion,
 * table, intermediate binding or reference executor is emitted. */
function normalizedRootParts(x:JsonExpression):{rounded:JsonExpression;powerExponent:JsonExpression} {
  if(x[1]!=='f64')throw new TypeError('Exactly widened positive normal F32 input required');
  const u=(n:bigint)=>c('u64',n),raw=o('reinterpret','u64',x);
  const m=o('reinterpret','f64',o('or','u64',
    o('and','u64',raw,u(0xfffffffffffffn)),u(0x3ff0000000000000n)));
  const exponent=o('shr','u64',raw,u(52n));
  // e=E-1023, k=floor(e/2). The biased exponent of 2^k is (E+1023)>>1.
  // E even means e odd; its scale contains the rounded sqrt(2) fraction.
  const parityFraction=o('mul','u64',o('xor','u64',o('and','u64',exponent,u(1n)),u(1n)),u(0x6a09e667f3bcdn));
  const normalizedScale=o('reinterpret','f64',o('or','u64',u(0x3ff0000000000000n),parityFraction));
  const powerExponent=o('shr','u64',o('add','u64',exponent,u(1023n)),u(1n));
  // Centering is exact by Sterbenz for m in [1,2]. Keep this particular
  // summation order: real-number equivalence alone does not certify F64
  // arithmetic close to a final F32 midpoint.
  const z=o('sub','f64',m,c('f64',1.5));
  let y:JsonExpression=c('f64',13.367174015436346);
  for(const [residue,pole] of [
    [-779.0174377628756,72.88886838020784],
    [-9.032202101157678,8.541941977833192],
    [-1.0005960489978687,3.4400824392419294],
    [-0.1852754751106736,2.096193424586035],
    [-0.029292444172441905,1.6235843576093156],
  ])y=o('add','f64',y,o('div','f64',c('f64',residue!),o('add','f64',z,c('f64',pole!))));
  // The F64 quantum at 2^29 is the final significand's F32 quantum.
  // Preserve this addition/subtraction boundary and the following exact
  // power-of-two scaling. The root is rounded before reciprocal division.
  const significand=o('mul','f64',y,normalizedScale);
  const rounded=lowerJsonFixedBinadeF32AsF64(significand,0);
  return {rounded,powerExponent};
}
export function lowerJsonRationalPositiveNormalSqrtAsF64(x:JsonExpression):JsonExpression {
  const {rounded,powerExponent}=normalizedRootParts(x);
  return o('mul','f64',rounded,o('reinterpret','f64',o('shl','u64',powerExponent,c('u64',52n))));
}

/** Exact composition R32(1 / R32(sqrt(x))). Both boundaries remain explicit
 * F64 quantum additions. Round the reciprocal significand in [0.5,1] at
 * quantum 2^-24, then scale by the reciprocal power of two. This commutes
 * with normal F32 rounding throughout the positive normal F32 input domain.
 * It is not a replacement by an independently rounded reciprocal sqrt. */
export function lowerJsonRationalPositiveNormalInverseSqrtAsF64(x:JsonExpression):JsonExpression {
  const {rounded,powerExponent}=normalizedRootParts(x);
  const reciprocal=o('div','f64',c('f64',1),rounded);
  const result=lowerJsonFixedBinadeF32AsF64(reciprocal,-1);
  const scale=o('reinterpret','f64',o('shl','u64',o('sub','u64',c('u64',2046n),powerExponent),c('u64',52n)));
  return o('mul','f64',result,scale);
}
