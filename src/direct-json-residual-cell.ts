import {decodeIeeeF16ToF32} from './utils.js';
import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';

/** Strict lower bound on every F16 midpoint distance at or above 2^exponent.
 * The F32 addition error is bounded by this distance / 4096. Reserve that
 * margin instead of relying on the tie ownership of a particular half code.
 * The subnormal lattice and its first normal binade both have radius 2^-25. */
export function jsonResidualCellThreshold(maximumCorrection:number):number|undefined {
  if(!Number.isFinite(maximumCorrection)||maximumCorrection<0)return undefined;
  // Corrections are F16, so an interval endpoint between two half codes
  // cannot admit the larger code. Tighten to the largest admitted magnitude
  // before reasoning about the rounding cell; this is discrete-domain proof.
  let low=0,high=0x7bff;
  while(low<high){
    const middle=Math.ceil((low+high)/2);
    if(decodeIeeeF16ToF32(middle)<=maximumCorrection)low=middle;else high=middle-1;
  }
  const admittedMaximum=decodeIeeeF16ToF32(low);
  for(let exponent=-24;exponent<=15;exponent++){
    const radius=Math.max(2**-25,2**(exponent-12));
    if(admittedMaximum<radius*(1-2**-12))return 2**exponent;
  }
  return undefined;
}
/** Compiler proof under finite F16 base/correction contracts. The guard covers
 * only nonzero bases whose entire correction interval remains strictly inside
 * their original rounded cell; the complementary branch retains the full sum.
 * No input-domain restriction or approximate zero is introduced. */
export function lowerJsonStableHalfResidual(base:JsonExpression,correction:JsonExpression,
  sum:JsonExpression,maximumCorrection:number):JsonExpression {
  if(base[1]!=='f16'||correction[1]!=='f16'||sum[1]!=='f16')throw new TypeError('Finite F16 residual operands required');
  const threshold=jsonResidualCellThreshold(maximumCorrection);if(threshold===undefined)return sum;
  const word=c('f64',threshold),magnitude=o('and','u64',o('reinterpret','u64',o('widen','f64',base)),c('u64',0x7fffffffffffffffn));
  const condition=o('le','bool',c('u64',BigInt(word[2] as string)),magnitude);
  return o('if','f16',condition,base,sum);
}
