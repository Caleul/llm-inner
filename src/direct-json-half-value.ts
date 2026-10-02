import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import type {JsonFloatRange} from './direct-json-range.js';

/** Direct finite F64 -> F16 conversion followed by exact F64 widening.
 * Exactly widened F32 values are a subset of this domain. Preserve sign,
 * subnormals, ties and overflow explicitly.
 * Combining the boundaries avoids encoding F16 and then decoding it again. */
export function lowerJsonFiniteF16AsF64(input:JsonExpression,range?:JsonFloatRange):JsonExpression {
  if(input[1]!=='f64')throw new TypeError('Finite F64 source required');
  if(range&&(!Number.isFinite(range.minimum)||!Number.isFinite(range.maximum)||range.minimum>range.maximum))
    throw new RangeError('Invalid certified F16 source interval');
  const u=(n:bigint)=>c('u64',n),bits=o('reinterpret','u64',input);
  const magnitudeBits=o('and','u64',bits,u(0x7fffffffffffffffn));
  const magnitude=o('reinterpret','f64',magnitudeBits);
  const odd=o('and','u64',o('shr','u64',magnitudeBits,u(42n)),u(1n));
  const biased=o('add','u64',magnitudeBits,o('add','u64',u(0x1ffffffffffn),odd));
  const normal=o('reinterpret','f64',o('and','u64',biased,u(0xfffffc0000000000n)));
  // Below the smallest F16 normal, a fixed dyadic quantum is sufficient.
  // The addition is an actual F64 rounding boundary; do not cancel the offset.
  const subnormal=o('sub','f64',o('add','f64',magnitude,c('f64',2**28)),c('f64',2**28));
  const maximum=range?Math.max(Math.abs(range.minimum),Math.abs(range.maximum)):Infinity;
  const minimum=range&&range.minimum>0?range.minimum:range&&range.maximum<0?-range.maximum:0;
  const aboveSubnormal=maximum<65520?normal:
    o('if','f64',o('lt','bool',magnitude,c('f64',65520)),normal,c('f64',Infinity));
  const positive=maximum<2**-14?subnormal:minimum>=2**-14?aboveSubnormal:
    o('if','f64',o('lt','bool',magnitude,c('f64',2**-14)),subnormal,aboveSubnormal);
  // Strict sign bounds preserve negative zero; a bound containing zero cannot
  // remove the sign decision merely because -0 compares equal to +0.
  if(range&&range.minimum>0)return positive;
  if(range&&range.maximum<0)return o('sub','f64',c('f64',-0),positive);
  // Sign restoration is an exact bit operation, not a decision. The positive
  // magnitude path yields +0, a positive finite value, or +infinity; OR-ing
  // the original sign preserves even underflow to negative zero.
  return o('reinterpret','f64',o('or','u64',o('reinterpret','u64',positive),
    o('and','u64',bits,u(0x8000000000000000n))));
}

/** Fuse F64->F32->F16 with exact double-rounding cells. Requires a proof that
 * the final F16 value is finite and normal. It is NOT direct F64->F16 rounding:
 * the first rounding widens each final tie cell by half an F32 quantum.
 * A target-even cell rounds up strictly above half+2^28; target-odd rounds up
 * at half-2^28. Bias + parity*(2^29+1) encodes both without a decision. */
export function lowerJsonNormalF32ThenF16AsF64(raw:JsonExpression):JsonExpression {
  if(raw[1]!=='f64')throw new TypeError('F64 unrounded source required');
  const u=(n:bigint)=>c('u64',n),bits=o('reinterpret','u64',raw);
  const odd=o('and','u64',o('shr','u64',bits,u(42n)),u(1n));
  const bias=o('add','u64',u((1n<<41n)-(1n<<28n)-1n),o('mul','u64',odd,u((1n<<29n)+1n)));
  return o('reinterpret','f64',o('and','u64',o('add','u64',bits,bias),u(0xfffffc0000000000n)));
}
