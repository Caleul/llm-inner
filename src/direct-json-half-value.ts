import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';

/** F16 conversion followed by exact F64 widening, for a finite value already
 * on the F32 lattice. Preserve sign, subnormals, ties and overflow explicitly.
 * Combining the boundaries avoids encoding F16 and then decoding it again. */
export function lowerJsonFiniteF16AsF64(input:JsonExpression):JsonExpression {
  if(input[1]!=='f64')throw new TypeError('Exactly widened F32 value required');
  const u=(n:bigint)=>c('u64',n),bits=o('reinterpret','u64',input);
  const magnitudeBits=o('and','u64',bits,u(0x7fffffffffffffffn));
  const magnitude=o('reinterpret','f64',magnitudeBits);
  const odd=o('and','u64',o('shr','u64',magnitudeBits,u(42n)),u(1n));
  const biased=o('add','u64',magnitudeBits,o('add','u64',u(0x1ffffffffffn),odd));
  const normal=o('reinterpret','f64',o('and','u64',biased,u(0xfffffc0000000000n)));
  // Below the smallest F16 normal, a fixed dyadic quantum is sufficient.
  // The addition is an actual F64 rounding boundary; do not cancel the offset.
  const subnormal=o('sub','f64',o('add','f64',magnitude,c('f64',2**28)),c('f64',2**28));
  const positive=o('if','f64',o('lt','bool',magnitude,c('f64',2**-14)),subnormal,
    o('if','f64',o('lt','bool',magnitude,c('f64',65520)),normal,c('f64',Infinity)));
  return o('if','f64',o('lt','bool',bits,u(0x8000000000000000n)),positive,
    o('sub','f64',c('f64',-0),positive));
}
