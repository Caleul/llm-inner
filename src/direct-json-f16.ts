import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
/** Exact IEEE finite F32 -> F16 conversion using explicit unsigned operations.
 * No exponent-band enumeration, rounding-mode node, runtime table or helper is
 * present in the returned JSON. Branches classify magnitude and resolve ties. */
export function lowerJsonF32ToF16(input:JsonExpression):JsonExpression {
  if(input[1]!=='f32')throw new TypeError('Exactly represented F32 operand required');
  const u=(n:number)=>c('u32',n),b=(op:'add'|'sub'|'and'|'or'|'shl'|'shr',a:JsonExpression,z:JsonExpression)=>o(op,'u32',a,z);
  const test=(op:'lt'|'le'|'eq',a:JsonExpression,z:JsonExpression)=>o(op,'bool',a,z);
  const choose=(cond:JsonExpression,yes:JsonExpression,no:JsonExpression)=>o('if','u32',cond,yes,no);
  const bits=o('reinterpret','u32',input),magnitude=b('and',bits,u(0x7fffffff));
  const sign=b('and',b('shr',bits,u(16)),u(0x8000));
  const exponent=b('shr',magnitude,u(23)),fraction=b('and',bits,u(0x7fffff));
  function rounded(mantissa:JsonExpression,shift:JsonExpression):JsonExpression {
    const quotient=b('shr',mantissa,shift);
    const half=b('shl',u(1),b('sub',shift,u(1)));
    // Bias by half-1 plus retained low bit: ties choose an even quotient.
    // Mantissas are at most 24 bits; this addition cannot wrap u32.
    return b('shr',b('add',mantissa,b('add',b('sub',half,u(1)),b('and',quotient,u(1)))),shift);
  }
  const normal=b('add',b('shl',b('sub',exponent,u(112)),u(10)),rounded(fraction,u(13)));
  const subnormal=rounded(b('or',fraction,u(0x800000)),b('sub',u(126),exponent));
  // The subnormal arm has exponent 102..112: all shifts are within 14..24.
  const finite=choose(test('lt',exponent,u(102)),u(0),choose(test('lt',exponent,u(113)),subnormal,
    choose(test('lt',exponent,u(143)),normal,u(0x7c00))));
  // Explicit NaN policy is PyTorch-style quiet NaN with payload high bits.
  // Model admission separately requires finite intermediates; NaNs are not
  // counted as proven backend parity merely because this conversion exists.
  const nonfinite=choose(test('eq',fraction,u(0)),u(0x7c00),b('or',u(0x7e00),b('shr',fraction,u(13))));
  const halfBits=o('convert','u16',b('or',sign,choose(test('eq',exponent,u(255)),nonfinite,finite)));
  return o('reinterpret','f16',halfBits);
}

/** Exact F64 -> F32 conversion. The returned syntax contains only integer
 * bit operations, arithmetic, comparisons, conditionals and bit reinterpretation. */
export function lowerJsonF64ToF32(input:JsonExpression):JsonExpression {
  if(input[1]!=='f64')throw new TypeError('F64 operand required');
  const u=(n:number|bigint)=>c('u64',n),b=(op:'add'|'sub'|'and'|'or'|'shl'|'shr',a:JsonExpression,z:JsonExpression)=>o(op,'u64',a,z);
  const test=(op:'lt'|'eq',a:JsonExpression,z:JsonExpression)=>o(op,'bool',a,z);
  const choose=(cond:JsonExpression,yes:JsonExpression,no:JsonExpression)=>o('if','u64',cond,yes,no);
  const bits=o('reinterpret','u64',input),magnitude=b('and',bits,u(0x7fffffffffffffffn));
  const sign=b('and',b('shr',bits,u(32)),u(0x80000000));
  const exponent=b('shr',magnitude,u(52)),fraction=b('and',bits,u(0xfffffffffffffn));
  function rounded(mantissa:JsonExpression,shift:JsonExpression):JsonExpression {
    const quotient=b('shr',mantissa,shift),half=b('shl',u(1),b('sub',shift,u(1)));
    // Mantissas are at most 53 bits, leaving room for this exact u64 bias.
    return b('shr',b('add',mantissa,b('add',b('sub',half,u(1)),b('and',quotient,u(1)))),shift);
  }
  const normal=b('add',b('shl',b('sub',exponent,u(896)),u(23)),rounded(fraction,u(29)));
  const subnormal=rounded(b('or',fraction,u(0x10000000000000n)),b('sub',u(926),exponent));
  const finite=choose(test('lt',exponent,u(873)),u(0),choose(test('lt',exponent,u(897)),subnormal,
    choose(test('lt',exponent,u(1151)),normal,u(0x7f800000))));
  const nonfinite=choose(test('eq',fraction,u(0)),u(0x7f800000),b('or',u(0x7fc00000),b('shr',fraction,u(29))));
  const result=o('convert','u32',b('or',sign,choose(test('eq',exponent,u(2047)),nonfinite,finite)));
  return o('reinterpret','f32',result);
}

/** Requires a proof that the rounded result is finite and normal F32, for
 * either sign. This is the same conversion with unreachable classifications
 * eliminated before condition-product expansion. */
export function lowerJsonF64ToNormalF32(input:JsonExpression):JsonExpression {
  if(input[1]!=='f64')throw new TypeError('F64 normal-conversion operand required');
  const u=(n:number|bigint)=>c('u64',n),b=(op:'add'|'sub'|'and'|'or'|'shl'|'shr',a:JsonExpression,z:JsonExpression)=>o(op,'u64',a,z);
  const bits=o('reinterpret','u64',input),magnitude=b('and',bits,u(0x7fffffffffffffffn));
  const fraction=b('and',bits,u(0xfffffffffffffn)),quotient=b('shr',fraction,u(29));
  const rounded=b('shr',b('add',fraction,b('add',u(0xfffffff),b('and',quotient,u(1)))),u(29));
  const exponent=b('sub',b('shr',magnitude,u(52)),u(896));
  const sign=b('and',b('shr',bits,u(32)),u(0x80000000));
  const result=b('or',sign,b('add',b('shl',exponent,u(23)),rounded));
  return o('reinterpret','f32',o('convert','u32',result));
}

/** Round to a proved finite normal F32 value (or signed zero) and keep its exact F64 widening.
 * Combining these boundaries eliminates exponent rebasing and repeated bit
 * reconstruction. The dropped 29 bits are rounded and masked in one word. */
export function lowerJsonRoundNormalF32AsF64(input:JsonExpression):JsonExpression {
  if(input[1]!=='f64')throw new TypeError('F64 normal-rounding operand required');
  const u=(n:bigint)=>c('u64',n),bits=o('reinterpret','u64',input);
  const odd=o('and','u64',o('shr','u64',bits,u(29n)),u(1n));
  const biased=o('add','u64',bits,o('add','u64',u(0xfffffffn),odd));
  return o('reinterpret','f64',o('and','u64',biased,u(0xffffffffe0000000n)));
}
