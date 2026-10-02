import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonWidenNormal} from './direct-json-widen.js';
import {lowerJsonF64ToNormalF32,lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';

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

export function lowerJsonSmallNonpositiveExpAsF64(x:JsonExpression):JsonExpression {
  if(x[1]!=='f64')throw new TypeError('Exactly widened F32 input required');
  let polynomial=c('f64',Math.fround(0.000198527617612853646278381));
  for(const coefficient of [0.00139304355252534151077271,0.00833336077630519866943359,
    0.0416664853692054748535156,0.166666671633720397949219,0.5])
    polynomial=lowerJsonRoundNormalF32AsF64(o('add','f64',o('mul','f64',polynomial,x),c('f64',Math.fround(coefficient))));
  const squared=lowerJsonRoundNormalF32AsF64(o('mul','f64',x,x));
  const tail=lowerJsonRoundNormalF32AsF64(o('add','f64',o('mul','f64',squared,polynomial),x));
  const result=lowerJsonRoundNormalF32AsF64(o('add','f64',c('f64',1),tail));
  return o('if','f64',o('lt','bool',x,c('f64',-(2**-25))),result,c('f64',1));
}
