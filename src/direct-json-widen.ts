import {jsonConstant as c,jsonOperation as o,jsonWidths,type JsonDtype,type JsonExpression} from './direct-json-expression.js';
/** Lossless floating widening lowered to explicit bit operations. Temporary
 * widening nodes never need to survive in the emitted expression. */
export function lowerJsonWiden(input:JsonExpression,target:'f32'|'f64'):JsonExpression {
  const source=input[1];
  if(source!=='f16'&&source!=='f32'||jsonWidths[target]<=jsonWidths[source])throw new TypeError('Unsupported exact widening');
  const integer=target==='f64'?'u64':'u32';
  const sourceInteger:JsonDtype=source==='f16'?'u16':'u32';
  const sourceFraction=source==='f16'?10:23,sourceBias=source==='f16'?15:127;
  const sourceMaxExponent=source==='f16'?31:255;
  const targetFraction=target==='f64'?52:23,targetBias=target==='f64'?1023:127;
  const u=(value:bigint|number)=>c(integer,value);
  const b=(op:'add'|'sub'|'and'|'or'|'shl'|'shr',a:JsonExpression,z:JsonExpression)=>o(op,integer,a,z);
  const eq=(a:JsonExpression,z:JsonExpression)=>o('eq','bool',a,z);
  const choose=(condition:JsonExpression,yes:JsonExpression,no:JsonExpression)=>o('if',integer,condition,yes,no);
  const raw=o('convert',integer,o('reinterpret',sourceInteger,input));
  const fractionMask=(1n<<BigInt(sourceFraction))-1n;
  const fraction=b('and',raw,u(fractionMask)),exponent=b('and',b('shr',raw,u(sourceFraction)),u(sourceMaxExponent));
  const sign=b('shl',b('and',raw,u(1n<<BigInt(jsonWidths[source]-1))),u(jsonWidths[target]-jsonWidths[source]));
  const biasDifference=targetBias-sourceBias;
  let normalized=fraction,adjustedExponent=u(biasDifference+1);
  const steps=source==='f16'?[8,4,2,1]:[16,8,4,2,1];
  for(const shift of steps){
    const test=o('lt','bool',normalized,u(1n<<BigInt(sourceFraction+1-shift)));
    normalized=choose(test,b('shl',normalized,u(shift)),normalized);
    adjustedExponent=choose(test,b('sub',adjustedExponent,u(shift)),adjustedExponent);
  }
  const subnormal=b('or',b('shl',adjustedExponent,u(targetFraction)),
    b('shl',b('and',normalized,u(fractionMask)),u(targetFraction-sourceFraction)));
  const normal=b('or',b('shl',b('add',exponent,u(biasDifference)),u(targetFraction)),
    b('shl',fraction,u(targetFraction-sourceFraction)));
  const special=b('or',u(((1n<<BigInt(target==='f64'?11:8))-1n)<<BigInt(targetFraction)),
    b('shl',fraction,u(targetFraction-sourceFraction)));
  const value=choose(eq(exponent,u(0)),choose(eq(fraction,u(0)),u(0),subnormal),
    choose(eq(exponent,u(sourceMaxExponent)),special,normal));
  return o('reinterpret',target,b('or',sign,value));
}

/** Requires a compiler proof that the source is finite and normal (sign may
 * vary). Avoids re-classifying zeros/subnormals after every already-proved step. */
export function lowerJsonWidenNormal(input:JsonExpression,target:'f32'|'f64'):JsonExpression {
  const source=input[1];
  if(source!=='f16'&&source!=='f32'||jsonWidths[target]<=jsonWidths[source])throw new TypeError('Unsupported normal widening');
  const integer=target==='f64'?'u64':'u32',sourceInteger=source==='f16'?'u16':'u32';
  const sourceFraction=source==='f16'?10:23,targetFraction=target==='f64'?52:23;
  const sourceBias=source==='f16'?15:127,targetBias=target==='f64'?1023:127;
  const u=(value:number|bigint)=>c(integer,value),b=(op:'add'|'and'|'or'|'shl'|'shr',a:JsonExpression,z:JsonExpression)=>o(op,integer,a,z);
  const bits=o('convert',integer,o('reinterpret',sourceInteger,input));
  const sign=b('shl',b('and',bits,u(1n<<BigInt(jsonWidths[source]-1))),u(jsonWidths[target]-jsonWidths[source]));
  const magnitude=b('and',bits,u((1n<<BigInt(jsonWidths[source]-1))-1n));
  const fraction=b('and',bits,u((1n<<BigInt(sourceFraction))-1n));
  const exponent=b('add',b('shr',magnitude,u(sourceFraction)),u(targetBias-sourceBias));
  return o('reinterpret',target,b('or',sign,b('or',b('shl',exponent,u(targetFraction)),
    b('shl',fraction,u(targetFraction-sourceFraction)))));
}
