/**
 * Emit binary rounding in place. The generated bodies use comparisons, addition,
 * subtraction and multiplication/division by literal powers of two only. No
 * numeric helper or lookup survives in the emitted expression.
 *
 * Inputs are finite JS numbers. Overflow, signed zero and midpoint ties follow
 * IEEE round-to-nearest-even. The F16 body returns the representation bits;
 * the F32 body returns the rounded numeric value.
 */
export function directRoundPrefix(kind: "f16Bits" | "Math.fround"): string {
  const half = kind === "f16Bits";
  const minimumNormal = half ? "0.00006103515625" : "1.1754943508222875e-38";
  const minimumStep = half ? "5.960464477539063e-8" : "1.401298464324817e-45";
  const largest = half ? "65504" : "3.4028234663852886e38";
  const mantissaSteps = half ? 10 : 23;
  const initialStep = half ? 512 : 4194304;
  const overflow = half ? "(negative?64512:31744)" : "(negative?-Infinity:Infinity)";
  const zero = half ? "(negative?32768:0)" : "(negative?-0:0)";
  const finish = half
    ? "return (negative?32768:0)+(normal?exponent*1024:0)+mantissa;"
    : "return negative?-lower:lower;";
  return `((input)=>{` +
    `if(input!==input||input>1.7976931348623157e308||input< -1.7976931348623157e308)throw new RangeError('Entrada não finita');` +
    (half ? `input=${directRoundPrefix("Math.fround")}input);` : "") +
    `if(input!==input||input>1.7976931348623157e308||input< -1.7976931348623157e308)throw new RangeError('Entrada não finita');` +
    `const negative=input<0||(input===0&&1/input<0);const magnitude=negative?-input:input;` +
    `if(magnitude===0)return ${zero};` +
    `let base=${minimumNormal},unit=${minimumStep},exponent=1,normal=magnitude>=base;` +
    `if(normal){while(magnitude>=base+base&&exponent<${half ? 30 : 254}){base+=base;unit+=unit;exponent++;}}` +
    `else{base=0;exponent=0;}` +
    `if(magnitude>=${largest}+unit/2)return ${overflow};` +
    `let lower=base,mantissa=0,step=unit*${initialStep},indexStep=${initialStep};` +
    `for(let bit=${mantissaSteps - 1};bit>=0;bit--){const candidate=lower+step;` +
    `if(magnitude>=candidate){lower=candidate;mantissa+=indexStep;}step/=2;indexStep/=2;}` +
    `const midpoint=lower+unit/2;` +
    `if(magnitude>midpoint||(magnitude===midpoint&&mantissa%2!==0)){` +
    `lower+=unit;mantissa++;if(mantissa===${half ? 1024 : 8388608}){mantissa=0;` +
    `if(normal)exponent++;else{normal=true;exponent=1;}}}` +
    `${finish}})(`;
}

/** Small-source diagnostic used to compare in-place rounding within an existing forward. */
export function lowerDirectRoundingInSource(source: string): string {
  return source.replace(/\b(f16Bits|Math\.fround)\(/g, (_match, kind: "f16Bits" | "Math.fround") =>
    directRoundPrefix(kind));
}
