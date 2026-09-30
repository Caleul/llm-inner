/**
 * Inline the pinned CPU F32 exponential used by attention softmax. Its
 * coefficients and rounding boundaries match sleefExpF32; the emitted source
 * has no exponential call or Euler constant. A subsequent streaming pass
 * lowers every Math.fround in this body to conditional arithmetic.
 */
export function directF32ExpPrefix(): string {
  const f = (expression: string): string => `Math.fround(${expression})`;
  const coefficients = [
    "0.00139304355252534151077271", "0.00833336077630519866943359",
    "0.0416664853692054748535156", "0.166666671633720397949219", "0.5",
  ];
  let polynomial = `let polynomial=${f("0.000198527617612853646278381")};`;
  for (const coefficient of coefficients) {
    polynomial += `polynomial=${f(`polynomial*reduced+${f(coefficient)}`)};`;
  }
  return `((value)=>{` +
    `const input=${f("value")};` +
    `if(input!==input)return NaN;if(input< -104)return 0;if(input>100)return Infinity;` +
    `const scaled=${f(`input*${f("1.4426950408889634")}`)};` +
    `let lower=0;if(scaled>=0){while(lower+1<=scaled)lower++;}` +
    `else{while(lower>scaled)lower--;}` +
    `const fraction=scaled-lower;` +
    `const exponent=lower+(fraction>0.5||(fraction===0.5&&lower%2!==0)?1:0);` +
    `let reduced=${f(`${f("exponent")}*${f("-0.693145751953125")}+input`)};` +
    `reduced=${f(`${f("exponent")}*${f("-1.428606765330187e-6")}+reduced`)};` +
    polynomial +
    `const squared=${f("reduced*reduced")};` +
    `const result=${f(`1+${f("squared*polynomial+reduced")}`)};` +
    `let half=0,remainder=exponent;` +
    `while(remainder<0){remainder+=2;half--;}` +
    `while(remainder>=2){remainder-=2;half++;}` +
    `let first=1,second=1;` +
    `for(let i=0;i<half;i++)first+=first;` +
    `for(let i=0;i>half;i--)first/=2;` +
    `const rest=exponent-half;` +
    `for(let i=0;i<rest;i++)second+=second;` +
    `for(let i=0;i>rest;i--)second/=2;` +
    `return ${f(`${f(`result*${f("first")}`)}*${f("second")}`)};` +
    `})(`;
}
