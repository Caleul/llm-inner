/** Decode a finite F16 bit pattern using only guards and affine leaves. */
export function compileF16BitDecodeBranches(): string {
  const clauses: string[] = [
    "if(bits<0||bits>65535||(bits>=31744&&bits<32768)||bits>=64512)throw new RangeError('F16 não finito');",
    `if(bits<1024)return bits*${2 ** -24};`,
  ];
  for (let exponent = 1; exponent <= 30; exponent++) {
    const start = exponent * 1024;
    clauses.push(`if(bits<${start + 1024})return (bits-${start}+1024)*${2 ** (exponent - 25)};`);
  }
  clauses.push("if(bits===32768)return 0;");
  clauses.push(`if(bits<33792)return (bits-32768)*${-(2 ** -24)};`);
  for (let exponent = 1; exponent <= 30; exponent++) {
    const start = 32768 + exponent * 1024;
    clauses.push(`if(bits<${start + 1024})return (bits-${start}+1024)*${-(2 ** (exponent - 25))};`);
  }
  clauses.push("throw new RangeError('F16 não finito');");
  return clauses.join("");
}
