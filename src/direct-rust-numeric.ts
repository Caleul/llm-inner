import { readFileSync } from "node:fs";
import { DirectRustStream, rustF64, type RustExpression } from "./direct-rust-stream.js";

/** Inline finite binary16 decoding from representation bits, using affine branches. */
export async function emitRustHalfDecode(s: DirectRustStream, input: RustExpression): Promise<void> {
  await s.write("{let bits:f64="); await input();
  await s.write(";let negative:bool=bits>=32768.0;let code:f64=if negative {bits-32768.0} else {bits};assert!(code<31744.0);let mut value:f64=code*5.960464477539063e-8;");
  for(let exponent=1;exponent<=30;exponent++) {
    await s.write(`if code>=${exponent*1024}.0 && code<${(exponent+1)*1024}.0 {value=(code-${exponent*1024}.0+1024.0)*${rustF64(2**(exponent-25))};}`);
  }
  await s.write("if negative {-value} else {value}}");
}
/** Exact representation conversion on the already-rounded finite half domain. */
async function emitRustHalfEncode(s: DirectRustStream, input: RustExpression): Promise<void> {
  await s.write("{let value:f64=");await input();
  await s.write(";assert!(value==value && value<=65504.0 && value>=-65504.0);let negative:bool=value<0.0 || (value==0.0 && 1.0/value<0.0);let magnitude:f64=if negative {-value} else {value};let mut code:f64=magnitude*16777216.0;");
  for(let exponent=1;exponent<=30;exponent++) {
    const base=2**(exponent-15);
    await s.write(`if magnitude>=${rustF64(base)} && magnitude<${rustF64(base*2)} {code=${exponent*1024}.0+(magnitude/${rustF64(base)}-1.0)*1024.0;}`);
  }
  await s.write("code+if negative {32768.0} else {0.0}}");
}
/** Profile is a declared backend rule consumed during compilation, never a runtime table. */
export async function emitRustSilu(s: DirectRustStream,input: RustExpression): Promise<void> {
  const profile=readFileSync(new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin",import.meta.url));
  if(profile.length!==131072)throw new Error("Invalid backend SiLU profile");
  await emitRustHalfDecode(s,async()=>{
    await s.write("'silu:{let bits:f64=");await emitRustHalfEncode(s,input);await s.write(";");
    // Stream maximal affine runs in exact representation space. No numeric DAG.
    let start=0,first=profile.readUInt16LE(0),end=0,slope:number|undefined;
    const flush=async():Promise<void>=>{
      const k=slope??0,b=first-k*start;
      if(end===64511)await s.write(`break 'silu ${k}.0*bits+(${b}.0);`);
      else await s.write(`if bits<=${end}.0 {break 'silu ${k}.0*bits+(${b}.0);}`);
    };
    for(let bits=1;bits<=64512;bits++) {
      if(bits===31744||bits===64512){await flush();}
      if(bits===64512)break;
      if(bits>=31744&&bits<32768||bits>=64512)continue;
      const output=profile.readUInt16LE(bits*2);
      if(bits===32768){start=end=bits;first=output;slope=undefined;continue;}
      const delta=output-(first+(slope??0)*(end-start));
      if(slope===undefined||slope===delta){slope=delta;end=bits;}
      else {await flush();start=end=bits;first=output;slope=undefined;}
    }
    await s.write("}");
  });
}
export async function emitRustSqrt(s:DirectRustStream,input:RustExpression):Promise<void>{
  await s.round("f32",async()=>{
    await s.write("{let value:f64=");await input();
    await s.write(";assert!(value>=0.0 && value<=1.7976931348623157e308);let mut scaled:f64=value;let mut scale:f64=1.0;if value!=0.0 {while scaled>=4.0 {scaled/=4.0;scale+=scale;}while scaled<1.0 {scaled*=4.0;scale/=2.0;}}let mut root:f64=if scaled>=2.0 {2.0} else {1.0};");
    for(let i=0;i<9;i++)await s.write("root=(root+scaled/root)/2.0;");
    await s.write("if value==0.0 {value} else {root*scale}}");
  });
}
export async function emitRustExp(s:DirectRustStream,input:RustExpression, alreadyF32=false):Promise<void>{
  const r=async(expression:string)=>s.round("f32",()=>s.write(expression));
  await s.write("'exponential:{let input:f64=");if(alreadyF32)await input();else await s.round("f32",input);await s.write(";if input< -104.0 {break 'exponential 0.0;}if input>100.0 {break 'exponential f64::INFINITY;}let scaled:f64=");
  await r(`input*${rustF64(Math.fround(1.4426950408889634))}`);
  await s.write(";let mut lower:f64=0.0;if scaled>=0.0 {while lower+1.0<=scaled {lower+=1.0;}} else {while lower>scaled {lower-=1.0;}}let fraction:f64=scaled-lower;let exponent:f64=lower+if fraction>0.5 || (fraction==0.5 && lower%2.0!=0.0) {1.0} else {0.0};let mut reduced:f64=");
  await r("exponent*(-0.693145751953125)+input");await s.write(";reduced=");
  await r(`exponent*${rustF64(Math.fround(-1.428606765330187e-6))}+reduced`);
  await s.write(`;let mut polynomial:f64=${rustF64(Math.fround(0.000198527617612853646278381))};`);
  for(const coefficient of [0.00139304355252534151077271,0.00833336077630519866943359,0.0416664853692054748535156,0.166666671633720397949219,0.5]) {
    await s.write("polynomial=");await r(`polynomial*reduced+${rustF64(Math.fround(coefficient))}`);await s.write(";");
  }
  await s.write("let squared:f64=");await r("reduced*reduced");await s.write(";let tail:f64=");await r("squared*polynomial+reduced");
  await s.write(";let result:f64=");await r("1.0+tail");
  await s.write(";let mut half:f64=0.0;let mut remainder:f64=exponent;while remainder<0.0 {remainder+=2.0;half-=1.0;}while remainder>=2.0 {remainder-=2.0;half+=1.0;}let mut first:f64=1.0;let mut second:f64=1.0;let mut i:f64=0.0;while i<half {first+=first;i+=1.0;}while i>half {first/=2.0;i-=1.0;}let rest:f64=exponent-half;i=0.0;while i<rest {second+=second;i+=1.0;}while i>rest {second/=2.0;i-=1.0;}let first_product:f64=");
  await s.write("result*first;");await r("first_product*second");await s.write("}");
}
