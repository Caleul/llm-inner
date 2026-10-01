import { readFileSync } from "node:fs";
import { DirectRustStream, rustF64, type RustExpression, type PositiveRoundRange } from "./direct-rust-stream.js";
import { decodeIeeeF16ToF32 } from "./utils.js";
import { f32BitsToDyadic,roundDyadicToF16IfElse } from "./fixed-f16-projection.js";

export interface NumericAffineRun {
  minimum:number; maximum:number; slope:number; offset:number;
  /** Constants preserve their signed zero without evaluating 0*x+b. */
  constant?:number;
  /** Compiler-certified F16 rounding of this exact F32 affine expression. */
  rounded?:"f16";
}
export type NumericRunConsumer=(run:NumericAffineRun)=>Promise<void>;

/** Midpoint-square certificate for the correctly rounded positive F32 root.
 * Compiler only: the host approximation supplies a seed, never a verdict.
 */
export function foldCertifiedF32Sqrt(x:number):number{
  if(!(x>0&&Number.isFinite(x)&&Math.fround(x)===x))throw new Error("Positive F32 root input required");
  const data=new DataView(new ArrayBuffer(4));
  const number=(code:number)=>{data.setUint32(0,code,true);return data.getFloat32(0,true);};
  data.setFloat32(0,Math.sqrt(x),true);let code=data.getUint32(0,true);
  for(;;){
    const y=number(code),below=(number(code-1)+y)/2,above=(y+number(code+1))/2;
    if(x<below*below||(x===below*below&&(code&1)!==0)){code--;continue;}
    if(x>above*above||(x===above*above&&(code&1)!==0)){code++;continue;}
    return y;
  }
}

/** F16 policy specialized directly to numerical input intervals. The profile
 * defines the declared backend rule, not model outputs. Every finite input in
 * an emitted affine run is checked at compilation; no representation bits,
 * table, numerical conversion or intermediate binding survives in Rust.
 */
export async function emitRustSilu(s:DirectRustStream,input:RustExpression,maximumMagnitude?:number,
  visit?:NumericRunConsumer,inputRange?:{minimum:number;maximum:number}):Promise<void>{
  if(inputRange&&(!visit||!Number.isFinite(inputRange.minimum)||!Number.isFinite(inputRange.maximum)||
    inputRange.minimum>inputRange.maximum))throw new Error("Invalid SiLU visitor domain");
  const profile=readFileSync(new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin",import.meta.url));
  if(profile.length!==131072)throw new Error("Invalid backend SiLU profile");
  let limit=31743;
  if(maximumMagnitude!==undefined && maximumMagnitude>=0 && maximumMagnitude<65504){
    let lo=0,hi=limit;
    while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(decodeIeeeF16ToF32(mid)<=maximumMagnitude)lo=mid;else hi=mid-1;}
    limit=lo;
  }
  const literal=(x:number)=>rustF64(Object.is(x,-0)?"-0":x);
  const value=async()=>{await s.write("(");await input();await s.write(")");};
  if(visit){if(!inputRange||(inputRange.minimum<=0&&inputRange.maximum>=0))
    await visit({minimum:0,maximum:0,slope:1,offset:0});}
  else {await s.write("'activation:{if ");await value();await s.write("==0.0 {break 'activation ");await value();await s.write(";}");}
  let firstX:number|undefined,firstY=0,lastX=0,slope:number|undefined,offset=0;
  let roundedFirst:number|undefined,roundedLast=0;
  const flushRounded=async()=>{
    if(roundedFirst===undefined)return;
    await visit!({minimum:roundedFirst,maximum:roundedLast,slope:0.5,offset:2**-26,rounded:"f16"});
    roundedFirst=undefined;
  };
  const reset=()=>{firstX=undefined;slope=undefined;offset=0;};
  const halfBuffer=new DataView(new ArrayBuffer(4));
  const flush=async(final=false)=>{
    if(firstX===undefined)return;
    if(visit){await visit({minimum:firstX,maximum:lastX,slope:slope??0,offset,
      ...((slope===undefined||slope===0)?{constant:firstY}:{})});return;}
    if(!final){await s.write("if ");await value();await s.write(`<=${literal(lastX)} {`);}
    await s.write("break 'activation ");
    if(slope===undefined||slope===0)await s.write(literal(firstY));
    else if(slope===1&&offset===0)await value();
    else {await s.write(`(${literal(slope)}*`);await value();await s.write(`+${literal(offset)})`);}
    await s.write(";");if(!final)await s.write("}");
  };
  const consume=async(bits:number)=>{
    const x=decodeIeeeF16ToF32(bits),y=decodeIeeeF16ToF32(profile.readUInt16LE(bits*2));
    if(visit&&Math.abs(x)<=2**-13){
      halfBuffer.setFloat32(0,0.5*x+2**-26,true);
      const code=halfBuffer.getUint32(0,true),rounded=(code&0x7fffffff)===0?(code>>>31?32768:0):
        roundDyadicToF16IfElse(f32BitsToDyadic(code));
      if(rounded===profile.readUInt16LE(bits*2)){
        if(firstX!==undefined){await flush();reset();}
        if(roundedFirst===undefined)roundedFirst=x;roundedLast=x;return;
      }
    }
    await flushRounded();
    if(firstX===undefined){firstX=lastX=x;firstY=y;return;}
    if(slope===undefined){
      if(Object.is(y,firstY)){slope=0;offset=firstY;lastX=x;return;}
      const a=(y-firstY)/(x-firstX),b=firstY-a*firstX;
      // The exact binary64 expression emitted below must reproduce BOTH
      // endpoints, including signed zero. Further points are checked before
      // joining this run, so no floating real-algebra assumption is made.
      if(Number.isFinite(a)&&Number.isFinite(b)&&Object.is(a*firstX+b,firstY)&&Object.is(a*x+b,y)){
        slope=a;offset=b;lastX=x;return;
      }
    }else if(Object.is(slope===0?firstY:slope*x+offset,y)){lastX=x;return;}
    await flush();firstX=lastX=x;firstY=y;slope=undefined;offset=0;
  };
  const firstAtLeast=(value:number)=>{
    let lo=1,hi=limit+1;while(lo<hi){const mid=Math.floor((lo+hi)/2);
      if(mid<=limit&&decodeIeeeF16ToF32(mid)>=value)hi=mid;else lo=mid+1;}return lo;
  };
  const lastAtMost=(value:number)=>{
    let lo=0,hi=limit;while(lo<hi){const mid=Math.ceil((lo+hi)/2);
      if(decodeIeeeF16ToF32(mid)<=value)lo=mid;else hi=mid-1;}return lo;
  };
  const negativeFirst=inputRange?firstAtLeast(-inputRange.maximum):1;
  const negativeLast=inputRange?lastAtMost(-inputRange.minimum):limit;
  for(let code=negativeLast;code>=negativeFirst;code--)await consume(32768+code);
  await flush();await flushRounded();reset();
  const positiveFirst=inputRange?firstAtLeast(inputRange.minimum):1;
  const positiveLast=inputRange?lastAtMost(inputRange.maximum):limit;
  for(let bits=positiveFirst;bits<=positiveLast;bits++)await consume(bits);
  await flush(true);await flushRounded();
  if(!visit){if(limit===0)await s.write('panic!("outside declared activation domain");');await s.write("}");}
}
/** Correctly-rounded F32 square root reduced into affine input runs. Each
 * numerical point is consumed, substituted into the current run and discarded.
 * Midpoint squares (at most 50 significant bits) are exact in binary64; they
 * certify the candidate independently of the host square-root approximation.
 */
export async function emitRustSqrt(s:DirectRustStream,input:RustExpression,positiveInput?:PositiveRoundRange,
  visit?:NumericRunConsumer):Promise<void>{
  const data=new DataView(new ArrayBuffer(4));
  const bits=(x:number)=>{data.setFloat32(0,x,true);return data.getUint32(0,true);};
  const number=(b:number)=>{data.setUint32(0,b,true);return data.getFloat32(0,true);};
  const minimum=positiveInput?.minimum??2**-149,maximum=positiveInput?.maximum??3.4028234663852886e38;
  if(!(minimum>0&&maximum>=minimum&&maximum<=3.4028234663852886e38))throw new Error("Undefined F32 square-root domain");
  let first=bits(minimum),last=bits(maximum);
  if(number(first)<minimum)first++;
  if(number(last)>maximum)last--;
  if(first>last)throw new Error("Empty F32 square-root domain");
  const root=(x:number)=>{
    let code=bits(Math.sqrt(x));
    for(;;){
      const y=number(code),below=(number(code-1)+y)/2,above=(y+number(code+1))/2;
      const lo=below*below,hi=above*above,odd=(code&1)!==0;
      if(x<lo||(x===lo&&odd)){code--;continue;}
      if(x>hi||(x===hi&&odd)){code++;continue;}
      return y;
    }
  };
  if(positiveInput&&positiveInput.minimum===positiveInput.maximum){
    if(number(first)!==positiveInput.minimum)throw new Error("Empty F32 square-root point domain");
    s.eliminatedBranches++;
    if(visit)await visit({minimum:number(first),maximum:number(first),slope:0,offset:0,constant:root(number(first))});
    else await s.write(rustF64(root(number(first))));return;
  }
  const value=async()=>{await s.write("(");await input();await s.write(")");};
  if(!visit){await s.write("'root_choice:{");
    if(!positiveInput){await s.write("if ");await value();await s.write("==0.0 {break 'root_choice ");await value();await s.write(";}");}}
  else if(!positiveInput)await visit({minimum:0,maximum:0,slope:1,offset:0});
  let firstX=number(first),firstY=root(firstX),lastX=firstX,slope:number|undefined,offset=0;
  const flush=async(final=false)=>{
    if(visit){await visit({minimum:firstX,maximum:lastX,slope:slope??0,offset,
      ...((slope===undefined||slope===0)?{constant:firstY}:{})});return;}
    if(!final){await s.write("if ");await value();await s.write(`<=${rustF64(lastX)} {`);}
    await s.write("break 'root_choice ");
    if(slope===undefined||slope===0)await s.write(rustF64(firstY));
    else if(slope===1&&offset===0)await value();
    else {await s.write(`(${rustF64(slope)}*`);await value();await s.write(`+${rustF64(offset)})`);}
    await s.write(";");if(!final)await s.write("}");
  };
  for(let code=first+1;code<=last;code++){
    const x=number(code),y=root(x);
    if(slope===undefined){
      const a=(y-firstY)/(x-firstX),b=firstY-a*firstX;
      if(Number.isFinite(a)&&Number.isFinite(b)&&Object.is(a*firstX+b,firstY)&&Object.is(a*x+b,y)){slope=a;offset=b;lastX=x;continue;}
    }else if(Object.is(slope===0?firstY:slope*x+offset,y)){lastX=x;continue;}
    await flush();firstX=lastX=x;firstY=y;slope=undefined;offset=0;
  }
  await flush(true);if(!visit)await s.write("}");
}
export function foldDeclaredCpuF32Exponential(x:number):number{
    if(x< -104)return 0;if(x>100)return Infinity;
    const scaled=Math.fround(x*Math.fround(1.4426950408889634));
    const lower=Math.floor(scaled),fraction=scaled-lower;
    const exponent=lower+(fraction>0.5||(fraction===0.5&&lower%2!==0)?1:0);
    let reduced=Math.fround(exponent*(-0.693145751953125)+x);
    reduced=Math.fround(exponent*Math.fround(-1.428606765330187e-6)+reduced);
    let polynomial=Math.fround(0.000198527617612853646278381);
    for(const coefficient of [0.00139304355252534151077271,0.00833336077630519866943359,
      0.0416664853692054748535156,0.166666671633720397949219,0.5])
      polynomial=Math.fround(polynomial*reduced+Math.fround(coefficient));
    const squared=Math.fround(reduced*reduced),tail=Math.fround(squared*polynomial+reduced);
    const result=Math.fround(1+tail),half=Math.floor(exponent/2);
    return Math.fround((result*2**half)*2**(exponent-half));
}
/** Fold the declared F32 range-reduction polynomial at compile time. No
 * exponential primitive, Euler constant or numerical temporary is emitted.
 * Runs are joined only after checking the emitted binary64 affine expression
 * at every discrete F32 point; floating identities are never assumed.
 */
export async function emitRustExp(s:DirectRustStream,input:RustExpression,alreadyF32=false,
  nonpositiveMagnitudeBound?:number,inputRange?:{minimum:number;maximum:number},inputQuantum?:number,
  visit?:NumericRunConsumer):Promise<void>{
  const range=inputRange??(nonpositiveMagnitudeBound===undefined?undefined:
    {minimum:-nonpositiveMagnitudeBound,maximum:0});
  if(!range||!Number.isFinite(range.minimum)||!Number.isFinite(range.maximum)||range.minimum>range.maximum)
    throw new Error("Exponential substitution requires a proven finite input interval");
  const data=new DataView(new ArrayBuffer(4));
  const bits=(x:number)=>{data.setFloat32(0,x,true);return data.getUint32(0,true);};
  const number=(b:number)=>{data.setUint32(0,b,true);return data.getFloat32(0,true);};
  const literal=(x:number)=>x===Infinity?"f64::INFINITY":rustF64(Object.is(x,-0)?"-0":x);
  if(range.minimum===range.maximum){
    const point=Math.fround(range.minimum);
    if(alreadyF32&&point!==range.minimum)throw new Error("Empty F32 exponential point domain");
    s.eliminatedBranches++;
    if(visit)await visit({minimum:point,maximum:point,slope:0,offset:0,constant:foldDeclaredCpuF32Exponential(point)});
    else await s.write(literal(foldDeclaredCpuF32Exponential(point)));return;
  }
  const value=async()=>{await s.write("(");if(alreadyF32)await input();else await s.round("f32",input);await s.write(")");};
  if(!visit)await s.write("'exponential_choice:{");
  let firstX:number|undefined,firstY=0,lastX=0,slope:number|undefined,offset=0;
  const flush=async(final=false)=>{
    if(firstX===undefined)return;
    if(visit){await visit({minimum:firstX,maximum:lastX,slope:slope??0,offset,
      ...((slope===undefined||slope===0)?{constant:firstY}:{})});return;}
    if(!final){await s.write("if ");await value();await s.write(`<=${literal(lastX)} {`);}
    await s.write("break 'exponential_choice ");
    if(slope===undefined||slope===0)await s.write(literal(firstY));
    else if(slope===1&&offset===0)await value();
    else {await s.write(`(${literal(slope)}*`);await value();await s.write(`+${literal(offset)})`);}
    await s.write(";");if(!final)await s.write("}");
  };
  const consume=async(x:number)=>{
    const y=foldDeclaredCpuF32Exponential(x);
    if(firstX===undefined){firstX=lastX=x;firstY=y;return;}
    if(slope===undefined){
      if(Object.is(y,firstY)){slope=0;offset=firstY;lastX=x;return;}
      const a=(y-firstY)/(x-firstX),b=firstY-a*firstX;
      if(Number.isFinite(a)&&Number.isFinite(b)&&Object.is(a*firstX+b,firstY)&&Object.is(a*x+b,y)){
        slope=a;offset=b;lastX=x;return;
      }
    }else if(Object.is(slope===0?firstY:slope*x+offset,y)){lastX=x;return;}
    await flush();firstX=lastX=x;firstY=y;slope=undefined;offset=0;
  };
  if(inputQuantum!==undefined){
    if(!(inputQuantum>0&&Number.isFinite(inputQuantum)&&Math.log2(inputQuantum)%1===0))
      throw new Error("Undefined exponential input lattice");
    const first=Math.ceil(range.minimum/inputQuantum),last=Math.floor(range.maximum/inputQuantum);
    if(!Number.isSafeInteger(first)||!Number.isSafeInteger(last))throw new Error("Exponential lattice exceeds exact index domain");
    let previous:number|undefined;
    for(let index=first;index<=last;index++){
      const x=Math.fround(index*inputQuantum);
      if(x<range.minimum||x>range.maximum||Object.is(x,previous))continue;
      previous=x;await consume(x);
    }
    if(firstX===undefined)throw new Error("Empty exponential input lattice");
    await flush(true);if(!visit)await s.write("}");return;
  }
  // Enumerate in numerical order, consuming and discarding one F32 value.
  // Domains are inclusive and are rounded outward when the input itself has
  // not yet been narrowed to F32 by its producer.
  const min=alreadyF32?range.minimum:Math.fround(range.minimum);
  const max=alreadyF32?range.maximum:Math.fround(range.maximum);
  // For this polynomial, in [-2^-25,2^-24] the range-reduction
  // exponent is zero. The squared polynomial term is nonnegative and below
  // half an ULP of the tail. Tail rounding therefore cannot cross either
  // midpoint of 1.0; the even significand of 1.0 wins both endpoint ties.
  // This proven constant cell skips all smaller exponents, including zeros.
  const constantMin=-(2**-25),constantMax=2**-24;
  if(min<constantMin){
    let first=bits(min),last=bits(Math.min(max,constantMin));
    if(number(first)<min)first--;if(number(last)>max)last++;
    if(number(last)>=constantMin)last++;
    for(let code=first;code>=last;code--)await consume(number(code));
  }
  const centralMin=Math.max(min,constantMin),centralMax=Math.min(max,constantMax);
  if(centralMin<=centralMax){
    await consume(centralMin);if(centralMax>centralMin)await consume(centralMax);
  }
  if(max>constantMax){
    let first=bits(Math.max(min,constantMax)),last=bits(max);
    if(number(first)<min||number(first)<=constantMax)first++;
    if(number(last)>max)last--;
    for(let code=first;code<=last;code++)await consume(number(code));
  }
  if(firstX===undefined)throw new Error("Empty F32 exponential domain");
  await flush(true);if(!visit)await s.write("}");
}
