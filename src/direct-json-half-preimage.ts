import {decodeIeeeF16ToF32} from './utils.js';

/** Greatest magnitude of a finite F32 source whose rounded finite half result
 * has magnitude <= bound. This is a backwards compilation proof, never a
 * floating conversion or lookup executed by the emitted artifact. */
export function maximumF32MagnitudeForHalfBound(bound:number):number|undefined {
  if(!Number.isFinite(bound)||bound<0)return undefined;
  let low=0,high=0x7bff;
  while(low<high){
    const middle=Math.ceil((low+high)/2);
    if(decodeIeeeF16ToF32(middle)<=bound)low=middle;else high=middle-1;
  }
  const last=decodeIeeeF16ToF32(low),next=low===0x7bff?65536:decodeIeeeF16ToF32(low+1);
  const midpoint=(last+next)/2;
  // Every half midpoint is exactly representable in F32. At an odd code
  // the midpoint belongs to the successor; finite-max also excludes its tie.
  if((low&1)===0)return midpoint;
  const word=new DataView(new ArrayBuffer(4));word.setFloat32(0,midpoint);
  word.setUint32(0,word.getUint32(0)-1);return word.getFloat32(0);
}
