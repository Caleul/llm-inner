import type {JsonFloatRange} from './direct-json-range.js';

/** Conservative one-ULP enclosure of an exact operation on finite F64 bounds.
 * This is a compiler proof, not arithmetic inserted into the model. */
function outward64(value:number,direction:-1|1):number {
  if(!Number.isFinite(value))return value;
  if(value===0)return direction*Number.MIN_VALUE;
  const word=new DataView(new ArrayBuffer(8));word.setFloat64(0,value);
  word.setBigUint64(0,word.getBigUint64(0)+BigInt(value>0?direction:-direction));return word.getFloat64(0);
}
/** Return a constant only when the complete real arithmetic enclosure lies
 * within a single nonzero finite F32 rounding cell. Padding also protects
 * endpoint ties from F64 rounding. Zero cells are deliberately not admitted:
 * a signed interval alone does not certify the sign of zero. */
export function constantJsonF32Cell(operation:string,a?:JsonFloatRange,b?:JsonFloatRange):number|undefined {
  if(!a||!b||![a.minimum,a.maximum,b.minimum,b.maximum].every(Number.isFinite)||
    a.minimum>a.maximum||b.minimum>b.maximum)return undefined;
  let endpoints:number[];
  if(operation==='add')endpoints=[a.minimum+b.minimum,a.maximum+b.maximum];
  else if(operation==='sub')endpoints=[a.minimum-b.maximum,a.maximum-b.minimum];
  else if(operation==='mul')endpoints=[a.minimum,a.maximum].flatMap(x=>[b.minimum,b.maximum].map(y=>x*y));
  else return undefined;
  const low=Math.fround(outward64(Math.min(...endpoints),-1));
  const high=Math.fround(outward64(Math.max(...endpoints),1));
  return Number.isFinite(low)&&low!==0&&Object.is(low,high)?low:undefined;
}
