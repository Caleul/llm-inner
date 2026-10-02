import {jsonConstantValue} from './direct-json-evaluator.js';
import {certifyJsonSmallSilu} from './direct-json-silu.js';
import {decodeIeeeF16ToF32} from './utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from './fixed-f16-projection.js';
import type {JsonExpression} from './direct-json-expression.js';
import type {JsonModelLoweringFacts} from './direct-json-model.js';

export interface JsonFloatRange {minimum:number;maximum:number}
const word=new DataView(new ArrayBuffer(4));
/** An outward F32 step covers double rounding in the endpoint calculation.
 * Bounds are compiler-only; this padding never changes the model arithmetic. */
function outward(value:number,direction:-1|1):number {
  const rounded=Math.fround(value);
  if(!Number.isFinite(rounded))return rounded;
  if(rounded===0)return direction*2**-149;
  word.setFloat32(0,rounded);
  const bits=word.getUint32(0)+(rounded>0?direction:-direction);
  word.setUint32(0,bits);return word.getFloat32(0);
}
function half(value:number):number {
  if(Object.is(value,-0))return -0;
  word.setFloat32(0,value);
  return decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(word.getUint32(0))));
}
function finite(minimum:number,maximum:number):JsonFloatRange|undefined {
  return Number.isFinite(minimum)&&Number.isFinite(maximum)&&minimum<=maximum?{minimum,maximum}:undefined;
}
/** Conservative intervals over working expressions, before bitwise lowering.
 * RMS correlated bounds come from the builder's checkpoint certificate, not
 * from treating x and its normalization denominator as independent variables.
 * No unknown operation or zero-crossing divisor receives an invented bound. */
export function jsonModelRangeAnalysis(facts:JsonModelLoweringFacts):
  (node:JsonExpression)=>JsonFloatRange|undefined {
  const memo=new WeakMap<object,JsonFloatRange|undefined>(),seen=new WeakSet<object>();
  function range(node:JsonExpression):JsonFloatRange|undefined {
    if(seen.has(node))return memo.get(node);
    seen.add(node);const result=calculate(node);memo.set(node,result);return result;
  }
  function calculate(node:JsonExpression):JsonFloatRange|undefined {
    const certificate=facts.ranges?.get(node);if(certificate)return certificate;
    if(node[0]==='constant'){
      if(!node[1].startsWith('f'))return undefined;
      const x=Number(jsonConstantValue(node));return finite(x,x);
    }
    if(node[0]==='input')return node[1]==='f16'?{minimum:-65504,maximum:65504}:undefined;
    const source=facts.halfSources.get(node);
    if(source){const r=range(source);return r?finite(half(r.minimum),half(r.maximum)):undefined;}
    if(node[0]==='widen')return range(node[2]!);
    if(node[0]==='pending-silu'){
      const bound=facts.activationBounds.get(node);if(bound===undefined||bound>.1)return undefined;
      const maximum=certifyJsonSmallSilu(bound).maximumMagnitude;
      return {minimum:-maximum,maximum};
    }
    if(node[0]==='pending-exp'){
      const bound=facts.exponentialBounds.get(node);
      // The declared Horner polynomial on [-.34,0] has positive factors,
      // square*factor < -x, and 1+x >= .66. This loose enclosure includes
      // every explicit F32 rounding and the constant-one tiny arm.
      return bound!==undefined&&bound<=.34?{minimum:.5,maximum:1.01}:undefined;
    }
    if(node[0]==='if'){
      const yes=range(node[3]!),no=range(node[4]!);
      return yes&&no?finite(Math.min(yes.minimum,no.minimum),Math.max(yes.maximum,no.maximum)):undefined;
    }
    if(node[0]==='pending-sqrt'){
      const a=range(node[2]!);return a&&a.minimum>0?
        finite(outward(Math.sqrt(a.minimum),-1),outward(Math.sqrt(a.maximum),1)):undefined;
    }
    if(node[1]!=='f32'||!['add','sub','mul','div'].includes(node[0]))return undefined;
    const a=range(node[2]!),b=range(node[3]!);if(!a||!b)return undefined;
    let minimum:number,maximum:number;
    if(node[0]==='add'){minimum=a.minimum+b.minimum;maximum=a.maximum+b.maximum;}
    else if(node[0]==='sub'){minimum=a.minimum-b.maximum;maximum=a.maximum-b.minimum;}
    else {
      if(node[0]==='div'&&b.minimum<=0&&b.maximum>=0)return undefined;
      const values=[a.minimum,a.maximum].flatMap(x=>[b.minimum,b.maximum].map(y=>node[0]==='mul'?x*y:x/y));
      minimum=Math.min(...values);maximum=Math.max(...values);
      if(node[0]==='mul'&&node[2]===node[3])minimum=a.minimum<=0&&a.maximum>=0?0:
        Math.min(a.minimum*a.minimum,a.maximum*a.maximum);
    }
    let lo=outward(minimum,-1),hi=outward(maximum,1);
    if(minimum>=0&&((node[0]==='mul'&&node[2]===node[3])||
      node[0]==='add'&&a.minimum>=0&&b.minimum>=0))lo=Math.max(0,lo);
    return finite(lo,hi);
  }
  return range;
}
