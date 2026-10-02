import {jsonConstantValue} from './direct-json-evaluator.js';
import type {JsonExpression} from './direct-json-expression.js';
import type {JsonModelLoweringFacts} from './direct-json-model.js';
import {jsonModelRangeAnalysis,outwardJsonF32Bound,roundJsonF32BoundToHalf} from './direct-json-range.js';

export interface JsonMagnitudeRange {minimum:number;maximum:number}
/** Compiler-only unsigned magnitude facts retain gaps around zero which a
 * signed interval cannot represent. A scoped session owns its own proof cache.
 * No scope or proof metadata is serialized into the resulting scalar JSON. */
export function jsonModelMagnitudeAnalysis(facts:JsonModelLoweringFacts):
  (node:JsonExpression)=>JsonMagnitudeRange|undefined {
  for(const [name,bound] of facts.inputMagnitudeBounds??[])
    if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(name)||!Number.isFinite(bound.minimum)||!Number.isFinite(bound.maximum)||
      bound.minimum<0||bound.minimum>bound.maximum||bound.maximum>65504)
      throw new RangeError('Invalid finite half input magnitude proof');
  const signed=jsonModelRangeAnalysis(facts),memo=new WeakMap<JsonExpression,JsonMagnitudeRange|undefined>(),seen=new WeakSet<JsonExpression>();
  const lower=(x:number)=>Math.max(0,outwardJsonF32Bound(x,-1)),upper=(x:number)=>outwardJsonF32Bound(x,1);
  function magnitude(node:JsonExpression):JsonMagnitudeRange|undefined {
    if(seen.has(node))return memo.get(node);seen.add(node);
    let result=calculate(node);const certificate=signed(node);
    if(certificate){
      const maximum=Math.max(Math.abs(certificate.minimum),Math.abs(certificate.maximum));
      const minimum=certificate.minimum>0?certificate.minimum:certificate.maximum<0?-certificate.maximum:0;
      result=result?{minimum:Math.max(minimum,result.minimum),maximum:Math.min(maximum,result.maximum)}:{minimum,maximum};
    }
    if(result&&(!Number.isFinite(result.minimum)||!Number.isFinite(result.maximum)||
      result.minimum<0||result.minimum>result.maximum))result=undefined;
    memo.set(node,result);return result;
  }
  function calculate(node:JsonExpression):JsonMagnitudeRange|undefined {
    if(node[0]==='constant'){
      if(!node[1].startsWith('f'))return undefined;
      const value=Math.abs(Number(jsonConstantValue(node)));return Number.isFinite(value)?{minimum:value,maximum:value}:undefined;
    }
    if(node[0]==='input')return node[1]==='f16'?facts.inputMagnitudeBounds?.get(node[2])??{minimum:0,maximum:65504}:undefined;
    const source=facts.halfSources.get(node);
    if(source){
      const value=magnitude(source);return value?{minimum:roundJsonF32BoundToHalf(value.minimum),maximum:roundJsonF32BoundToHalf(value.maximum)}:undefined;
    }
    if(node[0]==='widen')return magnitude(node[2]!);
    if(node[0]==='if'){
      const a=magnitude(node[3]!),b=magnitude(node[4]!);
      return a&&b?{minimum:Math.min(a.minimum,b.minimum),maximum:Math.max(a.maximum,b.maximum)}:undefined;
    }
    if(node[0]==='pending-sqrt'&&facts.positiveNormalRoots.has(node)){
      const value=magnitude(node[2]!);return value&&value.minimum>0?
        {minimum:lower(Math.sqrt(value.minimum)),maximum:upper(Math.sqrt(value.maximum))}:undefined;
    }
    if(node[1]!=='f32'||!['add','sub','mul','div'].includes(node[0]))return undefined;
    const a=magnitude(node[2]!),b=magnitude(node[3]!);if(!a||!b)return undefined;
    if(node[0]==='mul')return {minimum:lower(a.minimum*b.minimum),maximum:upper(a.maximum*b.maximum)};
    if(node[0]==='div')return b.minimum>0?{minimum:lower(a.minimum/b.maximum),maximum:upper(a.maximum/b.minimum)}:undefined;
    const ar=signed(node[2]!),br=signed(node[3]!);
    const aligned=ar&&br&&(node[0]==='add'?
      ar.minimum>=0&&br.minimum>=0||ar.maximum<=0&&br.maximum<=0:
      ar.minimum>=0&&br.maximum<=0||ar.maximum<=0&&br.minimum>=0);
    const minimum=aligned?a.minimum+b.minimum:Math.max(0,a.minimum-b.maximum,b.minimum-a.maximum);
    return {minimum:lower(minimum),maximum:upper(a.maximum+b.maximum)};
  }
  return magnitude;
}
