import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue} from './direct-json-evaluator.js';
import {sameJsonExpression} from './direct-json-simplify.js';
import type {JsonModelLoweringFacts} from './direct-json-model.js';
import type {JsonFloatRange} from './direct-json-range.js';

/** Compiler-only sign provenance for the finite model domain. F32 rounding and
 * F16 conversion retain sign, including underflow to -0. No proof is inferred
 * from a non-strict interval, because comparison treats both zeros as equal. */
export function createJsonModelSignProof(facts:JsonModelLoweringFacts,
  lower:(node:JsonExpression)=>JsonExpression,range:(node:JsonExpression)=>JsonFloatRange|undefined):
  (node:JsonExpression,closed:JsonExpression)=>JsonExpression {
  const memo=new WeakMap<JsonExpression,JsonExpression>(),sign=0x8000000000000000n;
  function prove(node:JsonExpression,closed?:JsonExpression):JsonExpression {
    if(!node[1].startsWith('f'))throw new TypeError('Floating sign provenance source required');
    const hit=memo.get(node);if(hit)return hit;
    let result:JsonExpression|undefined;
    if(node[0]==='constant'){
      const value=Number(jsonConstantValue(node));
      if(Number.isFinite(value))result=c('u64',value<0||Object.is(value,-0)?sign:0n);
    }else if(node[0]==='widen')result=prove(node[2]!);
    else if(facts.halfSources.has(node))result=prove(facts.halfSources.get(node)!);
    else if(node[0]==='pending-sqrt'&&facts.positiveNormalRoots.has(node)||
      node[0]==='pending-exp'&&facts.exponentialBounds.has(node))result=c('u64',0n);
    // The bounded SiLU certificate checks all finite half inputs in its
    // domain, including signed zeros. Its finite outputs preserve input sign.
    else if(node[0]==='pending-silu'&&facts.activationBounds.has(node))result=prove(node[2]!);
    else if(node[1]==='f32'&&(node[0]==='mul'||node[0]==='div')){
      const left=prove(node[2]!),right=prove(node[3]!);
      result=sameJsonExpression(left,right)?c('u64',0n):
        left[0]==='constant'&&jsonConstantValue(left)===0n?right:
        right[0]==='constant'&&jsonConstantValue(right)===0n?left:o('xor','u64',left,right);
    }
    else if(node[1]==='f32'&&node[0]==='add'){
      const left=prove(node[2]!),right=prove(node[3]!);
      if(sameJsonExpression(left,right))result=left;
    }
    if(!result){
      const interval=range(node);
      if(interval&&interval.minimum>0)result=c('u64',0n);
      else if(interval&&interval.maximum<0)result=c('u64',sign);
      else result=o('and','u64',o('reinterpret','u64',closed??lower(node)),c('u64',sign));
    }
    memo.set(node,result);return result;
  }
  return prove;
}
