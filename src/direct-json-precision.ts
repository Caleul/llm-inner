import {jsonOperation as o,jsonWidths,jsonInteger,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue} from './direct-json-evaluator.js';

/** Compiler-only certificates: number of low IEEE bitword bits known to be
 * zero. Producers must prove these facts for their complete input domain.
 * They are never serialized or treated as runtime annotations. */
export type JsonPrecisionFacts=WeakMap<JsonExpression,number>;
export interface JsonPrecisionStats { redundantMasks:number; redundantRoundings:number }

/** Cancel bitword rounding only when its input is already on that lattice.
 * No real-number factoring, change of float operation order, or elimination of
 * an evaluated operand is allowed. Masks also supply facts without certificates. */
export function simplifyJsonBitPrecision(root:JsonExpression,facts:JsonPrecisionFacts=new WeakMap(),
  stats:JsonPrecisionStats={redundantMasks:0,redundantRoundings:0}):JsonExpression {
  const memo=new WeakMap<object,JsonExpression>(),zerosMemo=new WeakMap<object,number>();
  const integer=(node:JsonExpression|undefined)=>node?.[0]==='constant'&&jsonInteger(node[1])?
    jsonConstantValue(node) as bigint:undefined;
  function zeros(node:JsonExpression):number {
    const hit=zerosMemo.get(node);if(hit!==undefined)return hit;
    const width=jsonWidths[node[1]],known=facts.get(node);
    let result=known??0;
    if(node[0]==='constant'&&node[1]!=='bool'){
      let bits=BigInt(node[2]);result=0;
      while(result<width&&(bits&1n)===0n){result++;bits>>=1n;}
    }else if(known===undefined&&node[0]!=='input'){
      const a=node[2] as JsonExpression,b=node[3] as JsonExpression|undefined;
      if(node[0]==='reinterpret'||node[0]==='convert')result=Math.min(width,zeros(a));
      else if(node[0]==='if')result=Math.min(zeros(node[3]!),zeros(node[4]!));
      else if(jsonInteger(node[1])){
        if(node[0]==='and')result=Math.max(zeros(a),zeros(b!));
        else if(['or','xor','add','sub'].includes(node[0]))result=Math.min(zeros(a),zeros(b!));
        else if(node[0]==='mul')result=Math.min(width,zeros(a)+zeros(b!));
        else if(node[0]==='shl'||node[0]==='shr'){
          const shift=integer(b);
          if(shift!==undefined&&shift<BigInt(width))result=node[0]==='shl'?
            Math.min(width,zeros(a)+Number(shift)):Math.max(0,zeros(a)-Number(shift));
        }
      }else if(node[1]==='f64'&&node[0]==='sub'&&a[0]==='constant'&&
        Object.is(jsonConstantValue(a),-0)){
        // IEEE -0 minus a float flips its sign (including both signed zeros).
        // This rule is enabled only for a certified value, not arbitrary NaNs.
        result=facts.get(b!)??0;
      }
    }
    zerosMemo.set(node,result);return result;
  }
  function visit(node:JsonExpression):JsonExpression {
    const hit=memo.get(node);if(hit)return hit;
    if(node[0]==='constant'||node[0]==='input')return node;
    const args=(node.slice(2) as JsonExpression[]).map(visit);
    let result=o(node[0],node[1],...args);
    if(node[0]==='and'&&jsonInteger(node[1])){
      let [source,mask]=args;
      if(integer(source)!==undefined)[source,mask]=[mask,source];
      const maskValue=integer(mask),width=jsonWidths[node[1]];
      if(maskValue!==undefined){
        const all=(1n<<BigInt(width))-1n,dropped=all^maskValue;
        // A contiguous mask of low bits, including the empty mask.
        if((dropped&(dropped+1n))===0n){
          const count=dropped===0n?0:dropped.toString(2).length;
          if(zeros(source!)>=count){result=source!;stats.redundantMasks++;}
          else if(count>0&&source![0]==='add'){
            const bits=source![2]!,bias=source![3]!;
            if(bias[0]==='add'&&integer(bias[2])===(1n<<BigInt(count-1))-1n){
              const odd=bias[3]!;
              if(odd[0]==='and'&&integer(odd[3])===1n&&odd[2]![0]==='shr'&&
                odd[2]![2]===bits&&integer(odd[2]![3])===BigInt(count)&&zeros(bits)>=count){
                result=bits;stats.redundantRoundings++;
              }
            }
          }
        }
      }
    }
    const certified=facts.get(node);if(certified!==undefined)facts.set(result,certified);
    memo.set(node,result);return result;
  }
  return visit(root);
}
