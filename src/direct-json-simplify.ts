import {jsonConstant,jsonInteger,jsonOperation,jsonWidths,type JsonExpression} from './direct-json-expression.js';
import {evaluateJsonExpression,jsonConstantValue} from './direct-json-evaluator.js';
import {jsonIntegerRange} from './direct-json-integer-range.js';

export interface JsonSimplificationStats { visited:number; folds:number; conditions:number; integerAlgebra:number }
export const newJsonSimplificationStats=():JsonSimplificationStats=>({visited:0,folds:0,conditions:0,integerAlgebra:0});
/** Structural equality only; no numerical equivalence inferred from rendered text. */
export function sameJsonExpression(a:JsonExpression,b:JsonExpression):boolean {
  const pairs:([JsonExpression,JsonExpression])[]=[[a,b]],seen=new WeakMap<object,WeakSet<object>>();
  while(pairs.length){
    const [left,right]=pairs.pop()!;
    if(left===right)continue;
    if(left.length!==right.length||left[0]!==right[0]||left[1]!==right[1])return false;
    if(left[0]==='input'||left[0]==='constant'){if(left[2]!==right[2])return false;continue;}
    if(right[0]==='input'||right[0]==='constant')return false;
    let matches=seen.get(left);if(matches?.has(right))continue;
    if(!matches){matches=new WeakSet();seen.set(left,matches);}matches.add(right);
    for(let i=2;i<left.length;i++)pairs.push([left[i] as JsonExpression,right[i] as JsonExpression]);
  }
  return true;
}
/** Exact structural IDs, not source strings or collision-prone numeric hashes.
 * A key contains only the node tag/type and canonical child IDs. */
class StructuralIndex {
  private ids=new WeakMap<object,number>();
  private shapes=new Map<string,number>();
  private features=new WeakMap<object,ReadonlySet<number>>();
  id(node:JsonExpression):number {
    const hit=this.ids.get(node);if(hit!==undefined)return hit;
    const key=node[0]==='constant'||node[0]==='input'?`${node[0]}:${node[1]}:${node[2]}`:
      `${node[0]}:${node[1]}:`+(node.slice(2) as JsonExpression[]).map(x=>this.id(x)).join(',');
    let id=this.shapes.get(key);if(id===undefined){id=this.shapes.size;this.shapes.set(key,id);}
    this.ids.set(node,id);return id;
  }
  comparisonFeatures(node:JsonExpression):ReadonlySet<number> {
    const hit=this.features.get(node);if(hit)return hit;
    const result=new Set<number>();
    if(node[0]!=='constant'&&node[0]!=='input'){
      for(const child of node.slice(2) as JsonExpression[])for(const id of this.comparisonFeatures(child))result.add(id);
    }
    if(node[1]==='bool'){
      result.add(this.id(node));
      if(node[0]==='lt'||node[0]==='le'||node[0]==='eq')result.add(this.id(node[2]!));
      if(['lt','le','eq'].includes(node[0])&&jsonInteger((node[2] as JsonExpression)[1])){
        const stack:JsonExpression[]=[node[2] as JsonExpression],seen=new WeakSet<object>();
        while(stack.length){
          const operand=stack.pop()!;if(seen.has(operand))continue;seen.add(operand);result.add(this.id(operand));
          if(operand[0]!=='input'&&operand[0]!=='constant')
            for(const arg of operand.slice(2) as JsonExpression[])if(jsonInteger(arg[1]))stack.push(arg);
        }
      }
    }
    this.features.set(node,result);return result;
  }
  relevant(node:JsonExpression,facts:readonly (readonly [JsonExpression,boolean])[]){
    const features=this.comparisonFeatures(node);
    return facts.filter(([fact])=>features.has(this.id(fact))||
      ['lt','le','eq'].includes(fact[0])&&features.has(this.id(fact[2] as JsonExpression)));
  }
}
export function jsonExpressionIsTotal(node:JsonExpression,memo=new WeakMap<object,boolean>()):boolean {
  const hit=memo.get(node);if(hit!==undefined)return hit;
  if(node[0]==='constant'||node[0]==='input')return true;
  const args=node.slice(2) as JsonExpression[];
  let result=args.every(x=>jsonExpressionIsTotal(x,memo));
  if(node[0].startsWith('pending-')||node[0]==='reinterpret'&&args[0]![1]==='f16')result=false;
  if(result&&jsonInteger(node[1])&&node[0]==='div')result=args[1]![0]==='constant'&&jsonConstantValue(args[1]!)!==0n;
  if(result&&(node[0]==='shl'||node[0]==='shr'))result=args[1]![0]==='constant'&&
    (jsonConstantValue(args[1]!) as bigint)<BigInt(jsonWidths[node[1]]);
  memo.set(node,result);return result;
}
function proveIntegerComparison(node:JsonExpression,facts:readonly (readonly [JsonExpression,boolean])[],maxNodes:number,
  total:(expression:JsonExpression)=>boolean):boolean|undefined {
  if(!['lt','le','eq'].includes(node[0]))return undefined;
  const args=node.slice(2) as JsonExpression[],left=args[0]!,right=args[1]!;
  if(!jsonInteger(left[1])||right[0]!=='constant'||!total(left))return undefined;
  const interval=jsonIntegerRange(left,facts,sameJsonExpression,maxNodes);if(!interval)return undefined;
  const [lower,upper]=interval;
  const value=jsonConstantValue(right) as bigint;
  if(node[0]==='lt')return upper<value?true:lower>=value?false:undefined;
  if(node[0]==='le')return upper<=value?true:lower>value?false:undefined;
  return lower===upper&&lower===value?true:value<lower||value>upper?false:undefined;
}
export function simplifyJsonExpression(root:JsonExpression,stats=newJsonSimplificationStats(),
  initialFacts:readonly (readonly [JsonExpression,boolean])[]=[],maxVisits=1_000_000):JsonExpression {
  // Totality depends on the immutable expression, not on its condition scope.
  // Reuse its proof within this pass without retaining facts across rewrites.
  const totalMemo=new WeakMap<object,boolean>(),total=(node:JsonExpression)=>jsonExpressionIsTotal(node,totalMemo);
  if(!Number.isSafeInteger(maxVisits)||maxVisits<1)throw new RangeError('Invalid simplification visit budget');
  for(const [condition,truth] of initialFacts)if(condition[1]!=='bool'||typeof truth!=='boolean')
    throw new TypeError('Invalid compilation condition fact');
  const memo=new WeakMap<object,Map<string,JsonExpression>>(),index=new StructuralIndex();
  let visits=0;
  function visit(node:JsonExpression,facts:readonly (readonly [JsonExpression,boolean])[]):JsonExpression {
    if(++visits>maxVisits)throw new RangeError('Simplification visit budget exceeded');
    stats.visited++;
    if(node[0]==='constant'||node[0]==='input'){
      const known=node[0]==='input'&&node[1]==='bool'?facts.find(([condition])=>sameJsonExpression(condition,node)):undefined;
      if(known){stats.conditions++;return jsonConstant('bool',known[1]);}
      return node;
    }
    // Outer decisions on later computations do not influence an earlier pure
    // expression unless a reachable comparison actually consumes those facts.
    // Dropping irrelevant scope prevents a Cartesian expansion during rewriting.
    facts=index.relevant(node,facts);
    const known=facts.find(([condition])=>sameJsonExpression(condition,node));
    if(known){stats.conditions++;return jsonConstant('bool',known[1]);}
    const context=facts.map(([fact,truth])=>`${index.id(fact)}:${truth?1:0}`).join(',');
    const hit=memo.get(node)?.get(context);if(hit)return hit;
    const implied=proveIntegerComparison(node,facts,maxVisits,total);
    if(implied!==undefined){
      stats.conditions++;const result=jsonConstant('bool',implied);
      let scoped=memo.get(node);if(!scoped){scoped=new Map();memo.set(node,scoped);}
      scoped.set(context,result);return result;
    }
    const op=node[0],type=node[1],raw=node.slice(2) as JsonExpression[];
    if(op==='if'){
      const condition=visit(raw[0]!,facts);
      if(condition[0]==='constant'){stats.conditions++;return visit(raw[jsonConstantValue(condition)?1:2]!,facts);}
      const yes=visit(raw[1]!,[...facts,[condition,true]]),no=visit(raw[2]!,[...facts,[condition,false]]);
      if(sameJsonExpression(yes,no)&&total(condition)){stats.conditions++;return yes;}
      const result=jsonOperation('if',type,condition,yes,no);
      let scoped=memo.get(node);if(!scoped){scoped=new Map();memo.set(node,scoped);}scoped.set(context,result);
      return result;
    }
    const args=raw.map(x=>visit(x,facts));
    let result=jsonOperation(op,type,...args);
    if(op==='reinterpret'&&args[0]![0]==='reinterpret'&&args[0]![2]![1]===type){
      stats.folds++;return args[0]![2]!;
    }
    if(!op.startsWith('pending-')&&args.every(x=>x[0]==='constant')){
      // Preserve reinterpretation bit strings, including NaN payloads. Diagnostic
      // floating execution cannot certify all payload-preserving rewrites.
      if(op==='reinterpret'&&args[0]![0]==='constant'){
        result=['constant',type,args[0]![2]];stats.folds++;
      }else if(type!=='f16'){
        try{const value=evaluateJsonExpression(result);
          if(typeof value!=='number'||Number.isFinite(value)){result=jsonConstant(type,value);stats.folds++;}
        }catch{
          // Undefined integer operations remain visible and are never removed.
        }
      }
    }else if(jsonInteger(type)){
      const [a,b]=args;
      const is=(x:JsonExpression|undefined,n:bigint)=>x?.[0]==='constant'&&jsonConstantValue(x)===n;
      if((op==='add'||op==='or'||op==='xor')&&is(b,0n)||op==='mul'&&is(b,1n)||
        (op==='shl'||op==='shr')&&is(b,0n)||op==='div'&&is(b,1n))result=a!;
      else if((op==='add'||op==='or'||op==='xor')&&is(a,0n)||op==='mul'&&is(a,1n))result=b!;
      else if((op==='mul'||op==='and')&&((is(a,0n)&&total(b!))||(is(b,0n)&&total(a!))))result=jsonConstant(type,0n);
      else if((op==='sub'||op==='xor')&&sameJsonExpression(a!,b!)&&total(a!))result=jsonConstant(type,0n);
      else if((op==='and'||op==='or')&&sameJsonExpression(a!,b!))result=a!;
      else if(op==='add'&&a?.[0]==='mul'&&b?.[0]==='mul'&&sameJsonExpression(a[2]!,b[2]!)){
        // Unsigned modular arithmetic permits factoring without float rounding.
        result=jsonOperation('mul',type,a[2]!,visit(jsonOperation('add',type,a[3]!,b[3]!),facts));
      }
      // Bitword identities remove conversion scaffolding without touching
      // float arithmetic. Evaluations that can fail remain visible.
      const integer=(x:JsonExpression|undefined):bigint|undefined=>
        x?.[0]==='constant'&&jsonInteger(x[1])?jsonConstantValue(x) as bigint:undefined;
      const maskOperand=(x:JsonExpression|undefined,tag:string):[JsonExpression,bigint]|undefined=>{
        if(x?.[0]!==tag)return undefined;
        const a=x[2] as JsonExpression,b=x[3] as JsonExpression;
        const left=integer(a),right=integer(b);
        return right!==undefined?[a,right]:left!==undefined?[b,left]:undefined;
      };
      const all=(1n<<BigInt(jsonWidths[type]))-1n;
      let source=a,constant=integer(b);
      if(['and','or','xor'].includes(op)&&constant===undefined&&integer(a)!==undefined){source=b;constant=integer(a);}
      if(result[0]===op&&jsonInteger(result[1])){
        if(op==='and'&&constant===all)result=source!;
        else if((op==='and'||op==='or'||op==='xor')&&constant!==undefined){
          const nested=maskOperand(source,op);
          if(nested)result=jsonOperation(op,type,nested[0],jsonConstant(type,
            op==='and'?nested[1]&constant:op==='or'?nested[1]|constant:nested[1]^constant));
          else if(op==='and'&&source?.[0]==='shr'){
            const masked=maskOperand(source[2],'and'),shift=integer(source[3]);
            if(masked&&shift!==undefined&&shift<BigInt(jsonWidths[type])&&
              ((masked[1]>>shift)&constant)===constant)
              result=jsonOperation('and',type,jsonOperation('shr',type,masked[0],source[3]!),jsonConstant(type,constant));
          }
        }else if(op==='or'||op==='xor'){
          const left=maskOperand(a,'and'),right=maskOperand(b,'and');
          if(left&&right&&sameJsonExpression(left[0],right[0])){
            const combined=op==='or'?left[1]|right[1]:left[1]^right[1];
            if(combined!==0n||total(left[0]))result=jsonOperation('and',type,left[0],jsonConstant(type,combined));
          }
        }else if((op==='shr'||op==='shl')&&a?.[0]===op){
          const first=integer(a[3]),second=integer(b);
          if(first!==undefined&&second!==undefined&&first+second<BigInt(jsonWidths[type]))
            result=jsonOperation(op,type,a[2]!,jsonConstant(type,first+second));
        }
      }
      if(result!==undefined&&!sameJsonExpression(result,jsonOperation(op,type,...args)))stats.integerAlgebra++;
    }
    let scoped=memo.get(node);if(!scoped){scoped=new Map();memo.set(node,scoped);}scoped.set(context,result);
    return result;
  }
  return visit(root,initialFacts);
}
/** Reach a structural fixed point before any condition-product expansion. A
 * rule set must terminate within the declared budget, not silently stop early. */
export function simplifyJsonFixedPoint(root:JsonExpression,maxPasses=32,maxVisits=1_000_000):{
  expression:JsonExpression;passes:number;stats:JsonSimplificationStats
} {
  if(!Number.isSafeInteger(maxPasses)||maxPasses<1)throw new RangeError('Invalid simplification pass budget');
  const stats=newJsonSimplificationStats();let current=root;
  for(let passes=1;passes<=maxPasses;passes++){
    const next=simplifyJsonExpression(current,stats,[],maxVisits);
    if(sameJsonExpression(current,next))return {expression:next,passes,stats};
    current=next;
  }
  throw new Error('JSON simplification has not reached a fixed point');
}
