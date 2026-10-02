import {jsonInteger,jsonWidths,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue} from './direct-json-evaluator.js';

export type JsonIntegerRange=readonly [bigint,bigint];
type Fact=readonly [JsonExpression,boolean];
/** Conservative unsigned intervals in a compiler condition scope. Modular
 * wrap is kept only when both endpoints lie in the same wrap cell. */
export function jsonIntegerRange(root:JsonExpression,facts:readonly Fact[],
  equal:(a:JsonExpression,b:JsonExpression)=>boolean,maxNodes=1_000_000):JsonIntegerRange|undefined {
  if(!jsonInteger(root[1]))return undefined;
  if(!Number.isSafeInteger(maxNodes)||maxNodes<1)throw new RangeError('Invalid integer interval budget');
  const memo=new WeakMap<object,JsonIntegerRange|undefined>(),seen=new WeakSet<object>(),active=new WeakSet<object>();
  const stack:{node:JsonExpression;finish:boolean}[]=[{node:root,finish:false}];let count=0;
  const full=(node:JsonExpression):JsonIntegerRange=>[0n,(1n<<BigInt(jsonWidths[node[1]]))-1n];
  function narrow(node:JsonExpression,range:JsonIntegerRange):JsonIntegerRange|undefined {
    let [lo,hi]=range;
    for(const [fact,truth] of facts){
      if(!['lt','le','eq'].includes(fact[0])||!equal(node,fact[2] as JsonExpression)||fact[3]?.[0]!=='constant'||!jsonInteger(fact[3][1]))continue;
      const n=jsonConstantValue(fact[3]) as bigint;
      if(fact[0]==='lt'){if(truth)hi=hi<n-1n?hi:n-1n;else lo=lo>n?lo:n;}
      else if(fact[0]==='le'){if(truth)hi=hi<n?hi:n;else lo=lo>n+1n?lo:n+1n;}
      else if(truth){lo=lo>n?lo:n;hi=hi<n?hi:n;}
      else {if(lo===n)lo++;if(hi===n)hi--;}
    }
    return lo<=hi?[lo,hi]:undefined;
  }
  function wrap(node:JsonExpression,lo:bigint,hi:bigint):JsonIntegerRange {
    const width=jsonWidths[node[1]],mod=1n<<BigInt(width);
    const cell=(n:bigint)=>n>=0n?n/mod:(n-mod+1n)/mod;
    return cell(lo)===cell(hi)?[BigInt.asUintN(width,lo),BigInt.asUintN(width,hi)]:full(node);
  }
  while(stack.length){
    const {node,finish}=stack.pop()!;
    if(seen.has(node))continue;
    if(!finish){
      if(active.has(node))throw new Error('Cyclic integer interval expression');
      if(++count>maxNodes)throw new RangeError('Integer interval node budget exceeded');
      active.add(node);stack.push({node,finish:true});
      if(node[0]!=='input'&&node[0]!=='constant')
        for(const arg of (node.slice(2) as JsonExpression[]).reverse())if(jsonInteger(arg[1]))stack.push({node:arg,finish:false});
    }else{
      let range=full(node);
      if(node[0]==='constant'){const n=jsonConstantValue(node) as bigint;range=[n,n];}
      else if(node[0]!=='input'){
        const args=node.slice(2) as JsonExpression[],a=memo.get(args[0]!),b=args[1]&&memo.get(args[1]);
        if(node[0]==='reinterpret'&&args[0]![0]==='constant'){
          const n=BigInt(args[0]![2] as string);range=[n,n];
        }else if(node[0]==='convert'&&a)range=wrap(node,a[0],a[1]);
        else if(a&&b){
          if(node[0]==='add')range=wrap(node,a[0]+b[0],a[1]+b[1]);
          else if(node[0]==='sub')range=wrap(node,a[0]-b[1],a[1]-b[0]);
          else if(node[0]==='mul')range=wrap(node,a[0]*b[0],a[1]*b[1]);
          else if(node[0]==='div'&&b[0]>0n)range=[a[0]/b[1],a[1]/b[0]];
          else if(node[0]==='and')range=[0n,a[1]<b[1]?a[1]:b[1]];
          else if(node[0]==='or'||node[0]==='xor'){
            const hi=a[1]>b[1]?a[1]:b[1],upper=hi===0n?0n:(1n<<BigInt(hi.toString(2).length))-1n;
            range=[node[0]==='or'?(a[0]>b[0]?a[0]:b[0]):0n,upper];
          }else if((node[0]==='shr'||node[0]==='shl')&&b[0]===b[1]&&b[1]<BigInt(jsonWidths[node[1]]))
            range=node[0]==='shr'?[a[0]>>b[0],a[1]>>b[0]]:wrap(node,a[0]<<b[0],a[1]<<b[0]);
        }
        if(node[0]==='if'){
          const known=args[0]![0]==='constant'?jsonConstantValue(args[0]!) as boolean:
            facts.find(([fact])=>equal(fact,args[0]!))?.[1];
          const yes=memo.get(args[1]!),no=memo.get(args[2]!);
          if(known!==undefined)range=(known?yes:no)??range;
          else if(yes&&no)range=[yes[0]<no[0]?yes[0]:no[0],yes[1]>no[1]?yes[1]:no[1]];
        }
      }
      memo.set(node,narrow(node,range));seen.add(node);active.delete(node);
    }
  }
  return memo.get(root);
}
