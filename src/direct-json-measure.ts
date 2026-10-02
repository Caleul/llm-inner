import {validateJsonNode,type JsonExpression} from './direct-json-expression.js';

export interface JsonExpressionMeasure {
  uniqueNodes:number;uniqueDecisions:number;occurrences:bigint;decisions:bigint;
  inputReferences:bigint;serializedBytes:bigint;depth:number;
}
/** Compute expanded size from compiler sharing without expanding or rendering
 * it. Exponential duplication is reported as a BigInt instead of exhausting RAM. */
export function measureJsonExpression(root:JsonExpression,maxUniqueNodes=1_000_000,maxDepth=4096):JsonExpressionMeasure {
  if(!Number.isSafeInteger(maxUniqueNodes)||maxUniqueNodes<1)throw new RangeError('Invalid measurement budget');
  if(!Number.isSafeInteger(maxDepth)||maxDepth<1)throw new RangeError('Invalid measurement depth budget');
  const memo=new WeakMap<object,Omit<JsonExpressionMeasure,'uniqueNodes'|'uniqueDecisions'>>();
  const active=new WeakSet<object>();let uniqueNodes=0,uniqueDecisions=0;
  const stack:{node:JsonExpression;finish:boolean}[]=[{node:root,finish:false}];
  while(stack.length){
    const {node,finish}=stack.pop()!;
    if(memo.has(node))continue;
    if(!finish){
      validateJsonNode(node);
      if(active.has(node))throw new Error('Cyclic expression cannot be measured');
      if(++uniqueNodes>maxUniqueNodes)throw new RangeError('Unique node measurement budget exceeded');
      if(node[0]==='if')uniqueDecisions++;
      active.add(node);stack.push({node,finish:true});
      if(node[0]!=='input'&&node[0]!=='constant')for(const arg of (node.slice(2) as JsonExpression[]).reverse())
        stack.push({node:arg,finish:false});
    }else{
      let occurrences=1n,decisions=node[0]==='if'?1n:0n,depth=1;
      let inputReferences=node[0]==='input'?1n:0n;
      let serializedBytes=BigInt(2+Buffer.byteLength(JSON.stringify(node[0]))+1+Buffer.byteLength(JSON.stringify(node[1])));
      if(node[0]==='input'||node[0]==='constant')serializedBytes+=BigInt(1+Buffer.byteLength(JSON.stringify(node[2])));
      else for(const arg of node.slice(2) as JsonExpression[]){
        const value=memo.get(arg)!;
        occurrences+=value.occurrences;decisions+=value.decisions;inputReferences+=value.inputReferences;serializedBytes+=1n+value.serializedBytes;
        depth=Math.max(depth,value.depth+1);
      }
      if(depth>maxDepth)throw new RangeError('Measurement depth budget exceeded');
      memo.set(node,{occurrences,decisions,inputReferences,serializedBytes,depth});active.delete(node);
    }
  }
  return {uniqueNodes,uniqueDecisions,...memo.get(root)!};
}
