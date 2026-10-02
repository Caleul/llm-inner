import {jsonOperation,validateJsonNode,type JsonExpression} from './direct-json-expression.js';

/** Compiler-only structural interning. This changes neither the expanded
 * expression nor its evaluation order. IDs and aliases are never serialized. */
export function shareJsonExpression(root:JsonExpression,maxNodes=1_000_000):{
  expression:JsonExpression;visited:number;uniqueNodes:number;mergedNodes:number
}{
  if(!Number.isSafeInteger(maxNodes)||maxNodes<1)throw new RangeError('Invalid structural sharing budget');
  interface Entry {expression:JsonExpression;id:number}
  const memo=new WeakMap<object,Entry>(),active=new WeakSet<object>(),shapes=new Map<string,Entry>();
  const stack:{node:JsonExpression;finish:boolean}[]=[{node:root,finish:false}];let visited=0;
  while(stack.length){
    const {node,finish}=stack.pop()!;
    if(memo.has(node))continue;
    if(!finish){
      validateJsonNode(node);
      if(active.has(node))throw new Error('Cyclic expression cannot be shared');
      if(++visited>maxNodes)throw new RangeError('Structural sharing node budget exceeded');
      active.add(node);stack.push({node,finish:true});
      if(node[0]!=='constant'&&node[0]!=='input')
        for(const arg of (node.slice(2) as JsonExpression[]).reverse())stack.push({node:arg,finish:false});
    }else{
      const leaf=node[0]==='constant'||node[0]==='input';
      const children=leaf?[]:(node.slice(2) as JsonExpression[]).map(arg=>memo.get(arg)!);
      // Only tags, exact leaf payloads and canonical child IDs enter the key.
      // No expanded expression is rendered, evaluated or algebraically changed.
      const key=JSON.stringify(leaf?node:[node[0],node[1],...children.map(arg=>arg.id)]);
      let entry=shapes.get(key);
      if(!entry){
        const expression=leaf||children.every((arg,i)=>arg.expression===node[i+2])?node:
          jsonOperation(node[0],node[1],...children.map(arg=>arg.expression));
        entry={expression,id:shapes.size};shapes.set(key,entry);
      }
      memo.set(node,entry);active.delete(node);
    }
  }
  return {expression:memo.get(root)!.expression,visited,uniqueNodes:shapes.size,mergedNodes:visited-shapes.size};
}
