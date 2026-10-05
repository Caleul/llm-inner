import {jsonConstant,jsonInteger,jsonOperation,jsonWidths,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue} from './direct-json-evaluator.js';
import {measureJsonExpression} from './direct-json-measure.js';

/** Global sum collection in Z/(2^width), never real-number float algebra.
 * Nonlinear/bitwise/control subtrees remain atoms. Structural IDs and totality
 * come from the caller's existing proof index; no strings encode expressions.
 * Search limits decline a rewrite, without truncating the original expression. */
export function factorJsonIntegerSum(root:JsonExpression,id:(node:JsonExpression)=>number,
  total:(node:JsonExpression)=>boolean):JsonExpression {
  const type=root[1];
  if(!jsonInteger(type)||!['add','sub'].includes(root[0]))return root;
  const width=jsonWidths[type],wrap=(value:bigint)=>BigInt.asUintN(width,value);
  const terms=new Map<number,{node:JsonExpression;coefficient:bigint}>();
  const pending:[JsonExpression,bigint][]=[[root,1n]];let constant=0n,visits=0;
  while(pending.length){
    if(++visits>256)return root;
    const [node,coefficient]=pending.pop()!;
    if(node[1]!==type)return root;
    if(node[0]==='add'||node[0]==='sub'){
      pending.push([node[3]!,node[0]==='add'?coefficient:-coefficient],[node[2]!,coefficient]);
    }else if(node[0]==='constant')constant=wrap(constant+coefficient*(jsonConstantValue(node) as bigint));
    else if(node[0]==='mul'&&(node[2]![0]==='constant'||node[3]![0]==='constant')){
      const left=node[2]![0]==='constant';
      pending.push([node[left?3:2]!,wrap(coefficient*(jsonConstantValue(node[left?2:3]!) as bigint))]);
    }else{
      // Removing/reordering evaluation is legal only for total scalar atoms.
      if(!total(node))return root;
      const key=id(node),previous=terms.get(key);
      if(previous)previous.coefficient=wrap(previous.coefficient+coefficient);
      else terms.set(key,{node,coefficient:wrap(coefficient)});
    }
  }
  const live=[...terms.values()].filter(term=>term.coefficient!==0n);
  const gcd=(a:bigint,b:bigint):bigint=>{while(b){[a,b]=[b,a%b];}return a;};
  let factor=constant;
  for(const term of live)factor=gcd(factor,term.coefficient);
  if(factor===0n)factor=1n;
  const pieces=live.map(({node,coefficient})=>coefficient/factor===1n?node:
    jsonOperation('mul',type,jsonConstant(type,coefficient/factor),node));
  if(constant!==0n)pieces.push(jsonConstant(type,constant/factor));
  let candidate=pieces[0]??jsonConstant(type,0n);
  for(const piece of pieces.slice(1))candidate=jsonOperation('add',type,candidate,piece);
  if(factor!==1n)candidate=jsonOperation('mul',type,jsonConstant(type,factor),candidate);
  if(id(candidate)===id(root))return root;
  try{
    // Count copies mathematically from shared compile-time syntax instead of
    // opening exponentially repeated atoms just to compare their sizes.
    const before=measureJsonExpression(root,4096),after=measureJsonExpression(candidate,4096);
    return after.serializedBytes<before.serializedBytes?candidate:root;
  }catch(error){
    if(error instanceof RangeError)return root;
    throw error;
  }
}
