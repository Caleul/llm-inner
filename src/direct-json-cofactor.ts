import {jsonOperation,type JsonExpression} from './direct-json-expression.js';
import {measureJsonExpression} from './direct-json-measure.js';
import {simplifyJsonFixedPoint,simplifyJsonExpression,jsonExpressionIsTotal} from './direct-json-simplify.js';
import {shareJsonExpression} from './direct-json-share.js';

export interface JsonCofactorStats {candidates:number;testedCandidates:number;reducingCandidates:number;accepted:number;unsafe:number;overBudget:number;
  beforeBytes:bigint;afterBytes:bigint;rounds:number;converged:boolean;roundBudgetFailures:number;
  stopReason:'fixed-point'|'round-budget'|'candidate-budget'|'resource-budget'}
/** Select one existing pure decision, propagate its truth into the whole scalar,
 * and keep the decision once outside both simplified results. This combines
 * repeated condition queries without reordering arithmetic within a path.
 * Only byte-reducing candidates are accepted, and no Cartesian condition
 * product is generated in advance. All base rules reach a fixed point first. */
function simplifyOneSharedCondition(root:JsonExpression,
  options:{maxCandidates?:number;maxUniqueNodes?:number}={}):{expression:JsonExpression;stats:JsonCofactorStats} {
  const maxCandidates=options.maxCandidates??256,maxNodes=options.maxUniqueNodes??100_000;
  if(!Number.isSafeInteger(maxCandidates)||maxCandidates<0||!Number.isSafeInteger(maxNodes)||maxNodes<1)
    throw new RangeError('Invalid shared-condition simplification budget');
  measureJsonExpression(root,maxNodes);
  const maxVisits=Math.min(1_000_000,maxNodes*8);
  const base=shareJsonExpression(simplifyJsonFixedPoint(root,32,maxVisits).expression,maxVisits).expression,
    before=measureJsonExpression(base,maxNodes);
  const stats:JsonCofactorStats={candidates:0,testedCandidates:0,reducingCandidates:0,accepted:0,unsafe:0,overBudget:0,beforeBytes:before.serializedBytes,afterBytes:before.serializedBytes,rounds:1,converged:false,roundBudgetFailures:0,stopReason:'round-budget'};
  // Postorder compiler nodes, never expanded occurrences. Weight each node by
  // its expanded multiplicity when ranking duplicated condition queries.
  const order:JsonExpression[]=[],seen=new WeakSet<object>();
  const allInputs=new WeakMap<object,Set<string>>(),strictInputs=new WeakMap<object,Set<string>>();
  function index(node:JsonExpression){
    if(seen.has(node))return;seen.add(node);
    const all=new Set<string>(),strict=new Set<string>();
    if(node[0]==='input'){all.add(node[2]);strict.add(node[2]);}
    else if(node[0]!=='constant'){
      const args=node.slice(2) as JsonExpression[];for(const arg of args)index(arg);
      for(const arg of args)for(const name of allInputs.get(arg)!)all.add(name);
      if(node[0]==='if'){
        for(const name of strictInputs.get(args[0]!)!)strict.add(name);
        for(const name of strictInputs.get(args[1]!)!)if(strictInputs.get(args[2]!)!.has(name))strict.add(name);
      }else for(const arg of args)for(const name of strictInputs.get(arg)!)strict.add(name);
    }
    allInputs.set(node,all);strictInputs.set(node,strict);order.push(node);
  }
  index(base);
  const weight=new WeakMap<object,bigint>();weight.set(base,1n);
  const counts=new Map<JsonExpression,bigint>();
  for(const node of order.reverse()){
    const count=weight.get(node)??0n;
    if(node[0]==='if')counts.set(node[2]!,count+(counts.get(node[2]!)??0n));
    if(node[0]!=='input'&&node[0]!=='constant')for(const arg of node.slice(2) as JsonExpression[])
      weight.set(arg,(weight.get(arg)??0n)+count);
  }
  const candidates=[...counts].filter(([,count])=>count>1n).sort((a,b)=>a[1]>b[1]?-1:a[1]<b[1]?1:0);
  let best=base,tested=0,exhausted=false;
  const totalMemo=new WeakMap<object,boolean>();
  for(const [condition] of candidates){
    // Moving a decision must neither expose an undefined integer operation nor
    // require an input that the original lazy expression could leave absent.
    if([...allInputs.get(condition)!].some(x=>!strictInputs.get(base)!.has(x))||!jsonExpressionIsTotal(condition,totalMemo)){
      stats.candidates++;stats.unsafe++;continue;
    }
    // The budget bounds expensive cofactor transformations, not metadata
    // rejections. Unsafe length-specific decisions cannot starve safe ones.
    if(tested>=maxCandidates){exhausted=true;break;}
    tested++;stats.testedCandidates++;stats.candidates++;
    let candidate:JsonExpression,size;
    try{
      const yes=simplifyJsonFixedPoint(simplifyJsonExpression(base,undefined,[[condition,true]],maxVisits),32,maxVisits).expression;
      const no=simplifyJsonFixedPoint(simplifyJsonExpression(base,undefined,[[condition,false]],maxVisits),32,maxVisits).expression;
      candidate=shareJsonExpression(jsonOperation('if',base[1],condition,yes,no),maxVisits).expression;
      size=measureJsonExpression(candidate,maxNodes);
    }catch(error){
      if(error instanceof RangeError){stats.overBudget++;continue;}throw error;
    }
    if(size.serializedBytes<stats.beforeBytes)stats.reducingCandidates++;
    if(size.serializedBytes<stats.afterBytes){best=candidate;stats.afterBytes=size.serializedBytes;stats.accepted=1;}
  }
  if(stats.accepted===0){
    stats.converged=stats.overBudget===0&&!exhausted;
    stats.stopReason=stats.overBudget>0?'resource-budget':stats.converged?'fixed-point':'candidate-budget';
  }
  return {expression:best,stats};
}

/** Continue strictly byte-reducing promotions after each base-rule fixed point.
 * Budgets bound compiler work, not model semantics. An exhausted budget is
 * reported as incomplete; convergence applies only to this transformation. */
export function simplifyJsonSharedConditions(root:JsonExpression,
  options:{maxCandidates?:number;maxUniqueNodes?:number;maxRounds?:number}={}):{expression:JsonExpression;stats:JsonCofactorStats} {
  const maxRounds=options.maxRounds??8;
  if(!Number.isSafeInteger(maxRounds)||maxRounds<1)throw new RangeError('Invalid shared-condition round budget');
  let expression=root,aggregate:JsonCofactorStats|undefined;
  for(let round=0;round<maxRounds;round++){
    let result:ReturnType<typeof simplifyOneSharedCondition>;
    try{result=simplifyOneSharedCondition(expression,options);}
    catch(error){
      if(!(error instanceof RangeError)||!aggregate)throw error;
      // The previous accepted expression is already measured and valid. A
      // later base-rule budget cannot discard that progress or claim closure.
      aggregate.rounds++;aggregate.roundBudgetFailures++;aggregate.converged=false;
      aggregate.stopReason='resource-budget';return {expression,stats:aggregate};
    }
    expression=result.expression;
    if(!aggregate)aggregate={...result.stats};
    else {
      for(const key of ['candidates','testedCandidates','reducingCandidates','accepted','unsafe','overBudget','rounds','roundBudgetFailures'] as const)
        aggregate[key]+=result.stats[key];
      aggregate.afterBytes=result.stats.afterBytes;
      aggregate.converged=result.stats.converged;aggregate.stopReason=result.stats.stopReason;
    }
    if(result.stats.accepted===0)return {expression,stats:aggregate};
  }
  aggregate!.converged=false;aggregate!.stopReason='round-budget';
  return {expression,stats:aggregate!};
}
