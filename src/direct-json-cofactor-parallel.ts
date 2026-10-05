import {Worker} from 'node:worker_threads';
import {availableParallelism} from 'node:os';
import type {JsonExpression} from './direct-json-expression.js';
import {measureJsonExpression} from './direct-json-measure.js';
import {simplifyJsonFixedPoint,jsonExpressionIsTotal} from './direct-json-simplify.js';
import {simplifyJsonSharedConditions,type JsonCofactorStats} from './direct-json-cofactor.js';
import {shareJsonExpression} from './direct-json-share.js';
export interface JsonParallelConditionOptions {maxCandidates?:number;maxUniqueNodes?:number;maxRounds?:number;workers?:number}
interface Reply {best?:JsonExpression;bestIndex:number;bestBytes:bigint;reducingCandidates:number;overBudget:number;fatal?:string}
interface Condition {index:number;condition:JsonExpression}

async function round(root:JsonExpression,options:JsonParallelConditionOptions){
  const maxCandidates=options.maxCandidates??256,maxNodes=options.maxUniqueNodes??100000;
  if(!Number.isSafeInteger(maxCandidates)||maxCandidates<0||!Number.isSafeInteger(maxNodes)||maxNodes<1)
    throw new RangeError('Invalid shared-condition simplification budget');
  measureJsonExpression(root,maxNodes);
  const maxVisits=Math.min(1000000,maxNodes*8),
    base=shareJsonExpression(simplifyJsonFixedPoint(root,32,maxVisits).expression,maxVisits).expression;
  const before=measureJsonExpression(base,maxNodes);
  const stats:JsonCofactorStats={candidates:0,testedCandidates:0,reducingCandidates:0,accepted:0,unsafe:0,overBudget:0,
    beforeBytes:before.serializedBytes,afterBytes:before.serializedBytes,rounds:1,roundBudgetFailures:0,
    converged:false,stopReason:'round-budget'};
  const order:JsonExpression[]=[],seen=new WeakSet<object>(),allInputs=new WeakMap<object,Set<string>>(),strictInputs=new WeakMap<object,Set<string>>();
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
  const weight=new WeakMap<object,bigint>();weight.set(base,1n);const counts=new Map<JsonExpression,bigint>();
  for(const node of order.reverse()){
    const count=weight.get(node)??0n;
    if(node[0]==='if')counts.set(node[2]!,count+(counts.get(node[2]!)??0n));
    if(node[0]!=='input'&&node[0]!=='constant')for(const arg of node.slice(2) as JsonExpression[])
      weight.set(arg,(weight.get(arg)??0n)+count);
  }
  const candidates=[...counts].filter(([,count])=>count>1n).sort((a,b)=>a[1]>b[1]?-1:a[1]<b[1]?1:0),safe:Condition[]=[];
  let exhausted=false;
  const totalMemo=new WeakMap<object,boolean>();
  for(const [condition] of candidates){
    const ordinal=stats.candidates;
    if([...allInputs.get(condition)!].some(x=>!strictInputs.get(base)!.has(x))||!jsonExpressionIsTotal(condition,totalMemo)){
      stats.candidates++;stats.unsafe++;
    }else{
      if(safe.length>=maxCandidates){exhausted=true;break;}
      stats.candidates++;stats.testedCandidates++;safe.push({index:ordinal,condition});
    }
  }
  const workerCount=Math.min(options.workers??availableParallelism(),availableParallelism(),safe.length),workers:Worker[]=[];
  const groups=Array.from({length:workerCount},()=>[] as Condition[]);
  for(let i=0;i<safe.length;i++)groups[i%workerCount]!.push(safe[i]!);
  let best=base,bestIndex=Infinity;
  try{
    const replies=await Promise.all(groups.map(conditions=>new Promise<Reply>((resolve,reject)=>{
      const worker=new Worker(new URL('./direct-json-cofactor-worker.js',import.meta.url),{
        workerData:{base,conditions,maxNodes,maxVisits,beforeBytes:before.serializedBytes},execArgv:[]});workers.push(worker);
      let delivered=false;
      worker.once('message',(reply:Reply)=>{delivered=true;reply.fatal!==undefined?reject(new Error(reply.fatal)):resolve(reply);});
      worker.once('error',reject);worker.once('exit',code=>{if(!delivered)reject(new Error(`Condition worker exited without a result (${code})`));});
    })));
    for(const reply of replies){
      stats.reducingCandidates+=reply.reducingCandidates;stats.overBudget+=reply.overBudget;
      // Match serial traversal on equal sizes; scheduling never changes choice.
      if(reply.best&&(reply.bestBytes<stats.afterBytes||reply.bestBytes===stats.afterBytes&&reply.bestIndex<bestIndex)){
        best=reply.best;bestIndex=reply.bestIndex;stats.afterBytes=reply.bestBytes;stats.accepted=1;
      }
    }
  }finally{await Promise.allSettled(workers.map(worker=>worker.terminate()));}
  if(stats.accepted===0){
    stats.converged=stats.overBudget===0&&!exhausted;
    stats.stopReason=stats.overBudget>0?'resource-budget':stats.converged?'fixed-point':'candidate-budget';
  }
  return {expression:best,stats};
}
/** Bounded CPU workers test independent candidates, never distribute arithmetic
 * reductions or alter path order. The output remains one closed scalar tree. */
export async function simplifyJsonSharedConditionsParallel(root:JsonExpression,options:JsonParallelConditionOptions={}){
  const workers=options.workers??availableParallelism(),maxRounds=options.maxRounds??8;
  if(!Number.isSafeInteger(workers)||workers<1||!Number.isSafeInteger(maxRounds)||maxRounds<1)
    throw new RangeError('Invalid parallel condition worker or round budget');
  if(workers===1)return simplifyJsonSharedConditions(root,options);
  let expression=root,aggregate:JsonCofactorStats|undefined;
  for(let n=0;n<maxRounds;n++){
    let result:Awaited<ReturnType<typeof round>>;
    try{result=await round(expression,options);}
    catch(error){
      if(!(error instanceof RangeError)||!aggregate)throw error;
      aggregate.rounds++;aggregate.roundBudgetFailures++;aggregate.converged=false;aggregate.stopReason='resource-budget';
      return {expression,stats:aggregate};
    }
    expression=result.expression;
    if(!aggregate)aggregate={...result.stats};
    else{
      for(const key of ['candidates','testedCandidates','reducingCandidates','accepted','unsafe','overBudget','rounds','roundBudgetFailures'] as const)aggregate[key]+=result.stats[key];
      aggregate.afterBytes=result.stats.afterBytes;aggregate.converged=result.stats.converged;aggregate.stopReason=result.stats.stopReason;
    }
    if(result.stats.accepted===0)return {expression,stats:aggregate};
  }
  aggregate!.converged=false;aggregate!.stopReason='round-budget';return {expression,stats:aggregate!};
}
