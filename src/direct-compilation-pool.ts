import { Worker } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { setInterval, clearInterval } from 'node:timers';
import type { DirectCompilationUnit } from './direct-compilation-units.js';

export interface DirectPoolOptions {
  workers?: number;
  memoryMiB?: number;
  heapMiB?: number;
  weightCacheMiB?: number;
  partitions?: number;
  partitionCoordinate?: number;
  maxOutputMiB?: number;
  signal?: AbortSignal;
}
export interface DirectPoolBudget {
  workers:number; memoryMiB:number; heapMiB:number; weightCacheMiB:number;
  partitions:number; partitionCoordinate:number; maxOutputBytes:number;
}
export function directPoolBudget(options:DirectPoolOptions={}):DirectPoolBudget {
  const memoryMiB=options.memoryMiB??2048,heapMiB=options.heapMiB??256,
    weightCacheMiB=options.weightCacheMiB??16,requested=options.workers??Math.min(4,availableParallelism()),
    partitions=options.partitions??4,partitionCoordinate=options.partitionCoordinate??0,
    maxOutputMiB=options.maxOutputMiB??8192;
  for(const [name,value] of Object.entries({memoryMiB,heapMiB,requested,partitions,maxOutputMiB}))
    if(!Number.isSafeInteger(value)||value<1)throw new RangeError(`Invalid ${name}`);
  for(const [name,value] of Object.entries({weightCacheMiB,partitionCoordinate}))
    if(!Number.isSafeInteger(value)||value<0)throw new RangeError(`Invalid ${name}`);
  const perWorker=heapMiB+weightCacheMiB+128;
  const capacity=Math.floor((memoryMiB-128)/perWorker);
  if(capacity<1)throw new RangeError('Memory budget cannot admit one worker');
  const maxOutputBytes=maxOutputMiB*1024*1024;
  if(!Number.isSafeInteger(maxOutputBytes))throw new RangeError('Output budget exceeds exact integer domain');
  return {workers:Math.min(requested,capacity),memoryMiB,heapMiB,weightCacheMiB,
    partitions,partitionCoordinate,maxOutputBytes};
}
export interface DirectUnitJob { id:number; unit:DirectCompilationUnit; path:string }
export interface DirectUnitResult { id:number; path:string; sha256:string; bytes:number;
  leaves:number; candidateLeaves:number; eliminatedBranches:number; inspectedExpressions:number;
  weightReads:number; weightPageHits:number }

/** Dynamic bounded queue. Only task descriptors cross worker boundaries;
 * workers retain no shared emitters, closures, path facts or calculations.
 */
export async function runDirectCompilationPool(workerURL:URL,workerData:unknown,
  jobs:Iterable<DirectUnitJob>,budget:DirectPoolBudget,signal?:AbortSignal,
  completed?:(result:DirectUnitResult)=>void):Promise<DirectUnitResult[]> {
  if(signal?.aborted)throw new Error('Compilation aborted');
  const iterator=jobs[Symbol.iterator](),workers:Worker[]=[],results:DirectUnitResult[]=[];
  const ids=new Set<number>();let done=false,pending=0,exhausted=false;
  let resolve!: (result:DirectUnitResult[])=>void,reject!:(reason:unknown)=>void;
  const completion=new Promise<DirectUnitResult[]>((ok,no)=>{resolve=ok;reject=no;});
  const fail=(error:unknown)=>{if(!done){done=true;reject(error);}};
  const abort=()=>fail(new Error('Compilation aborted'));
  signal?.addEventListener('abort',abort,{once:true});
  const monitor=setInterval(()=>{
    // Worker isolates share one process: RSS includes all of them and native
    // allocations excluded by V8 resourceLimits. Periodic check is not an OS quota.
    if(process.memoryUsage().rss>budget.memoryMiB*1024*1024)
      fail(new Error('Compilation memory budget exceeded'));
  },1000);monitor.unref();
  const assigned=new Map<Worker,DirectUnitJob>();
  const dispatch=(worker:Worker)=>{
    if(done)return;
    const next=iterator.next();
    if(next.done){exhausted=true;if(pending===0){done=true;resolve(results.sort((a,b)=>a.id-b.id));}return;}
    const job=next.value;
    if(!Number.isSafeInteger(job.id)||job.id<0||ids.has(job.id))throw new Error('Duplicate or invalid compilation job');
    ids.add(job.id);assigned.set(worker,job);pending++;
    worker.postMessage(job);
  };
  try {
    for(let i=0;i<budget.workers&&!done;i++){
      const worker=new Worker(workerURL,{workerData,resourceLimits:{
        maxOldGenerationSizeMb:budget.heapMiB,maxYoungGenerationSizeMb:16}});
      workers.push(worker);
      worker.on('message',(message:{result?:DirectUnitResult;error?:string})=>{
        if(done)return;
        try{
          if(message.error)throw new Error(message.error);
          const job=assigned.get(worker),result=message.result;
          if(!job||!result||result.id!==job.id||result.path!==job.path)throw new Error('Unexpected compilation result');
          assigned.delete(worker);pending--;results.push(result);completed?.(result);dispatch(worker);
        }catch(error){fail(error);}
      });
      worker.on('error',fail);
      worker.on('exit',code=>{if(!done&&(code!==0||assigned.has(worker)))fail(new Error(`Compilation worker exited (${code})`));});
      dispatch(worker);
      if(exhausted)break;
    }
    return await completion;
  }finally{
    done=true;clearInterval(monitor);signal?.removeEventListener('abort',abort);
    await Promise.all(workers.map(worker=>worker.terminate()));
  }
}
