import { mkdir, mkdtemp, open, readFile, rename, rm, stat, statfs, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { withDirectCompilationLease } from './direct-compilation-lease.js';
import { directCheckpointIdentity, verifyDirectCheckpointIdentity } from './direct-checkpoint-identity.js';
import { prepareDirectModel } from './direct-flat-rust-model.js';
import { directCompilationUnits, directRustHeader, directRustFooter } from './direct-compilation-units.js';
import { directPoolBudget, runDirectCompilationPool, type DirectPoolOptions, type DirectUnitJob,
  type DirectUnitResult } from './direct-compilation-pool.js';

/** Concatenate complete fragments in canonical order, checking integrity while
 * streaming. Does not parse/reassociate arithmetic or retain generated source.
 */
export async function assembleDirectFragments(path:string,header:string,footer:string,
  results:readonly DirectUnitResult[],maxBytes:number,signal?:AbortSignal):Promise<void>{
  const output=await open(path,'wx');let bytes=0;
  const write=async(chunk:string|Buffer)=>{
    if(signal?.aborted)throw new Error('Compilation aborted');
    const buffer=typeof chunk==='string'?Buffer.from(chunk):chunk;
    bytes+=buffer.length;if(bytes>maxBytes)throw new Error('Compilation output byte limit exceeded');
    let offset=0;while(offset<buffer.length){
      const {bytesWritten}=await output.write(buffer,offset,buffer.length-offset);
      if(!bytesWritten)throw new Error('Short fragment output write');offset+=bytesWritten;
    }
  };
  try{
    await write(header);
    for(let index=0;index<results.length;index++){
      const result=results[index]!;
      if(result.id!==index)throw new Error('Missing or reordered compilation fragment');
      const hash=createHash('sha256');let count=0;
      for await(const raw of createReadStream(result.path)){
        const chunk=raw as Buffer;hash.update(chunk);count+=chunk.length;await write(chunk);
      }
      if(count!==result.bytes||hash.digest('hex')!==result.sha256)throw new Error('Compilation fragment integrity mismatch');
    }
    await write(footer);await output.sync();
  }finally{await output.close();}
}
export async function writeParallelDirectFlatRustModel(directory:string,python:string,dimension:number,
  path:string,options:DirectPoolOptions={}):Promise<void>{
  // Validate before acquiring a publication lease.
  directPoolBudget(options);
  if(options.signal?.aborted)throw new Error("Compilation aborted");
  return withDirectCompilationLease(path,()=>compileParallelDirectFlatRustModel(directory,python,dimension,path,options));
}
async function compileParallelDirectFlatRustModel(directory:string,python:string,dimension:number,
  path:string,options:DirectPoolOptions):Promise<void>{
  const budget=directPoolBudget(options),started=Date.now(),target=resolve(path);
  if(options.signal?.aborted)throw new Error('Compilation aborted');
  // Discover once, before spawning: workers receive immutable source metadata,
  // not model tensors, a forward graph, expressions or activation values.
  const checkpointIdentity=await directCheckpointIdentity(directory);
  const discovered=await prepareDirectModel(directory,python,budget.weightCacheMiB*1024*1024),{output}=discovered;
  if(!Number.isSafeInteger(dimension)||dimension<0||dimension>=output.shape[0])throw new RangeError('Invalid logit dimension');
  // Validate the lazy task space before creating workers or touching the target.
  directCompilationUnits(output.maxPosition,output.shape[1],budget.partitions,budget.partitionCoordinate).next();
  await verifyDirectCheckpointIdentity(directory,checkpointIdentity);
  await mkdir(dirname(target),{recursive:true});
  const storage=await statfs(dirname(target),{bigint:true});
  if(storage.bavail*storage.bsize<BigInt(budget.maxOutputBytes)*2n)
    throw new Error('Insufficient disk space for fragments and assembled output; lower --max-output-mib');
  const temporary=await mkdtemp(join(dirname(target),'.direct-parallel-'));
  const controller=new AbortController(),interrupt=()=>controller.abort();
  options.signal?.addEventListener('abort',interrupt,{once:true});
  if(options.signal?.aborted)controller.abort();
  process.on('SIGINT',interrupt);process.on('SIGTERM',interrupt);
  const header=directRustHeader(output.shape[1],output.maxPosition),footer=directRustFooter;
  const counter=new SharedArrayBuffer(8);
  Atomics.store(new BigInt64Array(counter),0,BigInt(Buffer.byteLength(header)+Buffer.byteLength(footer)));
  let completedUnits=0,bytes=0,progressWrite=Promise.resolve(),progressError:unknown;
  const snapshot=()=>JSON.stringify({status:'generating',dimension,workers:budget.workers,partitions:budget.partitions,
    completedUnits,bytes,generatedBytes:Number(Atomics.load(new BigInt64Array(counter),0)),
    elapsedMilliseconds:Date.now()-started,finalParity:false,rustCompilationAdmitted:false},null,2)+'\n';
  const progress=setInterval(()=>{progressWrite=progressWrite.then(()=>writeFile(target+'.progress.json',snapshot())).catch(error=>{progressError=error;controller.abort();});},5000);
  progress.unref();
  function* jobs():Generator<DirectUnitJob>{let id=0;
    for(const unit of directCompilationUnits(output.maxPosition,output.shape[1],budget.partitions,budget.partitionCoordinate))
      yield {id,unit,path:join(temporary,`unit-${id++}.rs`)};
  }
  try{
    const results=await runDirectCompilationPool(new URL('./direct-compilation-worker.js',import.meta.url),
      {directory:resolve(directory),dimension,discovered,weightCacheBytes:budget.weightCacheMiB*1024*1024,
        maxOutputBytes:budget.maxOutputBytes,outputCounter:counter},jobs(),budget,controller.signal,
      result=>{completedUnits++;bytes+=result.bytes;});
    if(progressError)throw progressError;
    await verifyDirectCheckpointIdentity(directory,checkpointIdentity);
    const draft=join(temporary,'assembled.rs');
    await assembleDirectFragments(draft,header,footer,results,budget.maxOutputBytes,controller.signal);
    await verifyDirectCheckpointIdentity(directory,checkpointIdentity);
    const summary={checkpointIdentity,checkpointIdentityPolicy:'config/index SHA256; payload stat identity',status:'emitted',dimension,inputWidth:output.shape[1],context:output.maxPosition,
      bytes:(await stat(draft)).size,leaves:results.reduce((n,r)=>n+r.leaves,0),
      candidateLeaves:results.reduce((n,r)=>n+r.candidateLeaves,0),
      eliminatedBranches:results.reduce((n,r)=>n+r.eliminatedBranches,0),
      inspectedExpressions:results.reduce((n,r)=>n+r.inspectedExpressions,0),
      weightReads:results.reduce((n,r)=>n+r.weightReads,0),weightPageHits:results.reduce((n,r)=>n+r.weightPageHits,0),
      numericalPolicy:{torch:output.torch,backend:'cpu-arm64',weights:'finite-f16',
        rounding:'nearest-even F32/F16; ordered source reductions'},
      input:'finite-f16-embedding-matrix',runtimeIR:false,finalParity:false,
      parallel:{...budget,units:results.length},elapsedMilliseconds:Date.now()-started};
    await writeFile(join(temporary,'reduction.json'),JSON.stringify(summary,null,2)+'\n');
    if(controller.signal.aborted)throw new Error('Compilation aborted');
    // Source appears only after every fragment passed integrity. A sidecar is
    // explanatory; the CLI never accepts metadata from a failed invocation.
    clearInterval(progress);await progressWrite;
    if(progressError)throw progressError;
    await rename(draft,target);await rename(join(temporary,'reduction.json'),target+'.reduction.json');
    await writeFile(target+'.progress.json',JSON.stringify({...summary,rustCompilationAdmitted:true},null,2)+'\n');
  }catch(error){
    clearInterval(progress);await progressWrite;
    await writeFile(target+'.progress.json',JSON.stringify({status:'pending',dimension,completedUnits,bytes,
      reason:error instanceof Error?error.message:String(error),finalParity:false,rustCompilationAdmitted:false},null,2)+'\n');
    throw error;
  }finally{
    clearInterval(progress);process.off('SIGINT',interrupt);process.off('SIGTERM',interrupt);
    options.signal?.removeEventListener('abort',interrupt);await rm(temporary,{recursive:true,force:true});
  }
}
