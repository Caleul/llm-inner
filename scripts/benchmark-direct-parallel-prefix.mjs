/** Fixed-work diagnostic only. Cancels independently generated prefixes;
 * never emits/admit/executes a complete-model artifact. */
import { isMainThread, parentPort, workerData } from 'node:worker_threads';
import { mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { DirectRustStream } from '../dist/src/direct-rust-stream.js';
import { prepareDirectModel, writeDirectFlatRustUnit } from '../dist/src/direct-flat-rust-model.js';
import { directPoolBudget, runDirectCompilationPool } from '../dist/src/direct-compilation-pool.js';

if(!isMainThread){
  const original=DirectRustStream.prototype.write;
  DirectRustStream.prototype.write=function(source){
    if(this.bytes>=workerData.target)this.cancel('Fixed-work diagnostic complete');
    return original.call(this,source);
  };
  parentPort.on('message',async job=>{
    try{
      const path=job.path+'.compile';
      try{
        await writeDirectFlatRustUnit(workerData.checkpoint,workerData.dimension,path,workerData.discovered,job.unit,
          {weightCacheBytes:workerData.weightCacheBytes,maxOutputBytes:workerData.maxOutputBytes,outputCounter:workerData.outputCounter});
        throw new Error('Diagnostic workload completed before its requested byte target');
      }catch(error){if(!String(error).includes('Fixed-work diagnostic complete'))throw error;}
      const metadata=JSON.parse(await readFile(path+'.progress.json','utf8'));
      if(metadata.finalParity||metadata.rustCompilationAdmitted)throw new Error('Diagnostic draft admitted');
      await rename(path+'.draft',job.path);
      const hash=createHash('sha256');let bytes=0;
      for await(const chunk of createReadStream(job.path)){hash.update(chunk);bytes+=chunk.length;}
      parentPort.postMessage({result:{id:job.id,path:job.path,bytes,sha256:hash.digest('hex'),
        leaves:metadata.leaves,candidateLeaves:metadata.candidateLeaves,
        eliminatedBranches:metadata.eliminatedBranches,inspectedExpressions:metadata.inspectedExpressions,
        weightReads:metadata.weightReads,weightPageHits:metadata.weightPageHits}});
    }catch(error){parentPort.postMessage({error:String(error)});}
  });
}else{
  const [checkpoint,python,rawDimension,report,rawTarget='450000',rawJobs='4']=process.argv.slice(2);
  if(!report)throw new Error('Usage: benchmark-direct-parallel-prefix CHECKPOINT PYTHON DIMENSION REPORT [BYTE_TARGET] [JOBS]');
  const dimension=Number(rawDimension),target=Number(rawTarget),jobCount=Number(rawJobs);
  if(!Number.isSafeInteger(target)||target<1||!Number.isSafeInteger(jobCount)||jobCount<4||jobCount>64)
    throw new Error('Invalid diagnostic workload');
  const directory=await mkdtemp(join(tmpdir(),'direct-parallel-prefix-benchmark-'));
  try{
    const discovered=await prepareDirectModel(checkpoint,python,16*1024*1024),runs=[];
    for(const workers of [1,4]){
      const budget=directPoolBudget({workers,partitions:1,maxOutputMiB:Math.ceil(target*jobCount/1048576)+1});
      const jobs=Array.from({length:jobCount},(_,id)=>({id,unit:{position:0,fullVectorSoftmax:false},path:join(directory,`${workers}-${id}.partial`)}));
      const started=performance.now();
      const results=await runDirectCompilationPool(new URL(import.meta.url),{
        checkpoint:resolve(checkpoint),dimension,discovered,target,weightCacheBytes:budget.weightCacheMiB*1048576,
        maxOutputBytes:budget.maxOutputBytes,outputCounter:new SharedArrayBuffer(8)},jobs,budget);
      const hashes=new Set(results.map(r=>r.sha256));if(hashes.size!==1)throw new Error('Diagnostic jobs differ');
      if(runs.length&&results.some((r,i)=>r.sha256!==runs[0].results[i].sha256))throw new Error('Parallel compilation changed prefix bytes');
      const entry={workers,durationMs:performance.now()-started,results};runs.push(entry);
      console.log(JSON.stringify({...entry,results:undefined}));
    }
    await writeFile(report,JSON.stringify({scope:'Independent identical real-checkpoint prefixes; not partitioning or final-model parity',
      diagnosticOnly:true,completeModelParity:false,target,jobCount,byteIdentical:true,runs},null,2)+'\n');
  }finally{await rm(directory,{recursive:true,force:true});}
}
