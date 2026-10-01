import { parentPort, workerData } from 'node:worker_threads';
import { readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { createHash } from 'node:crypto';
import { writeDirectFlatRustUnit, type DirectModelDiscovery } from './direct-flat-rust-model.js';
import type { DirectUnitJob, DirectUnitResult } from './direct-compilation-pool.js';

const data=workerData as {directory:string;dimension:number;discovered:DirectModelDiscovery;
  weightCacheBytes:number;maxOutputBytes:number;outputCounter:SharedArrayBuffer};
if(!parentPort)throw new Error('Compiler worker requires a parent port');
parentPort.on('message',async(job:DirectUnitJob)=>{
  try{
    await writeDirectFlatRustUnit(data.directory,data.dimension,job.path,data.discovered,job.unit,
      {weightCacheBytes:data.weightCacheBytes,maxOutputBytes:data.maxOutputBytes,outputCounter:data.outputCounter});
    const metadata=JSON.parse(await readFile(job.path+'.reduction.json','utf8'));
    if(metadata.status!=='fragment'||metadata.finalParity!==false)throw new Error('Incomplete compilation fragment');
    const info=await stat(job.path);if(info.size!==metadata.bytes)throw new Error('Fragment byte count mismatch');
    const hash=createHash('sha256');for await(const chunk of createReadStream(job.path))hash.update(chunk);
    const result:DirectUnitResult={id:job.id,path:job.path,sha256:hash.digest('hex'),bytes:info.size,
      leaves:metadata.leaves,candidateLeaves:metadata.candidateLeaves,
      eliminatedBranches:metadata.eliminatedBranches,inspectedExpressions:metadata.inspectedExpressions,
      weightReads:metadata.weightReads,weightPageHits:metadata.weightPageHits};
    parentPort!.postMessage({result});
  }catch(error){parentPort!.postMessage({error:error instanceof Error?error.message:String(error)});}
});
