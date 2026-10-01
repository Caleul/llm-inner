import { parentPort, workerData } from 'node:worker_threads';
import { writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import type { DirectUnitJob } from '../src/direct-compilation-pool.js';
parentPort!.on('message',async(job:DirectUnitJob)=>{
  if(job.id===workerData.fail){parentPort!.postMessage({error:'fixture worker failure'});return;}
  if(job.id===workerData.exit){process.exit(0);}
  await new Promise(ok=>setTimeout(ok,workerData.delay??((4-job.id%4)*10)));
  const source=`unit ${job.id}\n`;
  await writeFile(job.path,source);
  parentPort!.postMessage({result:{id:job.id,path:job.path,bytes:Buffer.byteLength(source),
    sha256:createHash('sha256').update(source).digest('hex'),leaves:1,candidateLeaves:1,
    eliminatedBranches:0,inspectedExpressions:0,weightReads:0,weightPageHits:0}});
});
