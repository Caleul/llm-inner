/** Bounded compilation experiment; never claims complete-model parity. */
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import { pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import inspector from 'node:inspector';

if (!isMainThread) {
  const { moduleRoot, directory, jobs, iterations, bufferLimit, profile } = workerData;
  const { DirectRustStream } = await import(pathToFileURL(join(moduleRoot,'src/direct-rust-stream.js')));
  const { DirectFlatSubstitution, FlatConditions } = await import(pathToFileURL(join(moduleRoot,'src/direct-flat-substitution.js')));
  let session;
  const post=(method)=>new Promise((ok,no)=>session.post(method,(e,r)=>e?no(e):ok(r)));
  if(profile){session=new inspector.Session();session.connect();await post('Profiler.enable');await post('Profiler.start');}
  const results=[];
  for(const job of jobs){
    const path=join(directory,`${job}.rs`),s=new DirectRustStream(path,bufferLimit),f=new DirectFlatSubstitution(s);
    for(let i=0;i<iterations;i++){
      await s.write(`fn generated_${job}_${i}(x:f64)->f64 {'result:{`);s.beginReducedExpression();
      // Same actual substitution, range analysis, rounding, flattening and
      // admission APIs used by the model compiler. No forward/output lookup.
      const input=(p,k)=>f.input(p,()=>s.write('x'),1,1.001,k);
      const expression=(depth)=>(p,k)=>depth===0?input(p,k):f.binary(p,expression(depth-1),
        (q,c)=>f.literal(q,1+(job%3)*2**-12,c),'*',k);
      await f.round(new FlatConditions(),expression(8),'f32',false,(p,v)=>f.leaf(p,'result',v));
      await f.finishRound();await s.write('panic!("outside domain")}}\n');
    }
    await s.close();
    const bytes=await readFile(path);
    results.push({job,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),
      inspections:s.inspectedExpressions,outputWrites:s.outputWrites??null});
  }
  if(profile){const result=await post('Profiler.stop');await writeFile(profile,JSON.stringify(result.profile));session.disconnect();}
  parentPort.postMessage(results);
} else {
  const [baselineRoot,candidateRoot,reportPath,rawIterations='200',rawJobs='8']=process.argv.slice(2);
  if(!baselineRoot||!candidateRoot||!reportPath)throw new Error('Usage: benchmark-direct-stream BASELINE_BUILD CANDIDATE_BUILD REPORT [ITERATIONS] [JOBS]');
  const iterations=Number(rawIterations),jobCount=Number(rawJobs);
  if(!Number.isSafeInteger(iterations)||iterations<1||!Number.isSafeInteger(jobCount)||jobCount<4)throw new Error('Invalid workload');
  const directory=await mkdtemp(join(tmpdir(),'direct-stream-benchmark-'));
  const runs=[];
  try {
    for(const [name,root,workers,bufferLimit] of [
      ['baseline',baselineRoot,1,0],['buffered',candidateRoot,1,65536],['parallel-4',candidateRoot,4,65536],
      ['buffered-repeat',candidateRoot,1,65536],['baseline-repeat',baselineRoot,1,0],['parallel-4-repeat',candidateRoot,4,65536]]){
      const started=performance.now();
      const batches=Array.from({length:workers},()=>[]);
      for(let job=0;job<jobCount;job++)batches[job%workers].push(job);
      const results=(await Promise.all(batches.map((jobs,index)=>new Promise((ok,no)=>{
        const worker=new Worker(new URL(import.meta.url),{workerData:{moduleRoot:resolve(root),directory,jobs,iterations,bufferLimit,
          profile:name==='baseline'&&index===0?resolve(reportPath)+'.cpuprofile':undefined}});
        worker.once('message',ok);worker.once('error',no);worker.once('exit',code=>{if(code!==0)no(new Error(`Worker exit ${code}`));});
      })))).flat().sort((a,b)=>a.job-b.job);
      const durationMs=performance.now()-started;
      const reference=runs[0]?.results;
      if(reference&&results.some((r,i)=>r.sha256!==reference[i].sha256||r.bytes!==reference[i].bytes||r.inspections!==reference[i].inspections))
        throw new Error(`Changed compilation output: ${name}`);
      const entry={name,workers,durationMs,bytes:results.reduce((n,r)=>n+r.bytes,0),
        inspections:results.reduce((n,r)=>n+r.inspections,0),outputWrites:results.every(r=>r.outputWrites!==null)?results.reduce((n,r)=>n+r.outputWrites,0):null,results};
      runs.push(entry);console.log(JSON.stringify({...entry,results:undefined}));
    }
    await writeFile(reportPath,JSON.stringify({scope:'substitution/emission experiment, not full-model parity',iterations,jobCount,byteIdentical:true,runs},null,2)+'\n');
  }finally{await rm(directory,{recursive:true,force:true});}
}
