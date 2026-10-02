import {parentPort,workerData} from 'node:worker_threads';
import {jsonOperation,type JsonExpression} from './direct-json-expression.js';
import {simplifyJsonExpression,simplifyJsonFixedPoint} from './direct-json-simplify.js';
import {measureJsonExpression} from './direct-json-measure.js';
import {shareJsonExpression} from './direct-json-share.js';
interface Job {base:JsonExpression;conditions:{index:number;condition:JsonExpression}[];maxNodes:number;maxVisits:number;beforeBytes:bigint}
// Compiler-only jobs. Structured cloning preserves the alias between each
// condition and its occurrences in base; no refs or worker program are emitted.
const job=workerData as Job;
let best:JsonExpression|undefined,bestIndex=Infinity,bestBytes=job.beforeBytes,reducingCandidates=0,overBudget=0;
try{
  for(const {index,condition} of job.conditions){
    try{
      const yes=simplifyJsonFixedPoint(simplifyJsonExpression(job.base,undefined,[[condition,true]],job.maxVisits),32,job.maxVisits).expression;
      const no=simplifyJsonFixedPoint(simplifyJsonExpression(job.base,undefined,[[condition,false]],job.maxVisits),32,job.maxVisits).expression;
      const candidate=shareJsonExpression(jsonOperation('if',job.base[1],condition,yes,no),job.maxVisits).expression,
        bytes=measureJsonExpression(candidate,job.maxNodes).serializedBytes;
      if(bytes<job.beforeBytes)reducingCandidates++;
      if(bytes<bestBytes){best=candidate;bestBytes=bytes;bestIndex=index;}
    }catch(error){if(error instanceof RangeError){overBudget++;continue;}throw error;}
  }
  parentPort!.postMessage({best,bestIndex,bestBytes,reducingCandidates,overBudget});
}catch(error){parentPort!.postMessage({fatal:error instanceof Error?error.message:String(error)});}
