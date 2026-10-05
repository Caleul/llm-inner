/** CPU-only symbolic closure/parity probe. Target identity is restored from a
 * verified source-host snapshot, not rediscovered as an x64/CUDA forward.
 * The diagnostic evaluator is never part of the emitted model. */
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {dirname,join} from 'node:path';
import {openJsonModelBuilder} from '../dist/src/direct-json-model.js';
import {createJsonModelLowerer} from '../dist/src/direct-json-lower-model.js';
import {measureJsonExpression} from '../dist/src/direct-json-measure.js';
import {evaluateJsonExpression} from '../dist/src/direct-json-evaluator.js';
import {loweredJsonHeader} from '../dist/src/direct-json-lower-model.js';
import {writeJsonScalarUnits} from '../dist/src/direct-json-stream.js';
import {decodeIeeeF16ToF32} from '../dist/src/utils.js';
const [checkpoint,snapshotPath,referencePath,lengthText,path,dimensionText]=process.argv.slice(2),length=Number(lengthText);
if(!checkpoint||!snapshotPath||!referencePath||!path||!Number.isSafeInteger(length)||length<1)throw new Error('Usage: CHECKPOINT TARGET_SNAPSHOT REFERENCE LENGTH OUTPUT [DIMENSION]');
mkdirSync(dirname(path),{recursive:true});
const snapshot=JSON.parse(readFileSync(snapshotPath,'utf8'));
const builder=await openJsonModelBuilder(checkpoint,'unused-source-target-snapshot',{discoverySnapshot:snapshot});
if(length>builder.header.context)throw new Error('Length exceeds discovered context');
const corpus=JSON.parse(readFileSync(referencePath,'utf8'));
const selected=dimensionText===undefined?Array.from({length:builder.header.outputWidth},(_,i)=>i):[Number(dimensionText)];
if(selected.some(i=>!Number.isSafeInteger(i)||i<0||i>=builder.header.outputWidth))throw new Error('Invalid discovered output coordinate');
const report={length,position:length-1,outputs:[],finalArtifactEmitted:false,finalArtifactParity:false};
const record=()=>writeFileSync(path,JSON.stringify(report,(_,x)=>typeof x==='bigint'?x.toString():x,2)+'\n');
record();
try{
 for(const dimension of selected){
  let lowering,dependencies=0,substitutions=0;const precision=new WeakMap(),started=performance.now();
  const row={dimension,dependencies:0,substitutions:0,verifiedLogits:0,mismatches:[],status:'compiling'};
  report.outputs.push(row);record();
  const {expression}=await builder.build(length-1,dimension,{onDependency:(key,node,facts)=>{
   lowering??=createJsonModelLowerer(facts,precision,{incremental:true,onSubstitution:event=>{substitutions=event.ordinal;row.simplificationCacheHits=event.simplificationCacheHits;row.simplificationCacheMisses=event.simplificationCacheMisses;}});
   lowering.lower(node);dependencies++;row.dependencies=dependencies;row.substitutions=substitutions;row.lastDependency=key;record();
  }});
  const closed=lowering.lower(expression);row.measure=measureJsonExpression(closed);
  for(const sample of corpus.cases.filter(c=>c.inputBits.length===length)){
   const values={N:BigInt(length)};
   for(let i=0;i<length;i++)for(let j=0;j<corpus.width;j++)values[`X${i*corpus.width+j+1}`]=decodeIeeeF16ToF32(sample.inputBits[i][j]);
   const answer=Number(evaluateJsonExpression(closed,values,{allowPendingPrimitives:false})),word=new DataView(new ArrayBuffer(8));word.setFloat64(0,answer);
   const actual='0x'+word.getBigUint64(0).toString(16).padStart(16,'0'),expected=sample.logitF64Bits[length-1][dimension];
   row.verifiedLogits++;if(actual!==expected)row.mismatches.push({case:sample.label,actual,expected});
  }
  row.primitiveFree=true;
  if(!row.mismatches.length){
   // Literal emission uses the real admission gate and the accumulated budget.
   // Compiler sharing and successful diagnostic evaluation cannot bypass it.
   if(row.measure.serializedBytes<=512n*1024n**2n){
    const destination=join(dirname(path),`length-${length}-logit-${dimension}.jsonl`);
    async function* units(){yield {position:length-1,dimension,expression:closed};}
    try{row.emission=await writeJsonScalarUnits(destination,loweredJsonHeader(builder.header),units(),{coordinate:{position:length-1,dimension},maxBytes:512*1024**2});}
    catch(error){row.emissionStop=String(error);}
   }else row.emissionStop='Expanded literal JSON exceeds 512 MiB; no truncated artifact emitted';
  }
  row.status=row.mismatches.length?'parity-failed':'validated-without-emission';row.seconds=(performance.now()-started)/1000;record();global.gc?.();
 }
}catch(error){report.stop=String(error);record();process.exitCode=1;}finally{await builder.close();record();}
