import {writeDirectJsonModel,type DirectJsonCompileOptions} from './direct-json-compile.js';
const [directory,python,path,...flags]=process.argv.slice(2);
if(!directory||!python||!path)throw new Error('Usage: direct-json-cli CHECKPOINT PYTHON OUTPUT.jsonl [--max-output-mib N] [--max-occurrences N] [--weight-cache-mib N] [--max-dependencies N] [--max-condition-candidates N] [--max-expression-nodes N] [--max-condition-rounds N] [--condition-workers N]');
const options:DirectJsonCompileOptions={};
const numericFlags:Record<string,[keyof DirectJsonCompileOptions,number]>={
  '--max-output-mib':['maxBytes',1024*1024],'--weight-cache-mib':['weightCacheBytes',1024*1024],
  '--max-occurrences':['maxOccurrences',1],'--max-dependencies':['maxDependencies',1],
  '--condition-workers':['maxConditionWorkers',1],'--max-condition-rounds':['maxConditionRounds',1],'--max-condition-candidates':['maxConditionCandidates',1],'--max-expression-nodes':['maxUniqueNodes',1]};
for(let i=0;i<flags.length;i++){
  const flag=flags[i]!,spec=numericFlags[flag],raw=flags[++i];
  if(!spec||raw===undefined||!/^\d+$/.test(raw)||!Number.isSafeInteger(Number(raw)*spec[1]))
    throw new Error(`Invalid JSON compiler option: ${flag}`);
  (options as Record<string,unknown>)[spec[0]]=Number(raw)*spec[1];
}
options.onPrepared=unit=>console.log(`Prepared ${unit.preparedUnits}/${unit.totalUnits}: position=${unit.position}, dimension=${unit.dimension}, expressionBytes=${unit.measure.serializedBytes}, conditionRounds=${unit.cofactor.rounds}, promotions=${unit.cofactor.accepted}, conditionConverged=${unit.cofactor.converged}, conditionStop=${unit.cofactor.stopReason}`);
try {
  const result=await writeDirectJsonModel(directory,python,path,options);
  console.log(`Complete JSON emitted: ${result.units} units, ${result.bytes} bytes. Final parity remains unverified.`);
}catch(error){console.error(error instanceof Error?error.message:String(error));process.exitCode=1;}
