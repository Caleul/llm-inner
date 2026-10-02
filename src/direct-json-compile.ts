import {openJsonModelBuilder,type JsonModelConstructionStats} from './direct-json-model.js';
import {loweredJsonHeader,createJsonModelLowerer,type JsonSubstitutionProgress} from './direct-json-lower-model.js';
import {simplifyJsonBitPrecision,type JsonPrecisionFacts} from './direct-json-precision.js';
import type {JsonCofactorStats} from './direct-json-cofactor.js';
import {simplifyJsonSharedConditionsParallel} from './direct-json-cofactor-parallel.js';
import {measureJsonExpression,type JsonExpressionMeasure} from './direct-json-measure.js';
import {writeJsonScalarUnits,type JsonScalarUnit} from './direct-json-stream.js';

export interface DirectJsonCompileOptions {
  coordinate?:{position:number;dimension:number};
  weightCacheBytes?:number;maxDependencies?:number;maxBytes?:number;maxOccurrences?:number;
  maxConditionCandidates?:number;maxUniqueNodes?:number;maxConditionRounds?:number;maxConditionWorkers?:number;
  onSubstitution?:(event:JsonSubstitutionProgress & {position:number;dimension:number})=>void;
  onDependency?:(event:{position:number;dimension:number;key:string;measure:JsonExpressionMeasure})=>void;
  onPrepared?:(unit:{position:number;dimension:number;preparedUnits:number;totalUnits:number;
    construction:JsonModelConstructionStats;cofactor:JsonCofactorStats;measure:JsonExpressionMeasure})=>void;
}
/** Source/config -> closed expressions -> fixed-point rules -> condition
 * combination -> literal JSONL. Preparing a scalar is not successful emission;
 * the writer admits the full vector atomically or preserves the prior target.
 * One scalar unit is resident, and checkpoint reads stay in bounded pages. */
export async function writeDirectJsonModel(directory:string,python:string,path:string,
  options:DirectJsonCompileOptions={}):Promise<Awaited<ReturnType<typeof writeJsonScalarUnits>>> {
  const builder=await openJsonModelBuilder(directory,python,{
    ...(options.weightCacheBytes!==undefined?{weightCacheBytes:options.weightCacheBytes}:{}),
    ...(options.maxDependencies!==undefined?{maxDependencies:options.maxDependencies}:{})});
  try {
    const header=loweredJsonHeader(builder.header);
    async function* units():AsyncGenerator<JsonScalarUnit>{
      let preparedUnits=0;
      for(let position=options.coordinate?.position??0;position<(options.coordinate?options.coordinate.position+1:header.context);position++)
      for(let dimension=options.coordinate?.dimension??0;dimension<(options.coordinate?options.coordinate.dimension+1:header.outputWidth);dimension++){
        const precision:JsonPrecisionFacts=new WeakMap();
        let lowering:ReturnType<typeof createJsonModelLowerer>|undefined;
        const {expression,stats}=await builder.build(position,dimension,{onDependency:(key,node,facts)=>{
          lowering??=createJsonModelLowerer(facts,precision,{incremental:true,
            ...(options.onSubstitution?{onSubstitution:event=>options.onSubstitution!({...event,position,dimension})}:{})});
          const substituted=lowering.lower(node);
          options.onDependency?.({position,dimension,key,measure:measureJsonExpression(substituted)});
        }});
        const lowered=lowering!.lower(expression);
        const {expression:closed,stats:cofactor}=await simplifyJsonSharedConditionsParallel(simplifyJsonBitPrecision(lowered,precision),{
          ...(options.maxConditionCandidates!==undefined?{maxCandidates:options.maxConditionCandidates}:{}),
          ...(options.maxUniqueNodes!==undefined?{maxUniqueNodes:options.maxUniqueNodes}:{}),
          ...(options.maxConditionRounds!==undefined?{maxRounds:options.maxConditionRounds}:{}),
          ...(options.maxConditionWorkers!==undefined?{workers:options.maxConditionWorkers}:{})});
        const measure=measureJsonExpression(closed,options.maxUniqueNodes??100_000);
        options.onPrepared?.({position,dimension,preparedUnits:++preparedUnits,
          totalUnits:options.coordinate?1:header.context*header.outputWidth,construction:stats,cofactor,measure});
        yield {position,dimension,expression:closed};
      }
    }
    return await writeJsonScalarUnits(path,header,units(),{
      ...(options.coordinate?{coordinate:options.coordinate}:{}),
      ...(options.maxBytes!==undefined?{maxBytes:options.maxBytes}:{}),
      ...(options.maxOccurrences!==undefined?{maxOccurrences:options.maxOccurrences}:{})});
  }finally{await builder.close();}
}
