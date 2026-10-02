import {openJsonModelBuilder,type JsonModelConstructionStats} from './direct-json-model.js';
import {loweredJsonHeader,lowerJsonModelExpression} from './direct-json-lower-model.js';
import {simplifyJsonBitPrecision,type JsonPrecisionFacts} from './direct-json-precision.js';
import {simplifyJsonSharedConditions,type JsonCofactorStats} from './direct-json-cofactor.js';
import {measureJsonExpression,type JsonExpressionMeasure} from './direct-json-measure.js';
import {writeJsonScalarUnits,type JsonScalarUnit} from './direct-json-stream.js';

export interface DirectJsonCompileOptions {
  weightCacheBytes?:number;maxDependencies?:number;maxBytes?:number;maxOccurrences?:number;
  maxConditionCandidates?:number;maxUniqueNodes?:number;
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
      for(let position=0;position<header.context;position++)for(let dimension=0;dimension<header.outputWidth;dimension++){
        const {expression,stats,facts}=await builder.build(position,dimension),precision:JsonPrecisionFacts=new WeakMap();
        const lowered=lowerJsonModelExpression(expression,facts,precision);
        const {expression:closed,stats:cofactor}=simplifyJsonSharedConditions(simplifyJsonBitPrecision(lowered,precision),{
          ...(options.maxConditionCandidates!==undefined?{maxCandidates:options.maxConditionCandidates}:{}),
          ...(options.maxUniqueNodes!==undefined?{maxUniqueNodes:options.maxUniqueNodes}:{})});
        const measure=measureJsonExpression(closed,options.maxUniqueNodes??100_000);
        options.onPrepared?.({position,dimension,preparedUnits:++preparedUnits,
          totalUnits:header.context*header.outputWidth,construction:stats,cofactor,measure});
        yield {position,dimension,expression:closed};
      }
    }
    return await writeJsonScalarUnits(path,header,units(),{
      ...(options.maxBytes!==undefined?{maxBytes:options.maxBytes}:{}),
      ...(options.maxOccurrences!==undefined?{maxOccurrences:options.maxOccurrences}:{})});
  }finally{await builder.close();}
}
