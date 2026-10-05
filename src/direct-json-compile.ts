import {openJsonModelBuilder,type JsonModelConstructionStats} from './direct-json-model.js';
import {loweredJsonHeader,createJsonModelLowerer,type JsonSubstitutionProgress} from './direct-json-lower-model.js';
import {simplifyJsonBitPrecision,type JsonPrecisionFacts} from './direct-json-precision.js';
import type {JsonCofactorStats} from './direct-json-cofactor.js';
import {simplifyJsonSharedConditionsParallel} from './direct-json-cofactor-parallel.js';
import {measureJsonExpression,type JsonExpressionMeasure} from './direct-json-measure.js';
import {writeJsonScalarUnits,type JsonScalarUnit} from './direct-json-stream.js';
import {composeJsonNextToken} from './direct-json-next-token.js';
import type {DirectModelSnapshot} from './direct-model-snapshot.js';

export interface DirectJsonCompileOptions {
  coordinate?:{position:number;dimension:number};
  /** Matrix emission is an explicit diagnostic; the product is next-token. */
  outputScope?:'next-token'|'all-positions';
  discoverySnapshot?:DirectModelSnapshot;
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
  if(options.outputScope!==undefined&&options.outputScope!=='next-token'&&options.outputScope!=='all-positions')
    throw new TypeError('Unknown compiler output scope');
  const builder=await openJsonModelBuilder(directory,python,{
    ...(options.discoverySnapshot?{discoverySnapshot:options.discoverySnapshot}:{}),
    ...(options.weightCacheBytes!==undefined?{weightCacheBytes:options.weightCacheBytes}:{}),
    ...(options.maxDependencies!==undefined?{maxDependencies:options.maxDependencies}:{})});
  try {
    const nextToken=!options.coordinate&&options.outputScope!=='all-positions';
    const header={...loweredJsonHeader(builder.header),...(nextToken?{outputSelection:'last-position' as const}:{})};
    async function* units():AsyncGenerator<JsonScalarUnit>{
      let preparedUnits=0;
      for(let position=options.coordinate?.position??0;position<(options.coordinate?options.coordinate.position+1:nextToken?1:header.context);position++)
      for(let dimension=options.coordinate?.dimension??0;dimension<(options.coordinate?options.coordinate.dimension+1:header.outputWidth);dimension++){
        const precision:JsonPrecisionFacts=new WeakMap();
        let construction:JsonModelConstructionStats|undefined;
        const build=async(position:number)=>{
          let lowering:ReturnType<typeof createJsonModelLowerer>|undefined;
          const {expression,stats}=await builder.build(position,dimension,{onDependency:(key,node,facts)=>{
            lowering??=createJsonModelLowerer(facts,precision,{incremental:true,
              ...(options.onSubstitution?{onSubstitution:event=>options.onSubstitution!({...event,position,dimension})}:{})});
            const substituted=lowering.lower(node);
            options.onDependency?.({position,dimension,key,measure:measureJsonExpression(substituted)});
          }});
          if(!construction)construction={...stats};
          else for(const key of Object.keys(stats) as (keyof JsonModelConstructionStats)[])construction[key]+=stats[key];
          return lowering!.lower(expression);
        };
        const lowered=nextToken?await composeJsonNextToken(header.context,'N',build):await build(position);
        const {expression:closed,stats:cofactor}=await simplifyJsonSharedConditionsParallel(simplifyJsonBitPrecision(lowered,precision),{
          ...(options.maxConditionCandidates!==undefined?{maxCandidates:options.maxConditionCandidates}:{}),
          ...(options.maxUniqueNodes!==undefined?{maxUniqueNodes:options.maxUniqueNodes}:{}),
          ...(options.maxConditionRounds!==undefined?{maxRounds:options.maxConditionRounds}:{}),
          ...(options.maxConditionWorkers!==undefined?{workers:options.maxConditionWorkers}:{})});
        const measure=measureJsonExpression(closed,options.maxUniqueNodes??100_000);
        options.onPrepared?.({position,dimension,preparedUnits:++preparedUnits,
          totalUnits:options.coordinate?1:nextToken?header.outputWidth:header.context*header.outputWidth,construction:construction!,cofactor,measure});
        if(!cofactor.converged)throw new Error(`Shared-condition simplification is unfinished: ${cofactor.stopReason}`);
        yield {position,dimension,expression:closed};
      }
    }
    return await writeJsonScalarUnits(path,header,units(),{
      ...(options.coordinate?{coordinate:options.coordinate}:{}),
      maxBytes:options.maxBytes??512*1024*1024,
      ...(options.maxOccurrences!==undefined?{maxOccurrences:options.maxOccurrences}:{})});
  }finally{await builder.close();}
}
