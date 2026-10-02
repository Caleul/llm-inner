import {SafetensorsCatalogReader} from './safetensors.js';
import {DirectWeightPages} from './direct-weight-pages.js';
import {prepareDirectModel,type DirectModelDiscovery} from './direct-flat-rust-model.js';
import {fixedF16RopeLiteral} from './fixed-f16-rope-branches.js';
import {decodeIeeeF16ToF32} from './utils.js';
import {jsonConstant as c,jsonInput,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {lowerJsonF32ToF16} from './direct-json-f16.js';
import type {Projection} from './direct-mlp-output.js';
import type {JsonScalarHeader} from './direct-json-stream.js';
import type {JsonFloatRange} from './direct-json-range.js';

export interface JsonModelConstructionStats {
  f32Arithmetic:number;f16Conversions:number;squareRoots:number;exponentials:number;
  activations:number;maximumComparisons:number;lengthDecisions:number;dependencies:number;
}
export interface JsonModelLoweringFacts {
  halfSources:WeakMap<JsonExpression,JsonExpression>;
  positiveNormalRoots:WeakSet<JsonExpression>;
  exponentialBounds:WeakMap<JsonExpression,number>;
  activationBounds:WeakMap<JsonExpression,number>;
  ranges?:WeakMap<JsonExpression,JsonFloatRange>;
}
/** Builds fully substituted scalar syntax backwards from an arbitrary logit.
 * The cache contains compiler expressions, not activations or forward values.
 * It is cleared between scalar units. Pending numerical primitives are explicit
 * and cannot pass final JSON admission until their lowering has been proved. */
export async function openJsonModelBuilder(directory:string,python:string,
  resources:{weightCacheBytes?:number;maxDependencies?:number}={}):Promise<{
    header:JsonScalarHeader;discovered:DirectModelDiscovery;
    build:(position:number,dimension:number)=>Promise<{expression:JsonExpression;stats:JsonModelConstructionStats;facts:JsonModelLoweringFacts}>;
    close:()=>Promise<void>
  }> {
  const discovered=await prepareDirectModel(directory,python,resources.weightCacheBytes??16*1024*1024),{output,layers}=discovered;
  const reader=new SafetensorsCatalogReader(directory),catalog=await reader.inspect();
  const pages=new DirectWeightPages(reader,resources.weightCacheBytes);
  try {
  const width=output.shape[1],maxDependencies=resources.maxDependencies??100_000;
  if(!Number.isSafeInteger(maxDependencies)||maxDependencies<1)throw new RangeError('Invalid dependency budget');
  if(width>1_000_000||output.maxPosition>1_000_000||width*output.maxPosition>maxDependencies){
    throw new RangeError('Input declaration or normal/zero numerical proof budget exceeded');
  }
  const projectionShape=(projection:Projection,rows:number,columns:number)=>{
    const tensor=catalog.tensors.get(projection.weight);
    if(projection.shape[0]!==rows||projection.shape[1]!==columns||!tensor||tensor.storageDtype!=='F16'||
      tensor.logicalShape.length!==2||tensor.logicalShape[0]!==rows||tensor.logicalShape[1]!==columns)
      throw new Error('Discovered JSON projection shape is inconsistent with source tensor');
  };
  projectionShape({weight:output.weight,shape:output.shape},output.shape[0],width);
  for(const layer of layers){
    const a=layer.attention.attention,m=layer.mlp.mlp;
    if(a.headDim%2||a.heads%a.kvHeads||layer.mlp.normalizations.length!==2)
      throw new Error('Unsupported discovered JSON geometry');
    projectionShape(a.projections.q,a.heads*a.headDim,width);
    for(const p of [a.projections.k,a.projections.v])projectionShape(p,a.kvHeads*a.headDim,width);
    projectionShape(a.projections.o,width,a.heads*a.headDim);
    projectionShape(m.gate,m.gate.shape[0],width);
    projectionShape(m.up,m.gate.shape[0],width);
    projectionShape(m.down,width,m.gate.shape[0]);
  }
  const activationBounds:number[][]=[];
  for(const layer of layers){
    let gamma=0;const post=layer.mlp.normalizations[1]!;
    for(let i=0;i<width;i++)gamma=Math.max(gamma,Math.abs(await pages.read(catalog.tensors.get(post.weight)!,i)));
    const gate=layer.mlp.mlp.gate,bounds:number[]=[];
    for(let neuron=0;neuron<gate.shape[0];neuron++){
      let absolute=0;for(let i=0;i<gate.shape[1];i++)absolute+=Math.abs(await pages.read(catalog.tensors.get(gate.weight)!,neuron*gate.shape[1]+i));
      bounds.push((2*Math.sqrt(width)*gamma*absolute+absolute*2**-23)*1.125+2**-24);
    }
    activationBounds.push(bounds);
  }
  const header:JsonScalarHeader={schema:'direct-scalar-json-v1',inputWidth:width,context:output.maxPosition,
    outputWidth:output.shape[0],inputs:{N:{dtype:'u32',source:'inputLength'}}};
  for(let row=0;row<output.maxPosition;row++)for(let column=0;column<width;column++)
    header.inputs[`X${row*width+column+1}`]={dtype:'f16',tokenPosition:row,coordinate:column};
  const weight=async(name:string,index:number)=>{
    const tensor=catalog.tensors.get(name);
    if(!tensor||tensor.storageDtype!=='F16')throw new Error('Source-discovered F16 weight missing');
    return pages.read(tensor,index);
  };
  async function build(position:number,dimension:number){
    if(!Number.isSafeInteger(position)||position<0||position>=output.maxPosition||!Number.isSafeInteger(dimension)||
      dimension<0||dimension>=output.shape[0])throw new RangeError('Invalid discovered output coordinate');
    const stats:JsonModelConstructionStats={f32Arithmetic:0,f16Conversions:0,squareRoots:0,
      exponentials:0,activations:0,maximumComparisons:0,lengthDecisions:0,dependencies:0};
    const memo=new Map<string,Promise<JsonExpression>>();
    const facts:JsonModelLoweringFacts={halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),
      exponentialBounds:new WeakMap(),activationBounds:new WeakMap(),ranges:new WeakMap()};
    const dependency=(key:string,make:()=>Promise<JsonExpression>):Promise<JsonExpression>=>{
      const hit=memo.get(key);if(hit)return hit;
      if(++stats.dependencies>maxDependencies)throw new RangeError('Compilation dependency budget exceeded');
      const pending=make();memo.set(key,pending);return pending;
    };
    const f32=(op:'add'|'sub'|'mul'|'div',a:JsonExpression,b:JsonExpression)=>{
      stats.f32Arithmetic++;return o(op,'f32',a,b);
    };
    const widen=(x:JsonExpression)=>x[1]==='f32'?x:o('widen','f32',x);
    const half=(x:JsonExpression)=>{stats.f16Conversions++;const result=lowerJsonF32ToF16(x);facts.halfSources.set(result,x);return result;};
    const zero=c('f32',0);
    const add=(a:JsonExpression,b:JsonExpression)=>f32('add',a,b);
    async function reduction(size:number,term:(index:number)=>Promise<JsonExpression>):Promise<JsonExpression>{
      const terms:JsonExpression[]=[];for(let i=0;i<size;i++)terms.push(await term(i));
      const lanes:JsonExpression[]=[];
      for(let lane=0;lane<4;lane++){
        let value=zero;for(let i=lane;i<size;i+=4)value=add(value,terms[i]!);lanes.push(value);
      }
      return half(add(add(lanes[0]!,lanes[1]!),add(lanes[2]!,lanes[3]!)));
    }
    const linear=(projection:Projection,row:number,input:(index:number)=>Promise<JsonExpression>)=>
      reduction(projection.shape[1],async column=>f32('mul',widen(await input(column)),
        c('f32',await weight(projection.weight,row*projection.shape[1]+column))));
    // CPU arm64 RMS reduction order, including cascade geometry for larger widths.
    async function rmsSum(input:(index:number)=>Promise<JsonExpression>):Promise<JsonExpression>{
      const vector=width>=4,rows=vector?Math.floor(width/4):width,groups=Math.floor(rows/4);
      const step=2**Math.max(4,Math.floor(Math.ceil(Math.log2(Math.max(groups,1)))/4));
      const load=async(row:number,lane:number)=>{const x=widen(await input(vector?row*4+lane:row));return f32('mul',x,x);};
      const component=async(lane:number)=>{
        const partial=async(part:number)=>{
          const block=async(level:number,start:number):Promise<JsonExpression>=>{
            if(level===0)return load(part+4*start,lane);
            const span=step**(level-1);let value=await block(level-1,start);
            for(let i=1;i<step;i++)value=add(value,await block(level-1,start+i*span));return value;
          };
          let value:JsonExpression|undefined;
          // Ascending cascade levels preserve the reference nesting: lower
          // remainder groups precede each higher complete block sum.
          for(let level=0;level<=3;level++){
            const span=step**level,end=Math.floor(groups/span)*span;
            const start=level===3?0:Math.floor(groups/(span*step))*span*step;
            let accumulator:JsonExpression|undefined;
            for(let i=start;i<end;i+=span){const x=await block(level,i);accumulator=accumulator?add(accumulator,x):x;}
            if(accumulator)value=value?add(value,accumulator):accumulator;
          }
          for(let i=4*groups;i<rows&&part===0;i++){
            const x=await load(i,lane);value=value?add(value,x):x;
          }
          return value??zero;
        };
        let value=await partial(0);if(groups>0)for(let part=1;part<4;part++)value=add(value,await partial(part));return value;
      };
      if(!vector)return component(0);
      let value:JsonExpression|undefined;
      for(let i=rows*4;i<width;i++){
        const x=widen(await input(i)),square=f32('mul',x,x);value=value?add(value,square):square;
      }
      for(let lane=0;lane<4;lane++){const x=await component(lane);value=value?add(value,x):x;}
      return value!;
    }
    const norm=(key:string,name:string,epsilon:number,coordinate:number,input:(index:number)=>Promise<JsonExpression>)=>
      dependency(key+':'+coordinate,async()=>{
        if(!(Math.fround(epsilon)>=2**-126&&Number.isFinite(Math.fround(epsilon))))
          throw new Error('Positive normal epsilon proof required');
        const inverse=await dependency(key+':inverse',async()=>{
          const variance=add(f32('div',await rmsSum(input),c('f32',width)),c('f32',epsilon));
          stats.squareRoots++;const root=o('pending-sqrt','f32',variance);
          facts.positiveNormalRoots.add(root);
          return f32('div',c('f32',1),root);
        });
        const normalized=half(f32('mul',widen(await input(coordinate)),inverse));
        // RMS magnitude <=sqrt(width) in real arithmetic. The existing
        // finite-normalization certificate admits <1.25 relative/error loss;
        // factor two covers both explicit F32 and F16 rounding boundaries.
        const bound=2*Math.sqrt(width);
        facts.ranges!.set(normalized,{minimum:-bound,maximum:bound});
        return half(f32('mul',widen(normalized),c('f32',await weight(name,coordinate))));
      });
    const hidden=(layer:number,coordinate:number,row:number):Promise<JsonExpression>=>
      dependency(`h:${layer}:${row}:${coordinate}`,async()=>{
        if(layer<0)return jsonInput('f16',`X${row*width+coordinate+1}`);
        const mlp=layers[layer]!.mlp.mlp;
        const down=await linear(mlp.down,coordinate,neuron=>dependency(`gated:${layer}:${row}:${neuron}`,async()=>{
          const gate=await linear(mlp.gate,neuron,column=>postNorm(layer,column,row));
          stats.activations++;const activation=o('pending-silu','f16',gate);
          facts.activationBounds.set(activation,activationBounds[layer]![neuron]!);
          const up=await linear(mlp.up,neuron,column=>postNorm(layer,column,row));
          return half(f32('mul',widen(activation),widen(up)));
        }));
        return half(add(widen(await residual(layer,coordinate,row)),widen(down)));
      });
    const preNorm=(layer:number,coordinate:number,row:number)=>{
      const n=layers[layer]!.mlp.normalizations[0]!;
      return norm(`pre:${layer}:${row}`,n.weight,n.epsilon,coordinate,column=>hidden(layer-1,column,row));
    };
    const postNorm=(layer:number,coordinate:number,row:number)=>{
      const n=layers[layer]!.mlp.normalizations[1]!;
      return norm(`post:${layer}:${row}`,n.weight,n.epsilon,coordinate,column=>residual(layer,column,row));
    };
    const projected=(layer:number,role:'q'|'k'|'v',coordinate:number,row:number)=>
      dependency(`projection:${layer}:${role}:${row}:${coordinate}`,()=>
        linear(layers[layer]!.attention.attention.projections[role],coordinate,column=>preNorm(layer,column,row)));
    const rotated=(layer:number,role:'q'|'k',head:number,coordinate:number,row:number)=>
      dependency(`rope:${layer}:${role}:${head}:${row}:${coordinate}`,async()=>{
        const a=layers[layer]!.attention.attention,halfWidth=a.headDim/2;
        const projectionHead=role==='q'?head:Math.floor(head/(a.heads/a.kvHeads));
        const cos=decodeIeeeF16ToF32(fixedF16RopeLiteral(row,coordinate,a.headDim,a.ropeTheta,0));
        const sin=decodeIeeeF16ToF32(fixedF16RopeLiteral(row,coordinate,a.headDim,a.ropeTheta,1));
        const x=await projected(layer,role,projectionHead*a.headDim+coordinate,row);
        const other=await projected(layer,role,projectionHead*a.headDim+(coordinate+halfWidth)%a.headDim,row);
        return half(add(widen(half(f32('mul',widen(x),c('f32',cos)))),
          widen(half(f32('mul',widen(other),c('f32',coordinate<halfWidth?-sin:sin))))));
      });
    const score=(layer:number,head:number,query:number,key:number)=>
      dependency(`score:${layer}:${head}:${query}:${key}`,async()=>{
        const a=layers[layer]!.attention.attention;
        const dot=await reduction(a.headDim,async coordinate=>f32('mul',
          widen(await rotated(layer,'q',head,coordinate,query)),widen(await rotated(layer,'k',head,coordinate,key))));
        return half(f32('mul',widen(dot),c('f32',a.scaling)));
      });
    const maximum=(layer:number,head:number,query:number)=>dependency(`max:${layer}:${head}:${query}`,async()=>{
      let result=await score(layer,head,query,0);
      for(let key=1;key<=query;key++){
        const current=await score(layer,head,query,key);stats.maximumComparisons++;
        result=o('if','f16',o('lt','bool',result,current),current,result);
      }
      return result;
    });
    const exponent=(layer:number,head:number,query:number,key:number)=>dependency(`exp:${layer}:${head}:${query}:${key}`,async()=>{
      const difference=f32('sub',widen(await score(layer,head,query,key)),widen(await maximum(layer,head,query)));
      stats.exponentials++;const result=o('pending-exp','f32',difference);
      facts.exponentialBounds.set(result,2*discovered.proof!.scoreBounds[layer]!*1.01+2**-149);
      return result;
    });
    const reciprocal=(layer:number,head:number,query:number)=>dependency(`denominator:${layer}:${head}:${query}`,async()=>{
      const terms:JsonExpression[]=[];for(let key=0;key<=query;key++)terms.push(await exponent(layer,head,query,key));
      let sequential=terms[0]!;for(let key=1;key<=query;key++)sequential=add(sequential,terms[key]!);
      const lanes:JsonExpression[]=[];
      for(let lane=0;lane<4;lane++){
        let value:JsonExpression|undefined;for(let key=lane;key<=query;key+=4)value=value?add(value,terms[key]!):terms[key]!;
        lanes.push(value??zero);
      }
      const full=add(add(lanes[0]!,lanes[2]!),add(lanes[1]!,lanes[3]!));
      let denominator=query>=3?full:sequential;
      if(query===2&&output.maxPosition>=4){stats.lengthDecisions++;
        denominator=o('if','f32',o('lt','bool',jsonInput('u32','N'),c('u32',4)),sequential,full);}
      return f32('div',c('f32',1),denominator);
    });
    const context=(layer:number,head:number,coordinate:number,query:number)=>dependency(`context:${layer}:${head}:${query}:${coordinate}`,async()=>{
      const a=layers[layer]!.attention.attention;
      return reduction(query+1,async key=>{
        const probability=await dependency(`prob:${layer}:${head}:${query}:${key}`,async()=>
          half(f32('mul',await exponent(layer,head,query,key),await reciprocal(layer,head,query))));
        const value=await projected(layer,'v',Math.floor(head/(a.heads/a.kvHeads))*a.headDim+coordinate,key);
        return f32('mul',widen(probability),widen(value));
      });
    });
    const residual=(layer:number,coordinate:number,row:number)=>dependency(`residual:${layer}:${row}:${coordinate}`,async()=>{
      const a=layers[layer]!.attention.attention;
      const attention=await linear(a.projections.o,coordinate,column=>
        context(layer,Math.floor(column/a.headDim),column%a.headDim,row));
      return half(add(widen(await hidden(layer-1,coordinate,row)),widen(attention)));
    });
    const expression=await linear({weight:output.weight,shape:output.shape},dimension,column=>
      norm(`final:${position}`,output.finalNormWeight,output.finalNormEpsilon,column,
        index=>hidden(layers.length-1,index,position)));
    return {expression,stats,facts};
  }
  return {header,discovered,build,close:async()=>{pages.clear();await reader.close();}};
  } catch(error){pages.clear();await reader.close();throw error;}
}
