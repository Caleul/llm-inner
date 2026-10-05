import {validateDirectModelSnapshot,type DirectModelSnapshot} from './direct-model-snapshot.js';
import { withDirectCompilationLease } from "./direct-compilation-lease.js";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { DirectWeightPages } from "./direct-weight-pages.js";
import { directCompilationUnits, directUnitConditions, directRustHeader, directRustFooter, validateDirectUnit, type DirectCompilationUnit } from "./direct-compilation-units.js";
import { discoverDirectOutput } from "./direct-output-row.js";
import { discoverDirectMlp, type Projection } from "./direct-mlp-output.js";
import { discoverDirectAttention } from "./direct-attention-output.js";
import { DirectRustStream } from "./direct-rust-stream.js";
import { DirectFlatSubstitution, FlatConditions, type FlatProducer, type FlatInput } from "./direct-flat-substitution.js";
import { substituteFlatProjection, substituteFlatReduction } from "./direct-flat-projection.js";
import { substituteCpuArm64F32Sum } from "./direct-rust-mean.js";
import { substituteCpuArm64SoftmaxSum } from "./direct-rust-softmax.js";
import {proveInitialHiddenIdentity,proveInitialMlpProductZero} from "./direct-flat-zero-mlp.js";
import {proveFiniteNormalizationInputs} from "./direct-finite-model-proof.js";
import { rational } from "./direct-branch-domain.js";
import { fixedF16RopeLiteral } from "./fixed-f16-rope-branches.js";
import { decodeIeeeF16ToF32 } from "./utils.js";

/** Recursive substitution directly from discovered source semantics and scalar
 * Safetensors reads. Callbacks are invoked on the active path; no model IR,
 * expression nodes, activation cache or final-output lookup is constructed.
 * Runtime input is a variable-length matrix of finite F16 embedding values,
 * exactly widened to f64. Position is a parameter, not a validation prompt.
 */
export interface DirectModelDiscovery {
  output: Awaited<ReturnType<typeof discoverDirectOutput>>;
  /** Checkpoint-wide proof facts only; no per-coordinate bounds or values. */
  proof?: {scoreBounds:number[]};
  layers: {mlp:Awaited<ReturnType<typeof discoverDirectMlp>>;attention:Awaited<ReturnType<typeof discoverDirectAttention>>}[];
}
export interface DirectCompileResources { weightCacheBytes?:number; maxOutputBytes?:number; outputCounter?:SharedArrayBuffer }
export async function discoverDirectModel(directory:string,python:string):Promise<DirectModelDiscovery>{
  const output=await discoverDirectOutput(directory,python);
  if(output.torch.split("+")[0]!=="2.12.1"||process.arch!=="arm64")
    throw new Error("Direct numerical policy is defined for PyTorch 2.12.1 CPU arm64");
  const layers:DirectModelDiscovery["layers"]=[];
  for(let index=0;index<output.decoderLayers.length;index++)layers.push({
    mlp:await discoverDirectMlp(directory,python,index),attention:await discoverDirectAttention(directory,python,index)});
  return {output,layers};
}
export async function writeDirectFlatRustModel(directory:string,python:string,dimension:number,path:string,
  resources:DirectCompileResources={}):Promise<void>{
  return withDirectCompilationLease(path,async()=>
    compileDirectFlatRust(directory,dimension,path,await discoverDirectModel(directory,python),resources));
}
/** Internal fragments are never admitted as standalone models. */
export async function writeDirectFlatRustUnit(directory:string,dimension:number,path:string,
  discovered:DirectModelDiscovery,unit:DirectCompilationUnit,resources:DirectCompileResources={}):Promise<void>{
  return compileDirectFlatRust(directory,dimension,path,discovered,resources,unit);
}
async function compileDirectFlatRust(directory:string,dimension:number,path:string,
  {output,layers,proof}:DirectModelDiscovery,resources:DirectCompileResources,unit?:DirectCompilationUnit):Promise<void>{
  if(output.torch.split("+")[0]!=="2.12.1"||process.arch!=="arm64")
    throw new Error("Direct numerical policy is defined for PyTorch 2.12.1 CPU arm64");
  if(!Number.isSafeInteger(dimension)||dimension<0||dimension>=output.shape[0])throw new RangeError("Invalid logit dimension");
  if(unit)validateDirectUnit(unit,output.maxPosition,output.shape[1]);
  if(resources.maxOutputBytes!==undefined&&(!Number.isSafeInteger(resources.maxOutputBytes)||resources.maxOutputBytes<1))
    throw new RangeError("Invalid output byte limit");
  if(resources.weightCacheBytes!==undefined&&(!Number.isSafeInteger(resources.weightCacheBytes)||resources.weightCacheBytes<0))
    throw new RangeError("Invalid weight page budget");
  const reader=new SafetensorsCatalogReader(directory),draft=path+".draft";
  await mkdir(dirname(path),{recursive:true});
  const s=new DirectRustStream(draft),f=new DirectFlatSubstitution(s),width=output.shape[1];
  s.setSharedByteLimit(resources.outputCounter,resources.maxOutputBytes);
  const pages=new DirectWeightPages(reader,resources.weightCacheBytes);
  const interrupt=()=>s.cancel("Generation interrupted; draft is not an executable artifact");
  process.on("SIGINT",interrupt);
  const started=Date.now();let leaves=0,position=0,progressWrite=Promise.resolve(),status="generating";
  const snapshot=()=>JSON.stringify({status,dimension,position,leaves:f.emittedLeaves,candidateLeaves:leaves,bytes:s.bytes,
    eliminatedBranches:s.eliminatedBranches,inspectedExpressions:s.inspectedExpressions,
    elapsedMilliseconds:Date.now()-started,finalParity:false,rustCompilationAdmitted:!unit&&status==="emitted",weightReads:pages.reads,weightPageHits:pages.hits,weightPageBytes:pages.retainedBytes},null,2)+"\n";
  const progress=setInterval(()=>{
    progressWrite=progressWrite.then(()=>writeFile(path+".progress.json",snapshot()));
  },5000);progress.unref();
  try{
    const catalog=await reader.inspect();
    const weight=async(name:string,index:number)=>{
      const tensor=catalog.tensors.get(name);
      if(!tensor||tensor.storageDtype!=="F16")throw new Error(`Unsupported weight dtype: ${name}`);
      const value=await pages.read(tensor,index);
      if(!Number.isFinite(value))throw new Error(`Nonfinite checkpoint weight: ${name}[${index}]`);
      return value;
    };
    const verify=(name:string,shape:readonly number[])=>{
      const tensor=catalog.tensors.get(name);
      if(!tensor||tensor.storageDtype!=="F16"||tensor.logicalShape.length!==shape.length||
        tensor.logicalShape.some((value,index)=>value!==shape[index]))throw new Error(`Checkpoint shape mismatch: ${name}`);
    };
    verify(output.weight,output.shape);verify(output.finalNormWeight,[width]);
    for(const layer of layers){
      for(const p of Object.values(layer.mlp.mlp))verify(p.weight,p.shape);
      for(const p of Object.values(layer.attention.attention.projections))verify(p.weight,p.shape);
      for(const n of layer.mlp.normalizations)verify(n.weight,[width]);
      const a=layer.attention.attention;
      if(a.headDim%2||a.heads%a.kvHeads||a.projections.o.shape[0]!==width||layer.mlp.normalizations.length!==2)
        throw new Error("Unsupported discovered geometry");
    }
    const literal=(value:number):FlatProducer=>(p,k)=>f.literal(p,value,k);
    const binary=(a:FlatProducer,b:FlatProducer,operator:"+"|"-"|"*"|"/"):FlatProducer=>(p,k)=>f.binary(p,a,b,operator,k);
    const round=(input:FlatProducer,kind:"f16"|"f32",alreadyF32=false):FlatProducer=>(p,k)=>f.round(p,input,kind,alreadyF32,k);
    const bounded=(input:FlatProducer,minimum:number,maximum:number):FlatProducer=>(p,k)=>input(p,(p,v)=>{
      const low=Math.max(v.minimum,minimum),high=Math.min(v.maximum,maximum);
      if(low>high){s.eliminatedBranches++;return Promise.resolve();}
      return k(p,{...v,minimum:low,maximum:high});
    });
    const linear=(projection:Projection,row:number,input:(coordinate:number)=>FlatProducer,
      zeroInputMagnitude?:(path:FlatConditions,coordinate:number)=>Promise<boolean>):FlatProducer=>(p,k)=>
      substituteFlatProjection(f,p,projection.shape[1],c=>weight(projection.weight,row*projection.shape[1]+c),input,k,zeroInputMagnitude);
    let allNormalizationInputsFinite=false;
    const norm=(name:string,epsilon:number,coordinate:number,input:(coordinate:number)=>FlatProducer,
      finiteEmbeddingInput=false):FlatProducer=>async(p,k)=>{
      if(width>1000000||!Number.isFinite(Math.fround(epsilon))||!(Math.fround(epsilon)>=2**-126))
        throw new Error("Normalization finite-error domain not proved");
      const gamma=await weight(name,coordinate);
      const evaluate=(p:FlatConditions,coordinateInput:FlatProducer)=>{
      // This coordinate was already substituted on the active consumer path.
      // Expand its arithmetic again in the emitted sum, but do not revisit its
      // numerical choices. Later constraints refresh the reached operand.
      const sum:FlatProducer=(p,k)=>substituteCpuArm64F32Sum(f,p,width,c=>(p,k)=>
        f.square(p,c===coordinate?coordinateInput:input(c),k),k);
      const variance=round(binary(round(binary(sum,literal(width),"/"),"f32"),literal(Math.fround(epsilon)),"+"),"f32");
      const root:FlatProducer=(p,k)=>f.sqrt(p,variance,k);
      const inverse=round(binary(literal(1),root,"/"),"f32");
      // Nonnegative RMS reduction: accumulated F32 error <1/8 in this domain.
      // The factor two conservatively covers sum, rsqrt and F16 rounding.
      const product=bounded(binary(coordinateInput,inverse,"*"),-2*Math.sqrt(width),2*Math.sqrt(width));
      const normalized=round(round(product,"f32"),"f16",true);
      return round(binary(normalized,literal(gamma),"*"),"f16",true)(p,k);
      };
      if(!finiteEmbeddingInput&&!allNormalizationInputsFinite)return evaluate(p,input(coordinate));
      return input(coordinate)(p,(p,value)=>{
        const reached:FlatProducer=(p,k)=>k(p,value);
        // Embeddings are finite by the input contract; subsequent inputs
        // are finite by the checkpoint-wide proof. Positive finite epsilon
        // gives a positive finite reciprocal. Zero coordinates or a zero
        // learned weight therefore need only the coordinate's original sign.
        if((value.minimum===0&&value.maximum===0)||gamma===0){
          s.eliminatedBranches++;
          return round(binary(reached,literal(gamma),"*"),"f16",true)(p,k);
        }
        return evaluate(p,reached);
      });
    };
    const embeddings=(coordinate:number,position:number):FlatProducer=>(p,k)=>f.f16Input(p,
      ()=>s.write(`input_tokens[${position}][${coordinate}]`),k);
    const preNorm=(layer:number,coordinate:number,position:number):FlatProducer=>{
      const n=layers[layer]!.mlp.normalizations[0]!;
      return norm(n.weight,n.epsilon,coordinate,c=>hidden(layer-1,c,position),layer===0);
    };
    const postNorm=(layer:number,coordinate:number,position:number):FlatProducer=>{
      const n=layers[layer]!.mlp.normalizations[1]!;
      return norm(n.weight,n.epsilon,coordinate,c=>residual(layer,c,position));
    };
    const projected=(layer:number,role:"q"|"k"|"v",row:number,position:number):FlatProducer=>
      linear(layers[layer]!.attention.attention.projections[role],row,c=>preNorm(layer,c,position));
    const rotated=(layer:number,role:"q"|"k",head:number,coordinate:number,position:number):FlatProducer=>{
      const a=layers[layer]!.attention.attention,half=a.headDim/2;
      const h=role==="q"?head:Math.floor(head/(a.heads/a.kvHeads));
      const cos=decodeIeeeF16ToF32(fixedF16RopeLiteral(position,coordinate,a.headDim,a.ropeTheta,0));
      const sin=decodeIeeeF16ToF32(fixedF16RopeLiteral(position,coordinate,a.headDim,a.ropeTheta,1));
      const q=round(binary(projected(layer,role,h*a.headDim+coordinate,position),literal(cos),"*"),"f16",true);
      const other=binary(projected(layer,role,h*a.headDim+(coordinate+half)%a.headDim,position),literal(coordinate<half?-sin:sin),"*");
      return round(binary(q,round(other,"f16",true),"+"),"f16");
    };
    const score=(layer:number,head:number,query:number,key:number):FlatProducer=>{
      const a=layers[layer]!.attention.attention;
      const dot:FlatProducer=(p,k)=>substituteFlatReduction(f,p,a.headDim,c=>
        binary(rotated(layer,"q",head,c,query),rotated(layer,"k",head,c,key),"*"),k);
      return round(binary(dot,literal(Math.fround(a.scaling)),"*"),"f16");
    };
    const scoreBounds=proof?.scoreBounds??await proveDirectNumerics(output,layers,width,weight);
    if(scoreBounds.length!==layers.length)throw new Error("Invalid numerical proof certificate");
    allNormalizationInputsFinite=true;
    let fullVectorSoftmax=false;
    const context=(layer:number,head:number,coordinate:number,query:number):FlatProducer=>{
      const a=layers[layer]!.attention.attention;
      const v=(key:number)=>projected(layer,"v",Math.floor(head/(a.heads/a.kvHeads))*a.headDim+coordinate,key);
      if(query===0)return binary(literal(0),v(0),"+");
      const maximum=(last:number):FlatProducer=>(p,k)=>{
        if(last===0)return score(layer,head,query,0)(p,k);
        return maximum(last-1)(p,(p,previous)=>score(layer,head,query,last)(p,(p,current)=>{
          const left:FlatProducer=(p,k)=>k(p,current),right:FlatProducer=(p,k)=>k(p,previous);
          return f.comparison(p,binary(left,right,"-"),">",rational(0n),p=>k(p,current),p=>k(p,previous));
        }));
      };
      const differenceBound=2*scoreBounds[layer]!*1.01+2**-149;
      const exponent=(key:number):FlatProducer=>(p,k)=>f.exponential(p,
        bounded(round(binary(score(layer,head,query,key),maximum(query),"-"),"f32"),-differenceBound,0),k,2**-24);
      const denominator:FlatProducer=(p,k)=>substituteCpuArm64SoftmaxSum(f,p,query+1,fullVectorSoftmax,exponent,k);
      const reciprocal=round(binary(literal(1),denominator,"/"),"f32");
      const probability=(key:number)=>round(round(binary(exponent(key),reciprocal,"*"),"f32"),"f16",true);
      return (p,k)=>substituteFlatReduction(f,p,query+1,key=>binary(probability(key),v(key),"*"),k);
    };
    const residual=(layer:number,coordinate:number,position:number):FlatProducer=>{
      const a=layers[layer]!.attention.attention;
      return round(binary(hidden(layer-1,coordinate,position),linear(a.projections.o,coordinate,c=>
        context(layer,Math.floor(c/a.headDim),c%a.headDim,position)),"+"),"f16");
    };
    const hidden=(layer:number,coordinate:number,position:number):FlatProducer=>{
      if(layer<0)return embeddings(coordinate,position);
      const p=layers[layer]!.mlp.mlp;
      const down=linear(p.down,coordinate,neuron=>{
        const gate=linear(p.gate,neuron,c=>postNorm(layer,c,position));
        const activation:FlatProducer=(p,k)=>f.silu(p,gate,k);
        const up=linear(p.up,neuron,c=>postNorm(layer,c,position));
        return round(binary(activation,up,"*"),"f16",true);
      },layer===0?(path,neuron)=>{
        const a=layers[0]!.attention.attention,n=layers[0]!.mlp.normalizations;
        if(!(a.ropeTheta>=1)||width>1000000||a.headDim>1000000)return Promise.resolve(false);
        return proveInitialMlpProductZero(f,path,{width,context:output.maxPosition,heads:a.heads,kvHeads:a.kvHeads,
          headDim:a.headDim,pre:n[0]!,post:n[1]!,v:a.projections.v,o:a.projections.o,gate:p.gate,up:p.up},
          position,neuron,weight);
      }:undefined);
      const expanded=round(binary(residual(layer,coordinate,position),down,"+"),"f16");
      if(layer!==0)return expanded;
      return async(path,k)=>{
        const a=layers[0]!.attention.attention,n=layers[0]!.mlp.normalizations;
        if(await proveInitialHiddenIdentity(f,path,{width,context:output.maxPosition,heads:a.heads,kvHeads:a.kvHeads,
          headDim:a.headDim,pre:n[0]!,post:n[1]!,v:a.projections.v,o:a.projections.o,
          gate:p.gate,up:p.up,down:p.down},position,coordinate,weight)){
          s.eliminatedBranches++;
          return binary(embeddings(coordinate,position),literal(0),"+")(path,k);
        }
        return expanded(path,k);
      };
    };
    if(!unit)await s.write(directRustHeader(width,output.maxPosition));
    s.beginReducedExpression();
    for(const current of unit?[unit]:directCompilationUnits(output.maxPosition,width)){
      position=current.position;fullVectorSoftmax=current.fullVectorSoftmax;
      const pathConditions=directUnitConditions(current,output.maxPosition,width,s);
      const result=linear({weight:output.weight,shape:output.shape},dimension,c=>
        norm(output.finalNormWeight,output.finalNormEpsilon,c,column=>hidden(layers.length-1,column,position)));
      await result(pathConditions,async(p,value:FlatInput)=>{await f.leaf(p,"result",value);leaves++;});
      await f.finishRound();
    }
    if(!unit)await s.write(directRustFooter);
    await s.close();await rename(draft,path);
    status=unit?"fragment":"emitted";
    await writeFile(path+".reduction.json",JSON.stringify({status,dimension,inputWidth:width,context:output.maxPosition,unit,
      weightReads:pages.reads,weightPageHits:pages.hits,weightPageBytes:pages.retainedBytes,
      leaves:f.emittedLeaves,candidateLeaves:leaves,bytes:s.bytes,
      numericalPolicy:{torch:output.torch,backend:"cpu-arm64",weights:"finite-f16",
        rounding:"nearest-even F32/F16; ordered source reductions"},
      eliminatedBranches:s.eliminatedBranches,input:"finite-f16-embedding-matrix",runtimeIR:false,finalParity:false},null,2)+"\n");
  }catch(error){
    status="pending";
    s.destroy();await writeFile(path+".reduction.json",JSON.stringify({status:"pending",dimension,
      bytes:s.bytes,reason:error instanceof Error?error.message:String(error),rustCompilationAdmitted:false},null,2)+"\n");throw error;
  }finally{
    process.off("SIGINT",interrupt);clearInterval(progress);await progressWrite;
    await writeFile(path+".progress.json",snapshot());pages.clear();await reader.close();
  }
}

async function proveDirectNumerics(output:DirectModelDiscovery["output"],layers:DirectModelDiscovery["layers"],
  width:number,weight:(name:string,index:number)=>Promise<number>):Promise<number[]>{
    // Confirm masked cells underflow to zero using checkpoint-derived bounds.
    const scoreBounds:number[]=[];
    for(const layer of layers){
      const a=layer.attention.attention;let gamma=0;
      for(let c=0;c<width;c++)gamma=Math.max(gamma,Math.abs(await weight(layer.mlp.normalizations[0]!.weight,c)));
      const projectionBound=async(role:"q"|"k")=>{
        const projection=a.projections[role];let maximum=0;
        for(let row=0;row<projection.shape[0];row++){
          let squared=0,absolute=0;
          for(let c=0;c<width;c++){const w=await weight(projection.weight,row*width+c);squared+=w*w;absolute+=Math.abs(w);}
          maximum=Math.max(maximum,(2*Math.sqrt(width)*gamma*Math.sqrt(squared)+absolute*2**-23)*1.125+2**-24);
        }
        return maximum;
      };
      const q=await projectionBound("q"),k=await projectionBound("k");
      const rotatedQ=2*q*1.01+2**-23,rotatedK=2*k*1.01+2**-23;
      const bound=(a.headDim*rotatedQ*rotatedK*1.125+2**-24)*Math.abs(a.scaling)*1.01+2**-24;
      if(!(a.headDim<=1000000&&bound<16&&2*Math.sqrt(width)*gamma<65504&&
        rotatedQ<65504&&rotatedK<65504))throw new Error("Finite causal-mask elimination not proved");
      scoreBounds.push(bound);
    }
    function* finiteLayerMetadata(){for(const layer of layers){
      const a=layer.attention.attention,n=layer.mlp.normalizations,p=layer.mlp.mlp;
      yield {pre:n[0]!,post:n[1]!,heads:a.heads,kvHeads:a.kvHeads,headDim:a.headDim,ropeTheta:a.ropeTheta,
        v:a.projections.v,o:a.projections.o,gate:p.gate,up:p.up,down:p.down};
    }}
    const finite=await proveFiniteNormalizationInputs(width,output.maxPosition,finiteLayerMetadata(),
      {weight:output.finalNormWeight,epsilon:output.finalNormEpsilon},weight,scoreBounds.length===layers.length);
    if(!finite)throw new Error("Finite normalization inputs are not proved for this checkpoint and embedding domain");
    return scoreBounds;
}
/** Discover and prove once for the pool. Certificates contain only a bounded
 * score range per layer; they are compile-time facts and never emitted. */
export async function prepareDirectModel(directory:string,python:string,weightCacheBytes:number,snapshot?:DirectModelSnapshot):Promise<DirectModelDiscovery>{
  const discovered=snapshot?await validateDirectModelSnapshot(directory,snapshot):await discoverDirectModel(directory,python),reader=new SafetensorsCatalogReader(directory);
  const pages=new DirectWeightPages(reader,weightCacheBytes);
  try{
    const catalog=await reader.inspect();
    const weight=async(name:string,index:number)=>{
      const tensor=catalog.tensors.get(name);if(!tensor)throw new Error(`Missing checkpoint tensor: ${name}`);
      return pages.read(tensor,index);
    };
    return {...discovered,proof:{scoreBounds:await proveDirectNumerics(discovered.output,discovered.layers,discovered.output.shape[1],weight)}};
  }finally{pages.clear();await reader.close();}
}
