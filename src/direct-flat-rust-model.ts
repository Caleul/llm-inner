import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { readDirectF16Literal } from "./direct-weight-literal.js";
import { discoverDirectOutput } from "./direct-output-row.js";
import { discoverDirectMlp, type Projection } from "./direct-mlp-output.js";
import { discoverDirectAttention } from "./direct-attention-output.js";
import { DirectRustStream } from "./direct-rust-stream.js";
import { DirectFlatSubstitution, FlatConditions, type FlatProducer, type FlatInput } from "./direct-flat-substitution.js";
import { substituteFlatProjection, substituteFlatReduction } from "./direct-flat-projection.js";
import { substituteCpuArm64F32Sum } from "./direct-rust-mean.js";
import { rational } from "./direct-branch-domain.js";
import { fixedF16RopeLiteral } from "./fixed-f16-rope-branches.js";
import { decodeIeeeF16ToF32 } from "./utils.js";

/** Recursive substitution directly from discovered source semantics and scalar
 * Safetensors reads. Callbacks are invoked on the active path; no model IR,
 * expression nodes, activation cache or final-output lookup is constructed.
 * Runtime input is a variable-length matrix of finite F16 embedding values,
 * exactly widened to f64. Position is a parameter, not a validation prompt.
 */
export async function writeDirectFlatRustModel(directory:string,python:string,dimension:number,path:string):Promise<void>{
  const output=await discoverDirectOutput(directory,python);
  if(!Number.isSafeInteger(dimension)||dimension<0||dimension>=output.shape[0])throw new RangeError("Invalid logit dimension");
  const layers:{mlp:Awaited<ReturnType<typeof discoverDirectMlp>>;attention:Awaited<ReturnType<typeof discoverDirectAttention>>}[]=[];
  for(let index=0;index<output.decoderLayers.length;index++)layers.push({
    mlp:await discoverDirectMlp(directory,python,index),attention:await discoverDirectAttention(directory,python,index)});
  const reader=new SafetensorsCatalogReader(directory),draft=path+".draft";
  await mkdir(dirname(path),{recursive:true});
  const s=new DirectRustStream(draft),f=new DirectFlatSubstitution(s),width=output.shape[1];
  const interrupt=()=>s.cancel("Generation interrupted; draft is not an executable artifact");
  process.on("SIGINT",interrupt);
  const started=Date.now();let leaves=0,position=0,progressWrite=Promise.resolve(),status="generating";
  const snapshot=()=>JSON.stringify({status,dimension,position,leaves,bytes:s.bytes,
    eliminatedBranches:s.eliminatedBranches,inspectedExpressions:s.inspectedExpressions,
    elapsedMilliseconds:Date.now()-started,finalParity:false,rustCompilationAdmitted:status==="emitted"},null,2)+"\n";
  const progress=setInterval(()=>{
    progressWrite=progressWrite.then(()=>writeFile(path+".progress.json",snapshot()));
  },5000);progress.unref();
  try{
    const catalog=await reader.inspect();
    const weight=async(name:string,index:number)=>{
      const tensor=catalog.tensors.get(name);
      if(!tensor||tensor.storageDtype!=="F16")throw new Error(`Unsupported weight dtype: ${name}`);
      const value=Number(await readDirectF16Literal(reader,tensor,index));
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
    const linear=(projection:Projection,row:number,input:(coordinate:number)=>FlatProducer):FlatProducer=>(p,k)=>
      substituteFlatProjection(f,p,projection.shape[1],c=>weight(projection.weight,row*projection.shape[1]+c),input,k);
    const norm=(name:string,epsilon:number,coordinate:number,input:(coordinate:number)=>FlatProducer):FlatProducer=>async(p,k)=>{
      if(width>1000000||!(Math.fround(epsilon)>=2**-126))throw new Error("Normalization finite-error domain not proved");
      const sum:FlatProducer=(p,k)=>substituteCpuArm64F32Sum(f,p,width,c=>(p,k)=>f.square(p,input(c),k),k);
      const variance=round(binary(round(binary(sum,literal(width),"/"),"f32"),literal(Math.fround(epsilon)),"+"),"f32");
      const root:FlatProducer=(p,k)=>f.sqrt(p,variance,k);
      const inverse=round(binary(literal(1),root,"/"),"f32");
      // Nonnegative RMS reduction: accumulated F32 error <1/8 in this domain.
      // The factor two conservatively covers sum, rsqrt and F16 rounding.
      const product=bounded(binary(input(coordinate),inverse,"*"),-2*Math.sqrt(width),2*Math.sqrt(width));
      const normalized=round(round(product,"f32"),"f16",true);
      const gamma=await weight(name,coordinate);
      return round(binary(normalized,literal(gamma),"*"),"f16",true)(p,k);
    };
    const embeddings=(coordinate:number,position:number):FlatProducer=>(p,k)=>f.f16Input(p,
      ()=>s.write(`input_tokens[${position}][${coordinate}]`),k);
    const preNorm=(layer:number,coordinate:number,position:number):FlatProducer=>{
      const n=layers[layer]!.mlp.normalizations[0]!;
      return norm(n.weight,n.epsilon,coordinate,c=>hidden(layer-1,c,position));
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
          maximum=Math.max(maximum,(2*Math.sqrt(width)*gamma*Math.sqrt(squared)+absolute*2**-23)*1.01+2**-24);
        }
        return maximum;
      };
      const q=await projectionBound("q"),k=await projectionBound("k");
      const bound=(a.headDim*(2*q*1.01+2**-23)*(2*k*1.01+2**-23)*1.01+2**-24)*Math.abs(a.scaling)*1.01+2**-24;
      if(!(bound<16&&2*Math.sqrt(width)*gamma<65504&&q<65504&&k<65504))throw new Error("Finite causal-mask elimination not proved");
      scoreBounds.push(bound);
    }
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
      // softmax_lastdim's CPU F32 reduce_all accumulates four lanes,
      // including a partial final vector, then combines lanes sequentially.
      // Masked trailing zeros do not change this order for any valid n.
      const lane=(index:number,last:number):FlatProducer=>last<index?literal(0):last===index?exponent(last):
        round(binary(lane(index,last-4),exponent(last),"+"),"f32");
      const denominator=(lastLane:number):FlatProducer=>lastLane<0?literal(0):
        round(binary(denominator(lastLane-1),lane(lastLane,lastLane+4*Math.floor((query-lastLane)/4)),"+"),"f32");
      const reciprocal=round(binary(literal(1),denominator(3),"/"),"f32");
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
      });
      return round(binary(residual(layer,coordinate,position),down,"+"),"f16");
    };
    await s.write(`// Input: finite F16 embedding matrix, widened exactly to f64.\n// Policy: PyTorch 2.12.1 CPU arm64; scalar substitution, flat path conditions.\n#![recursion_limit="65536"]\npub fn compiled_dimension(input_tokens:&[Vec<f64>],t:usize)->f64 {let n=input_tokens.len();assert!(n>0 && n<=${output.maxPosition} && t<n);'result:{`);
    s.beginReducedExpression();
    for(position=0;position<output.maxPosition;position++){
      const currentPosition=position;
      const pathConditions=new FlatConditions(undefined,[],[()=>s.write(`t==${currentPosition}`)]);
      const result=linear({weight:output.weight,shape:output.shape},dimension,c=>
        norm(output.finalNormWeight,output.finalNormEpsilon,c,column=>hidden(layers.length-1,column,position)));
      await result(pathConditions,async(p,value:FlatInput)=>{await f.leaf(p,"result",value);leaves++;});
      await f.finishRound();
    }
    await s.write('panic!("outside declared embedding domain")}}\n');await s.close();await rename(draft,path);
    status="emitted";
    await writeFile(path+".reduction.json",JSON.stringify({status:"emitted",dimension,leaves,bytes:s.bytes,
      eliminatedBranches:s.eliminatedBranches,input:"finite-f16-embedding-matrix",runtimeIR:false,finalParity:false},null,2)+"\n");
  }catch(error){
    status="pending";
    s.destroy();await writeFile(path+".reduction.json",JSON.stringify({status:"pending",dimension,
      bytes:s.bytes,reason:error instanceof Error?error.message:String(error),rustCompilationAdmitted:false},null,2)+"\n");throw error;
  }finally{
    process.off("SIGINT",interrupt);clearInterval(progress);await progressWrite;
    await writeFile(path+".progress.json",snapshot());await reader.close();
  }
}
