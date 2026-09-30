import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { SafetensorsCatalogReader } from "./safetensors.js";
import type { TensorInfo } from "./types.js";
import { readDirectF16Literal } from "./direct-weight-literal.js";
import { discoverDirectOutput } from "./direct-output-row.js";
import { discoverDirectMlp, type Projection } from "./direct-mlp-output.js";
import { discoverDirectAttention } from "./direct-attention-output.js";
import { DirectBranchDomain, rational } from "./direct-branch-domain.js";
import { DirectRustStream, rustF64, type RustExpression } from "./direct-rust-stream.js";
import { emitCpuArm64F32Sum, foldCpuArm64F32Sum } from "./direct-rust-mean.js";
import { emitRustExp, emitRustSilu, emitRustSqrt } from "./direct-rust-numeric.js";
import { f32BitsToDyadic, roundDyadicToF16IfElse } from "./fixed-f16-projection.js";
import { decodeIeeeF16ToF32 } from "./utils.js";
import { fixedF16RopeLiteral } from "./fixed-f16-rope-branches.js";

/** Recursive source emission: callbacks write their dependencies immediately.
 * No calculation graph, expression cache, stage program or runtime model is built.
 */
export async function writeDirectRustModel(directory:string,python:string,dimension:number,path:string):Promise<void>{
  const output=await discoverDirectOutput(directory,python);
  if(!Number.isSafeInteger(dimension)||dimension<0||dimension>=output.shape[0])throw new RangeError("Invalid logit dimension");
  const layers: {mlp: Awaited<ReturnType<typeof discoverDirectMlp>>; attention: Awaited<ReturnType<typeof discoverDirectAttention>>}[]=[];
  for(let index=0;index<output.decoderLayers.length;index++) {
    layers.push({mlp:await discoverDirectMlp(directory,python,index),attention:await discoverDirectAttention(directory,python,index)});
  }
  const reader=new SafetensorsCatalogReader(directory);
  await mkdir(dirname(path),{recursive:true});const draft=path+".draft",s=new DirectRustStream(draft);
  try {
    const catalog=await reader.inspect();
    const tensor=(name:string):TensorInfo=>{
      const value=catalog.tensors.get(name);
      if(!value||value.storageDtype!=="F16")throw new Error(`Undefined numeric storage: ${name}`);
      return value;
    };
    const verify=(name:string,shape:readonly number[])=>{
      const actual=tensor(name).logicalShape;
      if(actual.length!==shape.length||actual.some((value,index)=>value!==shape[index]))throw new Error(`Checkpoint/config shape mismatch: ${name}`);
    };
    verify(output.weight,output.shape);verify(output.embeddingWeight,output.embeddingShape);
    verify(output.finalNormWeight,[output.shape[1]]);
    for(const layer of layers){
      for(const p of Object.values(layer.mlp.mlp))verify(p.weight,p.shape);
      for(const p of Object.values(layer.attention.attention.projections))verify(p.weight,p.shape);
      for(const n of layer.mlp.normalizations)verify(n.weight,[output.shape[1]]);
    }
    const weight=async(name:string,index:number)=>rustF64(await readDirectF16Literal(reader,tensor(name),index));
    const width=output.shape[1];
    // This is the explicitly declared scalar four-lane CPU reduction policy.
    // Its equivalence to a particular PyTorch shape/backend must be validated.
    const reduction=async(size:number,term:(coordinate:number)=>Promise<void>):Promise<void>=>{
      const laneSum=async(lane:number,last:number):Promise<void>=>{
        if(last<lane){await s.write("0.0");return;}
        // Products of two finite F16 values have at most 22 significant bits:
        // widening and the first F32 accumulation are exact, including +0.
        if(last===lane){await s.write("(0.0+");await term(last);await s.write(")");return;}
        await s.round("f32",async()=>{await laneSum(lane,last-4);await s.write("+");await term(last);});
      };
      const pair=async(index:number):Promise<void>=>{
        const first=index*2,second=first+1;
        const emit=async()=>{
          await laneSum(first,first+4*Math.floor((size-1-first)/4));await s.write("+");
          await laneSum(second,second+4*Math.floor((size-1-second)/4));
        };
        if(second>=size){await s.write("(");await emit();await s.write(")");}
        else await s.round("f32",emit);
      };
      await s.round("f16",async()=>{
        const emit=async()=>{await pair(0);await s.write("+");await pair(1);};
        if(size<=2){await s.write("(");await emit();await s.write(")");}
        else await s.round("f32",emit);
      },true);
    };
    const linear=async(projection:Projection,row:number,input:(coordinate:number)=>Promise<void>)=>{
      if(row<0||row>=projection.shape[0]||projection.shape[1]<=0)throw new Error("Invalid discovered projection");
      await reduction(projection.shape[1],async coordinate=>{
        // Preserve zero terms until the input's finite/signed-zero domain proves removal safe.
        await input(coordinate);await s.write(`*${await weight(projection.weight,row*projection.shape[1]+coordinate)}`);
      });
    };
    const embedding=async(coordinate:number,position:string)=>{
      await s.write("'embedding:{");
      // Consecutive equal literals are combined before emitting token conditions.
      let from=0,previous:string|undefined;
      const flush=async(end:number)=>{
        if(previous===undefined)return;
        const condition=from===end?`input_tokens[${position}]==${from}`:
          `input_tokens[${position}]>=${from} && input_tokens[${position}]<=${end}`;
        await s.write(`if ${condition} {break 'embedding ${previous};}`);
      };
      for(let token=0;token<output.embeddingShape[0];token++) {
        const literal=await weight(output.embeddingWeight,token*output.embeddingShape[1]+coordinate);
        if(literal!==previous){await flush(token-1);from=token;previous=literal;}
      }
      await flush(output.embeddingShape[0]-1);await s.write("panic!(\"token out of vocabulary\")}");
    };
    const norm=async(name:string,epsilon:number,coordinate:number,input:(coordinate:number)=>Promise<void>)=>{
      const eps=Math.fround(epsilon),varianceMaximum=4*65504**2+2*eps;
      // A sum of at most 10^6 nonnegative F16 squares incurs <1/8 relative
      // F32 error. Division, epsilon addition and rounding stay strictly below
      // this doubled envelope. The F32 epsilon is an exactly representable lower bound.
      const varianceRange=width<=1000000 && eps>=2**-126 && varianceMaximum<3.4028234663852886e38?
        {minimum:eps,maximum:varianceMaximum}:undefined;
      const inverseRange=varianceRange?{minimum:0.25/Math.sqrt(varianceMaximum),maximum:4/Math.sqrt(eps)}:undefined;
      await s.round("f16",async()=>{
        await s.round("f16",async()=>s.round("f32",async()=>{
          await input(coordinate);await s.write("*");
          await s.round("f32",async()=>{
            await s.write("1.0/");await emitRustSqrt(s,async()=>s.round("f32",async()=>{
              await s.round("f32",async()=>{
                await emitCpuArm64F32Sum(s,width,async c=>{await input(c);await s.write("*");await input(c);});await s.write(`/${width}.0`);
              });await s.write(`+${rustF64(eps)}`);
            },false,varianceRange),varianceRange);
          },false,inverseRange);
        }),true);await s.write(`*${await weight(name,coordinate)}`);
      },true);
    };
    const hidden=async(layer:number,coordinate:number,position:string):Promise<void>=>{
      if(layer<0){await embedding(coordinate,position);return;}
      await s.round("f16",async()=>{
        await residual(layer,coordinate,position);await s.write("+");
        const p=layers[layer]!.mlp.mlp;
        await linear(p.down,coordinate,async neuron=>{
          let gateBound=0,normBound=0;
          const normalization=layers[layer]!.mlp.normalizations[1]!;
          for(let c=0;c<width;c++){
            const gamma=Math.abs(Number(await readDirectF16Literal(reader,tensor(normalization.weight),c)));
            const bound=2*Math.sqrt(width)*gamma+2**-23;
            normBound=Math.max(normBound,bound);
            const w=Math.abs(Number(await readDirectF16Literal(reader,tensor(p.gate.weight),neuron*width+c)));
            gateBound+=bound*w;
          }
          gateBound=gateBound*1.01+2**-24;
          const finiteGateBound=width<=1000000 && normBound<65504 && gateBound<65504?gateBound:undefined;
          await s.round("f16",async()=>{
            await emitRustSilu(s,()=>linear(p.gate,neuron,column=>postNorm(layer,column,position)),finiteGateBound);
            await s.write("*");await linear(p.up,neuron,column=>postNorm(layer,column,position));
          },true);
        });
      });
    };
    const postNorm=async(layer:number,coordinate:number,position:string)=>{
      const n=layers[layer]!.mlp.normalizations[1]!;
      await norm(n.weight,n.epsilon,coordinate,column=>residual(layer,column,position));
    };
    const roundHalf=(value:number):number=>{
      const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,value,true);
      const bits=data.getUint32(0,true);
      if((bits&0x7fffffff)===0)return bits>>>31?-0:0;
      return decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(bits)));
    };
    const firstNormValue=async(coordinate:number,token:number):Promise<number>=>{
      const n=layers[0]!.mlp.normalizations[0]!;
      const sum=await foldCpuArm64F32Sum(width,async c=>{
        const value=Number(await readDirectF16Literal(reader,tensor(output.embeddingWeight),token*width+c));
        return Math.fround(value*value);
      });
      const variance=Math.fround(Math.fround(sum/width)+Math.fround(n.epsilon));
      const factor=Math.fround(1/Math.fround(Math.sqrt(variance)));
      const value=Number(await readDirectF16Literal(reader,tensor(output.embeddingWeight),token*width+coordinate));
      const learned=Number(await readDirectF16Literal(reader,tensor(n.weight),coordinate));
      return roundHalf(roundHalf(Math.fround(value*factor))*learned);
    };
    const tokenBranches=async(position:string,evaluate:(token:number)=>Promise<number>)=>{
      const selector=`input_tokens[${position}]`;
      await s.write(`'token_value:{assert!(${selector}<${output.embeddingShape[0]});`);
      let domain=new DirectBranchDomain().split("token",">=",rational(0n)).truth!
        .split("token","<=",rational(BigInt(output.embeddingShape[0]-1))).truth!;
      let previous:number|undefined;
      const flush=async(end:number)=>{
        if(previous===undefined)return;
        const narrowed=domain.split("token","<=",rational(BigInt(end)));
        const literal=rustF64(Object.is(previous,-0)?"-0":previous);
        if(!narrowed.truth){s.eliminatedBranches++;return;}
        if(!narrowed.falsity){s.eliminatedBranches++;await s.write(`break 'token_value ${literal};`);return;}
        await s.write(`if ${selector}<=${end} {break 'token_value ${literal};}`);
        domain=narrowed.falsity;
      };
      for(let token=0;token<output.embeddingShape[0];token++){
        const value=await evaluate(token);
        if(!Object.is(value,previous)){await flush(token-1);previous=value;}
      }
      await flush(output.embeddingShape[0]-1);await s.write("}");
    };
    const preNorm=async(layer:number,coordinate:number,position:string)=>{
      const n=layers[layer]!.mlp.normalizations[0]!;
      if(layer===0){await tokenBranches(position,token=>firstNormValue(coordinate,token));return;}
      await norm(n.weight,n.epsilon,coordinate,column=>hidden(layer-1,column,position));
    };
    const inputProjection=async(layer:number,role:"q"|"k"|"v",row:number,position:string)=>{
      const projection=layers[layer]!.attention.attention.projections[role];
      if(layer!==0){await linear(projection,row,c=>preNorm(layer,c,position));return;}
      // Substitution of token embedding literals proves every first-layer
      // normalized projection constant within that token branch. Evaluate it
      // with the same ordered rounding, then remove all internal conditions.
      await tokenBranches(position,async token=>{
        const lanes=[0,0,0,0];
        for(let c=0;c<projection.shape[1];c++){
          const x=await firstNormValue(c,token);
          const w=Number(await readDirectF16Literal(reader,tensor(projection.weight),row*projection.shape[1]+c));
          lanes[c%4]=Math.fround(lanes[c%4]!+x*w);
        }
        return roundHalf(Math.fround(Math.fround(lanes[0]!+lanes[1]!)+Math.fround(lanes[2]!+lanes[3]!)));
      });
    };
    const residual=async(layer:number,coordinate:number,position:string)=>{
      await s.round("f16",async()=>{
        await hidden(layer-1,coordinate,position);await s.write("+");
        const a=layers[layer]!.attention.attention;
        await linear(a.projections.o,coordinate,column=>context(layer,Math.floor(column/a.headDim),column%a.headDim,position));
      });
    };
    const rope=async(layer:number,coordinate:number,position:string,sine:0|1)=>{
      const a=layers[layer]!.attention.attention;
      const known=Number(position);
      if(Number.isSafeInteger(known)&&known>=0&&known<a.maxPosition){
        await s.write(rustF64(decodeIeeeF16ToF32(fixedF16RopeLiteral(known,coordinate,a.headDim,a.ropeTheta,sine))));
        return;
      }
      await s.write("'rope:{");let from=0,prior:number|undefined;
      const flush=async(end:number)=>{
        if(prior===undefined)return;
        await s.write(`if ${position}>=${from} && ${position}<=${end} {break 'rope ${rustF64(prior)};}`);
      };
      for(let pos=0;pos<a.maxPosition;pos++){
        const value=decodeIeeeF16ToF32(fixedF16RopeLiteral(pos,coordinate,a.headDim,a.ropeTheta,sine));
        if(!Object.is(value,prior)){await flush(pos-1);from=pos;prior=value;}
      }
      await flush(a.maxPosition-1);await s.write("panic!(\"position out of context\")}");
    };
    const rotated=async(layer:number,role:"q"|"k",head:number,coordinate:number,position:string)=>{
      const a=layers[layer]!.attention.attention,half=a.headDim/2;
      const h=role==="q"?head:Math.floor(head/(a.heads/a.kvHeads));
      await s.round("f16",async()=>{
        await s.round("f16",async()=>{await inputProjection(layer,role,h*a.headDim+coordinate,position);await s.write("*");await rope(layer,coordinate,position,0);});
        await s.write("+");await s.round("f16",async()=>{
          if(coordinate<half)await s.write("-");
          await inputProjection(layer,role,h*a.headDim+(coordinate+half)%a.headDim,position);
          await s.write("*");await rope(layer,coordinate,position,1);
        });
      });
    };
    const unmaskedScore=async(layer:number,head:number,position:string,key:string)=>{
      const a=layers[layer]!.attention.attention;
      await s.round("f16",async()=>{
        await reduction(a.headDim,async c=>{await rotated(layer,"q",head,c,position);await s.write("*");await rotated(layer,"k",head,c,key);});
        await s.write(`*${rustF64(a.scaling)}`);
      });
    };
    // Nonnegative RMS reductions and Cauchy-Schwarz give a conservative
    // bound independent of the token sequence. Only use it within the domain
    // where the accumulated F32 error is <1/8; the factor 2 covers that error,
    // reciprocal/sqrt rounding and both F16 normalization boundaries.
    const causalOnly: boolean[]=[];
    const scoreBounds:number[]=[];
    for(let layer=0;layer<layers.length;layer++){
      const a=layers[layer]!.attention.attention,normWeight=layers[layer]!.mlp.normalizations[0]!.weight;
      let normMaximum=0;
      for(let c=0;c<width;c++)normMaximum=Math.max(normMaximum,Math.abs(Number(await readDirectF16Literal(reader,tensor(normWeight),c))));
      const projectionBound=async(role:"q"|"k")=>{
        const p=a.projections[role];let maximum=0;
        for(let row=0;row<p.shape[0];row++){
          let squared=0,absolute=0;
          for(let c=0;c<p.shape[1];c++){
            const w=Number(await readDirectF16Literal(reader,tensor(p.weight),row*p.shape[1]+c));
            squared+=w*w;absolute+=Math.abs(w);
          }
          const bound=(2*Math.sqrt(width)*normMaximum*Math.sqrt(squared)+absolute*2**-23)*1.01+2**-24;
          maximum=Math.max(maximum,bound);
        }
        return maximum;
      };
      const q=await projectionBound("q"),k=await projectionBound("k");
      const qRot=2*q*1.01+2**-23,kRot=2*k*1.01+2**-23;
      const scoreMaximum=(a.headDim*qRot*kRot*1.01+2**-24)*Math.abs(a.scaling)*1.01+2**-24;
      causalOnly.push(width<=1000000 && 2*Math.sqrt(width)*normMaximum<65504 && q<65504 && k<65504 && scoreMaximum<16);
      scoreBounds.push(scoreMaximum);
    }
    const context=async(layer:number,head:number,coordinate:number,position:string)=>{
      const a=layers[layer]!.attention.attention;
      const query=Number(position);
      if(!Number.isSafeInteger(query)||query<0||query>=output.maxPosition)
        throw new Error("Attention position was not substituted before reduction");
      if(!causalOnly[layer])throw new Error("Finite causal-mask elimination has not been proved for this checkpoint");
      const v=(key:number)=>inputProjection(layer,"v",Math.floor(head/(a.heads/a.kvHeads))*a.headDim+coordinate,String(key));
      // A singleton softmax is exactly one for finite scores. Propagate its
      // domain before reaching subtraction, exponential or either reduction.
      if(query===0){
        s.eliminatedBranches++;
        // The ordered attention accumulator starts at +0. Keep that addition:
        // returning V directly would change an underflowed negative zero.
        await s.write("(0.0+");await v(0);await s.write(")");return;
      }
      const reachedScore=(key:number)=>unmaskedScore(layer,head,position,String(key));
      const maximum:RustExpression=async()=>{
        await s.write("'maximum_choice:{");
        for(let winner=0;winner<=query;winner++){
          await s.write("if ");let first=true;
          for(let other=0;other<=query;other++){
            if(other===winner)continue;
            if(!first)await s.write(" && ");first=false;
            await s.write("(");await reachedScore(winner);await s.write(other<winner?">":">=");await reachedScore(other);await s.write(")");
          }
          await s.write(" {break 'maximum_choice ");await reachedScore(winner);await s.write(";}");
        }
        await s.write('panic!("outside proved finite score domain")}');
      };
      const differenceBound=2*scoreBounds[layer]!*1.01+2**-149;
      const exponent=(key:number):Promise<void>=>emitRustExp(s,()=>s.round("f32",async()=>{
        await reachedScore(key);await s.write("-");await maximum();
      // Every finite F16 score is an integer multiple of 2^-24. Its
      // difference with another F16 score stays on that lattice after F32
      // rounding (larger F32 ULPs are also multiples of the same quantum).
      }),true,differenceBound,undefined,2**-24);
      const denominator=(last:number):Promise<void>=>{
        if(last<0)return s.write("0.0");
        return s.round("f32",async()=>{await denominator(last-1);await s.write("+");await exponent(last);});
      };
      const sum=(last:number):Promise<void>=>{
        if(last<0)return s.write("0.0");
        return s.round("f32",async()=>{
          await sum(last-1);await s.write("+");await s.round("f32",async()=>{
            await s.round("f16",()=>s.round("f32",async()=>{
              await exponent(last);await s.write("/");await denominator(query);
            }),true);
            await s.write("*");await v(last);
          });
        });
      };
      await s.round("f16",()=>sum(query),true);
    };
    if(output.embeddingShape[1]!==width)throw new Error("Embedding width mismatch");
    for(const layer of layers) {
      const a=layer.attention.attention;
      if(a.headDim%2!==0||a.heads%a.kvHeads!==0||a.projections.o.shape[0]!==width||layer.mlp.normalizations.length!==2)throw new Error("Unsupported discovered geometry");
    }
    await s.write(`// Direct checkpoint specialization. Numeric policy: PyTorch CPU arm64 F16; ordered F32 reductions.\n#![recursion_limit="65536"]\npub fn compiled_dimension(input_tokens:&[usize],t:usize)->f64 {let n=input_tokens.len();assert!(n>0 && n<=${output.maxPosition} && t<n);`);
    s.beginReducedExpression();
    await s.write("'position_choice:{");
    for(let position=0;position<output.maxPosition;position++){
      await s.write(`if t==${position} {break 'position_choice `);
      await linear({weight:output.weight,shape:output.shape},dimension,c=>norm(output.finalNormWeight,output.finalNormEpsilon,c,column=>hidden(layers.length-1,column,String(position))));
      await s.write(";}");
    }
    await s.write('panic!("outside context domain")}');
    await s.write("}\n");await s.close();await rename(draft,path);
  } catch(error){
    s.destroy();
    await writeFile(path+".reduction.json",JSON.stringify({status:"pending",rustCompilationAdmitted:false,
      reason:error instanceof Error?error.message:String(error),draft,dimension},null,2)+"\n");
    throw error;
  }finally{await reader.close();}
}
