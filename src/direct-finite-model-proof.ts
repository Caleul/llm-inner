import {decodeIeeeF16ToF32} from "./utils.js";
import {f32BitsToDyadic,roundDyadicToF16IfElse} from "./fixed-f16-projection.js";

interface Projection {weight:string;shape:[number,number]}
interface Normalization {weight:string;epsilon:number}
export interface FiniteModelLayer {
  pre:Normalization;post:Normalization;heads:number;kvHeads:number;headDim:number;ropeTheta:number;
  v:Projection;o:Projection;gate:Projection;up:Projection;down:Projection;
}
type WeightReader=(name:string,index:number)=>Promise<number>;

/** Checkpoint-dependent numerical proof only. Read one coefficient at a time;
 * do not retain rows, activations, bounds by coordinate, or an operation graph.
 * On success every normalization input remains finite F16 for every finite
 * F16 embedding matrix and admitted sequence length. A failed conservative
 * proof does not assert that the real forward overflows.
 * The caller must first prove finite rotated Q/K and |score| < 16 for every
 * layer. This supplies the finite softmax domain used by the probability bound.
 */
export async function proveFiniteNormalizationInputs(width:number,context:number,
  layers:Iterable<FiniteModelLayer>,final:Normalization,weight:WeightReader,finiteScoresBounded:boolean):Promise<boolean>{
  if(!(finiteScoresBounded&&width>0&&width<=1000000&&context>0&&context<=1000000))return false;
  const normBudget=async(norm:Normalization)=>{
    const epsilon=Math.fround(norm.epsilon);
    if(!(epsilon>=2**-126&&Number.isFinite(epsilon)))return Infinity;
    let gamma=0;
    for(let c=0;c<width;c++)gamma=Math.max(gamma,Math.abs(await weight(norm.weight,c)));
    if(gamma===0)return 0;
    // Ordered positive F32 variance has relative loss <1/32 in this
    // domain. Root, reciprocal, and the two half conversions fit strictly
    // inside this 1.25 factor; absolute half errors are covered separately.
    const bound=1.25*Math.sqrt(width)*gamma+Math.sqrt(width)*2**-23;
    return bound<65504?bound:Infinity;
  };
  const squaredRow=async(p:Projection,row:number)=>{
    if(p.shape[1]>1000000)return Infinity;
    let sum=0;
    for(let c=0;c<p.shape[1];c++){
      const w=await weight(p.weight,row*p.shape[1]+c);sum+=w*w;
    }
    return sum;
  };
  const projection=async(p:Projection,row:number,inputNorm:number)=>
    upperRoundedHalf(1.125*Math.sqrt(await squaredRow(p,row))*inputNorm);
  for(const layer of layers){
    if(!(layer.headDim>0&&layer.headDim<=1000000&&layer.heads>0&&layer.kvHeads>0&&layer.heads%layer.kvHeads===0&&
      layer.ropeTheta>=1&&Number.isFinite(layer.ropeTheta)&&
      layer.o.shape[1]===layer.heads*layer.headDim&&layer.v.shape[0]===layer.kvHeads*layer.headDim&&
      layer.o.shape[1]<=1000000&&layer.gate.shape[0]<=1000000))return false;
    const pre=await normBudget(layer.pre),post=await normBudget(layer.post);
    if(!Number.isFinite(pre)||!Number.isFinite(post))return false;
    let valueSquared=0;
    for(let row=0;row<layer.v.shape[0];row++){
      if(!Number.isFinite(await projection(layer.v,row,pre)))return false;
      valueSquared+=await squaredRow(layer.v,row);
    }
    const valueNorm=pre===0||valueSquared===0?0:1.125*pre*Math.sqrt(valueSquared)+Math.sqrt(layer.v.shape[0])*2**-23;
    // Positive half probabilities sum to <1.125 for n<=1e6. The weighted
    // F32 reduction and half conversion fit inside a factor of two. Repeat
    // KV heads in the bound exactly as the discovered geometry requires.
    const contextNorm=valueNorm===0?0:2*Math.sqrt(layer.heads/layer.kvHeads)*valueNorm+Math.sqrt(layer.o.shape[1])*2**-23;
    if(!(contextNorm<65504))return false;
    for(let row=0;row<width;row++){
      // Finite half input magnitude is <=65504. An addition of magnitude
      // <15 cannot reach the composed half overflow boundary near 65520.
      if(!(await projection(layer.o,row,contextNorm)<15))return false;
    }
    let productSquared=0;
    for(let row=0;row<layer.gate.shape[0];row++){
      const gate=await projection(layer.gate,row,post),up=await projection(layer.up,row,post);
      // The pinned finite-F16 SiLU policy satisfies |SiLU(x)|<=|x|.
      const product=upperRoundedHalf(gate*up);if(!Number.isFinite(product))return false;
      productSquared+=product*product;
    }
    for(let row=0;row<width;row++)if(!(await projection(layer.down,row,Math.sqrt(productSquared))<15))return false;
    // Residual and MLP additions both round to finite half; the next
    // decoder layer therefore starts with the same <=65504 invariant.
  }
  return Number.isFinite(await normBudget(final));
}

function upperRoundedHalf(value:number):number{
  if(!(value>=0&&Number.isFinite(value))||value>=65520)return Infinity;
  const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,value,true);
  return decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0,true))));
}
