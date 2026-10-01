import {DirectFlatSubstitution,type FlatConditions} from "./direct-flat-substitution.js";
import {decodeIeeeF16ToF32} from "./utils.js";
import {f32BitsToDyadic,roundDyadicToF16IfElse} from "./fixed-f16-projection.js";

interface Projection {weight:string;shape:[number,number]}
interface Normalization {weight:string;epsilon:number}
export interface InitialMlpZeroGeometry {
  width:number;context:number;heads:number;kvHeads:number;headDim:number;
  pre:Normalization;post:Normalization;
  v:Projection;o:Projection;gate:Projection;up:Projection;
}
/** A magnitude proof, not a forward executor or an activation cache. Each
 * reached coefficient is read progressively. No row or scalar result is
 * retained across invocations. Only a proved zero term is removed from the
 * ordered F32 dot, whose +0 seed makes the term's zero sign irrelevant.
 */
export async function proveInitialMlpProductZero(f:DirectFlatSubstitution,path:FlatConditions,
  geometry:InitialMlpZeroGeometry,query:number,neuron:number,
  weight:(name:string,index:number)=>Promise<number>):Promise<boolean>{
  return proveInitialZero(f,path,geometry,query,weight,{neuron});
}

/** Eliminate both corrections when their rounded additions provably retain
 * a nonzero embedding. At zero, require the stronger signed-zero attention
 * and +0 down-projection proof: that final addition settles a -0 embedding.
 */
export async function proveInitialHiddenIdentity(f:DirectFlatSubstitution,path:FlatConditions,
  geometry:InitialMlpZeroGeometry&{down:Projection},query:number,coordinate:number,
  weight:(name:string,index:number)=>Promise<number>):Promise<boolean>{
  return proveInitialZero(f,path,geometry,query,weight,{coordinate,down:geometry.down});
}

async function proveInitialZero(f:DirectFlatSubstitution,path:FlatConditions,
  geometry:InitialMlpZeroGeometry,query:number,weight:(name:string,index:number)=>Promise<number>,
  request:{neuron:number}|{coordinate:number;down:Projection}):Promise<boolean>{
  const g=geometry;
  if(g.width>1000000||g.context>1000000||g.headDim>1000000||g.o.shape[1]>1000000||
    query<0||query>=g.context||g.heads%g.kvHeads)return false;
  const embedding=async(position:number,coordinate:number)=>{
    let magnitude=Infinity;
    await f.f16Input(path,()=>f.stream.write(`input_tokens[${position}][${coordinate}]`),async(_p,value)=>{
      magnitude=Math.max(Math.abs(value.minimum),Math.abs(value.maximum));
    });
    return magnitude;
  };
  const normalized=async(norm:Normalization,coordinate:number,input:()=>Promise<number>)=>{
    const epsilon=Math.fround(norm.epsilon),magnitude=await input();
    if(!Number.isFinite(magnitude)||!(epsilon>=2**-126&&Number.isFinite(epsilon)))return Infinity;
    const gamma=Math.abs(await weight(norm.weight,coordinate));
    // Variance >= epsilon. The 1.01 factor covers both normal F32 root
    // and reciprocal rounding; 2*sqrt(width) covers ordered mean error.
    const converted=upperHalf(Math.min(2*Math.sqrt(g.width),magnitude*1.01/Math.sqrt(epsilon)));
    // The source converts the normalized coordinate to half BEFORE the
    // learned-weight multiplication. Bound both conversions separately;
    // a single rounding after gamma could underestimate subnormal values.
    return upperHalf(converted*gamma);
  };
  const pre=(position:number,coordinate:number)=>normalized(g.pre,coordinate,()=>embedding(position,coordinate));
  const dot=async(projection:Projection,row:number,input:(coordinate:number)=>Promise<number>)=>{
    if(projection.shape[1]>1000000)return Infinity;
    let sum=0;
    for(let c=0;c<projection.shape[1];c++){
      const coefficient=Math.abs(await weight(projection.weight,row*projection.shape[1]+c)),bound=await input(c);
      // Do not erase a possible nonfinite operand even for a zero weight.
      if(!Number.isFinite(coefficient)||!Number.isFinite(bound))return Infinity;
      sum+=coefficient*bound;
    }
    // <=250002 rounded F32 additions: gamma_k <0.016. This factor also
    // covers the tiny binary64 error of this positive bound calculation.
    return upperHalf(sum*1.125);
  };
  const value=(position:number,row:number)=>dot(g.v,row,c=>pre(position,c));
  const context=async(column:number)=>{
    const head=Math.floor(column/g.headDim),coordinate=column%g.headDim;
    const row=Math.floor(head/(g.heads/g.kvHeads))*g.headDim+coordinate;
    if(query===0)return value(0,row);
    let maximum=0;
    for(let position=0;position<=query;position++)maximum=Math.max(maximum,await value(position,row));
    // Caller proves finite scores with differences <=32. Exponentials and
    // normalized F32 probabilities are normal and positive. For n<=1e6,
    // gamma_sum<0.016 and total F16 probability <=
    // (1+2^-24)^2/(1-gamma_sum)*(1+2^-11)+n*2^-25 <1.125.
    // The second factor bounds the ordered F32 weighted-value dot error.
    return upperHalf(maximum*1.125*1.125);
  };
  const residual=async(coordinate:number)=>{
    const original=await embedding(query,coordinate),attention=await dot(g.o,coordinate,context);
    return upperHalf(original+attention);
  };
  const post=(coordinate:number)=>normalized(g.post,coordinate,()=>residual(coordinate));
  const productZero=async(neuron:number)=>{
  const gate=await dot(g.gate,neuron,post),up=await dot(g.up,neuron,post);
  if(!Number.isFinite(gate)||!Number.isFinite(up))return false;
  // The declared finite-F16 SiLU policy has |SiLU(x)|<=|x|. Products of
  // two finite F16 values are exact F32. Half nearest-even maps magnitude
  // <=2^-25 to signed zero, including the tie with the even zero code.
  return gate*up<=2**-25;
  };
  if("neuron" in request)return productZero(request.neuron);
  if(request.coordinate<0||request.coordinate>=g.width||request.down.shape[1]!==g.gate.shape[0]||
    g.gate.shape[0]>1000000)return false;
  const attention=await dot(g.o,request.coordinate,context);
  let margin=0;
  await f.f16Input(path,()=>f.stream.write(`input_tokens[${query}][${request.coordinate}]`),async(_p,value)=>{
    if(value.minimum>0||value.maximum<0)
      margin=halfResidualMargin(Math.min(Math.abs(value.minimum),Math.abs(value.maximum)));
  });
  if(margin>0){
    if(!(attention<margin))return false;
    const down=await dot(request.down,request.coordinate,async neuron=>{
      const gate=await dot(g.gate,neuron,post),up=await dot(g.up,neuron,post);
      return upperHalf(gate*up);
    });
    return down<margin;
  }
  if(attention!==0)return false;
  // A zero MAGNITUDE bound for down is insufficient at a -0 embedding:
  // a nonzero negative dot could round to -0. Require zero input terms so
  // its ordered +0 reduction seed proves the actual positive-zero result.
  for(let neuron=0;neuron<g.gate.shape[0];neuron++){
    if(!Number.isFinite(await weight(request.down.weight,request.coordinate*request.down.shape[1]+neuron))||
      !await productZero(neuron))return false;
  }
  return true;
}

function halfResidualMargin(minimumMagnitude:number):number{
  if(!(minimumMagnitude>0&&minimumMagnitude<=65504))return 0;
  // At every normal half x in this or a larger binade, the distance to
  // either half midpoint is >= 2^(e-12); subnormals use 2^-25. The F32
  // addition error is <= that actual midpoint distance / 4096, including
  // powers of two. A STRICT correction bound below this margin therefore
  // keeps R16(R32(x+delta)) == x without relying on tie ownership.
  let gap=2**-25;
  if(minimumMagnitude>=2**-14){
    let binade=2**-14;gap=2**-26;
    while(binade*2<=minimumMagnitude){binade*=2;gap*=2;}
  }
  return gap*(1-2**-12);
}

function upperHalf(value:number):number{
  if(value===0)return 0;
  if(!(value>0&&Number.isFinite(value))||value>=65520)return Infinity;
  const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,value,true);
  const bits=roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0,true)));
  // This bounds the R32 -> R16 RESULT, not the unrounded real value.
  // Both rounding maps are monotone: actualF32<=value implies
  // R16(actualF32)<=R16(R32(value)). A half ceiling unnecessarily kept
  // subnormal terms that are provably nearest-even zero.
  return decodeIeeeF16ToF32(bits);
}
