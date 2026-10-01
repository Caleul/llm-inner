import { DirectFlatSubstitution, type FlatConditions, type FlatConsumer, type FlatProducer } from "./direct-flat-substitution.js";

/** One scalar weight is read at the reached multiplication. No weight row,
 * activation vector, source cache or calculation graph is retained.
 * Input and weights must have the finite F16 domain declared by the caller.
 */
export async function substituteFlatProjection(f:DirectFlatSubstitution,path:FlatConditions,
  size:number,weight:(coordinate:number)=>Promise<number>,input:(coordinate:number)=>FlatProducer,
  consume:FlatConsumer):Promise<void>{
  if(!Number.isSafeInteger(size)||size<=0)throw new Error("Invalid discovered projection width");
  const term=(coordinate:number):FlatProducer=>async(path,k)=>{
    const value=await weight(coordinate);
    // Each lane starts at +0. Finite F16 products and their F32 sums are
    // multiples of 2^-48, so an accumulator cannot underflow to -0. A zero
    // weight's signed product can therefore be removed before its dependency.
    if(value===0){f.stream.eliminatedBranches++;return f.literal(path,0,k);}
    return f.binary(path,input(coordinate),async(path,k)=>{
      return f.literal(path,value,(path,scalar)=>k(path,{...scalar,precision:"f16"}));
    },"*",(path,product)=>{
      // Two finite F16 significands have at most 22 significant bits and an
      // exponent inside F32's normal range, so their F64 product is exact F32.
      return k(path,{...product,precision:"f32"});
    });
  };
  return substituteFlatReduction(f,path,size,term,consume);
}

export async function substituteFlatReduction(f:DirectFlatSubstitution,path:FlatConditions,size:number,
  term:(coordinate:number)=>FlatProducer,consume:FlatConsumer):Promise<void>{
  if(!Number.isSafeInteger(size)||size<=0)throw new Error("Invalid discovered reduction width");
  const zero:FlatProducer=(path,k)=>f.literal(path,0,k);
  const lane=(index:number,last:number):FlatProducer=>(path,k)=>{
    if(last<index)return zero(path,k);
    const add:FlatProducer=(path,k)=>f.binary(path,last===index?zero:lane(index,last-4),term(last),"+",k);
    if(last===index)return add(path,(path,value)=>k(path,{...value,precision:"f32"}));
    return f.round(path,add,"f32",false,k);
  };
  const pair=(index:number):FlatProducer=>(path,k)=>{
    const a=index*2,b=a+1;
    const add:FlatProducer=(path,k)=>f.binary(path,
      lane(a,a+4*Math.floor((size-1-a)/4)),lane(b,b+4*Math.floor((size-1-b)/4)),"+",k);
    if(b>=size)return add(path,k);
    return f.round(path,add,"f32",false,k);
  };
  const finalSum:FlatProducer=(path,k)=>f.binary(path,pair(0),pair(1),"+",k);
  const sum:FlatProducer=(path,k)=>size<=2?finalSum(path,k):f.round(path,finalSum,"f32",false,k);
  return f.round(path,sum,"f16",true,consume);
}
