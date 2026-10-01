import {DirectFlatSubstitution,type FlatConditions,type FlatConsumer,type FlatProducer} from "./direct-flat-substitution.js";

/** PyTorch CPU arm64 F32 reduce_all: partial-vector lane accumulation,
 * sequential horizontal reduction below a full vector, SIMD pair reduction
 * otherwise. Active values precede exact positive masked zeros. The full row
 * length determines the horizontal order even when activeSize is smaller.
 * This emits substituted scalar operations, never a runtime reduction.
 */
export async function substituteCpuArm64SoftmaxSum(f:DirectFlatSubstitution,path:FlatConditions,
  activeSize:number,fullVector:boolean,term:(index:number)=>FlatProducer,consume:FlatConsumer):Promise<void>{
  if(!Number.isSafeInteger(activeSize)||activeSize<1||(!fullVector&&activeSize>=4))
    throw new Error("Invalid softmax reduction domain");
  const zero:FlatProducer=(p,k)=>f.literal(p,0,k);
  const add=(a:FlatProducer,b:FlatProducer):FlatProducer=>(p,k)=>f.round(p,
    (p,k)=>f.binary(p,a,b,"+",k),"f32",false,k);
  const lane=(index:number,last:number):FlatProducer=>last<index?zero:last===index?term(last):
    add(lane(index,last-4),term(last));
  const value=(index:number)=>lane(index,index+4*Math.floor((activeSize-1-index)/4));
  if(fullVector)return add(add(value(0),value(2)),add(value(1),value(3)))(path,consume);
  const sequential=(last:number):FlatProducer=>last===0?term(0):add(sequential(last-1),term(last));
  return sequential(activeSize-1)(path,consume);
}
