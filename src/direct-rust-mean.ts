import { DirectRustStream, type RustExpression } from './direct-rust-stream.js';

/** Declared PyTorch 2.12.1 CPU arm64 contiguous F32 sum policy.
 * Backend vector width is four F32 lanes; it is not a model dimension.
 * The cascade and final left-to-right additions follow SumKernel.cpp.
 * Inputs are finite nonnegative F32 terms (RMS squares). This proves that
 * empty sums and additions of positive zero may be eliminated exactly.
 * Only recursion indices are retained, never an operation graph or row cache.
 */
const vectorWidth=4;
function geometry(size:number){
  if(!Number.isSafeInteger(size)||size<=0)throw new Error('Invalid mean reduction width');
  const vector=size>=vectorWidth;
  const rows=vector?Math.floor(size/vectorWidth):size;
  const groups=Math.floor(rows/4);
  const power=Math.max(4,Math.floor(Math.ceil(Math.log2(Math.max(groups,1)))/4));
  return {vector,rows,groups,step:2**power};
}
export async function foldCpuArm64F32Sum(size:number,input:(index:number)=>Promise<number>):Promise<number>{
  const g=geometry(size),load=(row:number,lane:number)=>input(g.vector?row*vectorWidth+lane:row);
  const component=async(lane:number):Promise<number>=>{
    const partial=async(part:number):Promise<number>=>{
      const block=async(level:number,start:number):Promise<number>=>{
        if(level===0)return load(part+4*start,lane);
        let sum=0;const span=g.step**(level-1);
        for(let i=0;i<g.step;i++)sum=Math.fround(sum+await block(level-1,start+i*span));
        return sum;
      };
      let sum=0;
      for(let level=0;level<4;level++){
        const span=g.step**level,end=Math.floor(g.groups/span)*span;
        const start=level===3?0:Math.floor(g.groups/(span*g.step))*span*g.step;
        let accumulator=0;
        for(let i=start;i<end;i+=span)accumulator=Math.fround(accumulator+await block(level,i));
        sum=Math.fround(sum+accumulator);
      }
      if(part===0)for(let row=4*g.groups;row<g.rows;row++)sum=Math.fround(sum+await load(row,lane));
      return sum;
    };
    let sum=await partial(0);
    for(let part=1;part<4;part++)sum=Math.fround(sum+await partial(part));
    return sum;
  };
  if(!g.vector)return Math.fround(await component(0));
  let sum=0;
  for(let i=g.rows*vectorWidth;i<size;i++)sum=Math.fround(sum+await input(i));
  for(let lane=0;lane<vectorWidth;lane++)sum=Math.fround(sum+await component(lane));
  return Math.fround(sum+0);
}
export async function emitCpuArm64F32Sum(s:DirectRustStream,size:number,input:(index:number)=>Promise<void>):Promise<void>{
  const g=geometry(size);
  const add=(a:RustExpression,b:RustExpression)=>s.round('f32',async()=>{await a();await s.write('+');await b();});
  const zero=()=>s.write('0.0');
  const load=(row:number,lane:number)=>input(g.vector?row*vectorWidth+lane:row);
  const component=async(lane:number):Promise<void>=>{
    const partial=async(part:number):Promise<void>=>{
      const block=async(level:number,start:number):Promise<void>=>{
        if(level===0){await load(part+4*start,lane);return;}
        const span=g.step**(level-1);
        const sum=async(last:number):Promise<void>=>{
          if(last===0){await block(level-1,start);return;}
          await add(()=>sum(last-1),()=>block(level-1,start+last*span));
        };
        await sum(g.step-1);
      };
      const accumulator=async(level:number,last:number,start:number,span:number):Promise<void>=>{
        if(last<start){await zero();return;}
        if(last===start){await block(level,last);return;}
        await add(()=>accumulator(level,last-span,start,span),()=>block(level,last));
      };
      const levels=async(last:number):Promise<void>=>{
        if(last<0){await zero();return;}
        const span=g.step**last,end=Math.floor(g.groups/span)*span;
        const start=last===3?0:Math.floor(g.groups/(span*g.step))*span*g.step;
        if(start===end){s.eliminatedBranches++;await levels(last-1);return;}
        let previous=last-1;
        while(previous>=0){
          const previousSpan=g.step**previous,previousEnd=Math.floor(g.groups/previousSpan)*previousSpan;
          const previousStart=Math.floor(g.groups/(previousSpan*g.step))*previousSpan*g.step;
          if(previousStart<previousEnd)break;previous--;
        }
        if(previous<0){await accumulator(last,end-span,start,span);return;}
        await add(()=>levels(previous),()=>accumulator(last,end-span,start,span));
      };
      const tail=async(last:number):Promise<void>=>{
        if(part!==0||last<4*g.groups){await levels(3);return;}
        if(g.groups===0&&last===0){await load(last,lane);return;}
        await add(()=>tail(last-1),()=>load(last,lane));
      };
      await tail(g.rows-1);
    };
    const combine=async(last:number):Promise<void>=>{
      if(last===0||g.groups===0){await partial(0);return;}
      await add(()=>combine(last-1),()=>partial(last));
    };
    await combine(3);
  };
  if(!g.vector){await component(0);return;}
  const tail=async(last:number):Promise<void>=>{
    if(last<g.rows*vectorWidth){await zero();return;}
    if(last===g.rows*vectorWidth){await input(last);return;}
    await add(()=>tail(last-1),()=>input(last));
  };
  const lanes=async(last:number):Promise<void>=>{
    if(last<0){await tail(size-1);return;}
    if(last===0&&g.rows*vectorWidth===size){await component(0);return;}
    await add(()=>lanes(last-1),()=>component(last));
  };
  await lanes(vectorWidth-1);
}
