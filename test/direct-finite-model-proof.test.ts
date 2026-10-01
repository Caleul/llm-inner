import assert from "node:assert/strict";
import {test} from "node:test";
import {proveFiniteNormalizationInputs,type FiniteModelLayer} from "../src/direct-finite-model-proof.js";

function layer(width:number,intermediate:number):FiniteModelLayer{
  const projection=(weight:string,rows=width,columns=width)=>({weight,shape:[rows,columns] as [number,number]});
  return {pre:{weight:"pre",epsilon:1e-6},post:{weight:"post",epsilon:1e-6},
    heads:1,kvHeads:1,headDim:width,ropeTheta:10000,v:projection("v"),o:projection("o"),
    gate:projection("gate",intermediate),up:projection("up",intermediate),down:projection("down",width,intermediate)};
}

test("finite-input proof reads weights progressively and covers variable discovered geometry",async()=>{
  for(const [width,intermediate,count] of [[2,1,1],[4,3,2],[16,64,2]]){
    let reads=0;
    const weight=async(name:string,index:number)=>{
      reads++;assert.ok(Number.isSafeInteger(index)&&index>=0);
      return ["pre","post","final"].includes(name)?1:1/256;
    };
    assert.equal(await proveFiniteNormalizationInputs(width!,17,
      Array.from({length:count!},()=>layer(width!,intermediate!)),{weight:"final",epsilon:1e-6},weight,true),true);
    assert.ok(reads>width!);
  }
});

test("finite-input proof rejects unsupported angular, normalization, and overflowing residual bounds",async()=>{
  const weight=async(name:string,_index:number)=>["pre","post","final"].includes(name)?1:1/64;
  const base=layer(2,1),final={weight:"final",epsilon:1e-6};
  assert.equal(await proveFiniteNormalizationInputs(2,8,[base],final,weight,false),false);
  assert.equal(await proveFiniteNormalizationInputs(2,8,[{...base,ropeTheta:0}],final,weight,true),false);
  assert.equal(await proveFiniteNormalizationInputs(2,8,[{...base,post:{weight:"post",epsilon:0}}],final,weight,true),false);
  assert.equal(await proveFiniteNormalizationInputs(2,8,[base],final,async(name,index)=>name==="o"?65504:weight(name,index),true),false);
  assert.equal(await proveFiniteNormalizationInputs(2,8,[base],final,async(name,index)=>name==="final"?65504:weight(name,index),true),false);
});
