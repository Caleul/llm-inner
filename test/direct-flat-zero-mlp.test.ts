import assert from "node:assert/strict";
import {test} from "node:test";
import {mkdtemp,readFile,rm} from "node:fs/promises";
import {join} from "node:path";
import {tmpdir} from "node:os";
import {createHash} from "node:crypto";
import {DirectRustStream} from "../src/direct-rust-stream.js";
import {DirectFlatSubstitution,FlatConditions} from "../src/direct-flat-substitution.js";
import {exactNumberRational} from "../src/direct-round-preimage.js";
import {decodeIeeeF16ToF32} from "../src/utils.js";
import {proveInitialHiddenIdentity,proveInitialMlpProductZero,type InitialMlpZeroGeometry} from "../src/direct-flat-zero-mlp.js";
import {substituteFlatProjection} from "../src/direct-flat-projection.js";

test("the pinned finite half SiLU policy satisfies the magnitude bound used for zero elimination",async()=>{
  const data=await readFile(new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin",import.meta.url));
  for(let code=0;code<65536;code++){
    const x=decodeIeeeF16ToF32(code);if(!Number.isFinite(x))continue;
    const y=decodeIeeeF16ToF32(data.readUInt16LE(code*2));
    assert.ok(Math.abs(y)<=Math.abs(x),`${code}: ${x} -> ${y}`);
  }
});

test("initial MLP zero proof uses inherited embedding bounds and rejects unrestricted inputs",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-zero-mlp-"));
  try{
    const s=new DirectRustStream(join(dir,"unused.rs")),f=new DirectFlatSubstitution(s);
    for(const width of [2,4,8]){
      const projection=(weight:string)=>({weight,shape:[width,width] as [number,number]});
      const g:InitialMlpZeroGeometry={width,context:3,heads:1,kvHeads:1,headDim:width,
        pre:{weight:"pre",epsilon:1e-6},post:{weight:"post",epsilon:1e-6},
        v:projection("v"),o:projection("o"),gate:projection("gate"),up:projection("up")};
      const weight=async(name:string,_index:number)=>name==="pre"||name==="post"?1:1/256;
      let path=new FlatConditions();
      for(let p=0;p<3;p++)for(let c=0;c<width;c++){
        const source=`input_tokens[${p}][${c}]`,key=createHash("sha256").update(source).digest("hex");
        path=path.refine(key,()=>s.write(source),{lower:{value:exactNumberRational(-(2**-24)),inclusive:true},
          upper:{value:exactNumberRational(2**-24),inclusive:true}},true)!;
      }
      for(let query=0;query<3;query++){
        assert.equal(await proveInitialMlpProductZero(f,path,g,query,0,weight),true,`${width},${query}`);
        assert.equal(await proveInitialMlpProductZero(f,new FlatConditions(),g,query,0,weight),false);
      }
      for(const gamma of [-0.25,0,0.25]){
        const learned=async(name:string,index:number)=>name==="pre"||name==="post"?gamma:weight(name,index);
        assert.equal(await proveInitialMlpProductZero(f,path,g,2,0,learned),true,`${width},gamma=${gamma}`);
      }
      const identityGeometry={...g,down:projection("down")};
      const tiny=async(name:string,_index:number)=>name==="pre"||name==="post"?1:1/1024;
      for(let query=0;query<3;query++){
        assert.equal(await proveInitialHiddenIdentity(f,path,identityGeometry,query,0,tiny),true,`identity ${width},${query}`);
        assert.equal(await proveInitialHiddenIdentity(f,new FlatConditions(),identityGeometry,query,0,tiny),false);
      }
      assert.equal(await proveInitialHiddenIdentity(f,path,identityGeometry,0,0,
        async(name,index)=>name==="down"?NaN:tiny(name,index)),false);
    }
    await s.close();
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("a proved signed-zero input eliminates its dependency in the ordered projection",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-zero-term-"));
  try{
    const s=new DirectRustStream(join(dir,"unused.rs")),f=new DirectFlatSubstitution(s);
    for(const width of [1,2,3,5,9]){
      let proofs=0,visits=0;
      await substituteFlatProjection(f,new FlatConditions(),width,async c=>c%2?-65504:65504,
        ()=>async()=>{throw new Error("Proved zero dependency visited");},async(_p,value)=>{
          visits++;assert.ok(Object.is(value.literal,0));
        },async()=>{proofs++;return true;});
      assert.equal(proofs,width);assert.equal(visits,1);
    }
    await s.close();
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("nonzero residual corrections disappear when both rounded sums retain the embedding",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-absorbed-residual-"));
  try{
    const s=new DirectRustStream(join(dir,"unused.rs")),f=new DirectFlatSubstitution(s);
    for(const width of [2,4,8]){
      const projection=(weight:string)=>({weight,shape:[width,width] as [number,number]});
      const g={width,context:3,heads:1,kvHeads:1,headDim:width,
        pre:{weight:"pre",epsilon:1e-6},post:{weight:"post",epsilon:1e-6},
        v:projection("v"),o:projection("o"),gate:projection("gate"),up:projection("up"),down:projection("down")};
      const weight=async(name:string,_index:number)=>name==="pre"||name==="post"?1:1/1024;
      for(let query=0;query<3;query++)for(const sign of [-1,1]){
        const source=`input_tokens[${query}][0]`,key=createHash("sha256").update(source).digest("hex");
        const path=new FlatConditions().refine(key,()=>s.write(source),sign>0?
          {lower:{value:exactNumberRational(4),inclusive:true}}:
          {upper:{value:exactNumberRational(-4),inclusive:true}},true)!;
        assert.equal(await proveInitialHiddenIdentity(f,path,g,query,0,weight),true,`${width},${query},${sign}`);
        const nearZero=new FlatConditions().refine(key,()=>s.write(source),{
          lower:{value:exactNumberRational(2**-24),inclusive:true},
          upper:{value:exactNumberRational(2**-24),inclusive:true}},true)!;
        assert.equal(await proveInitialHiddenIdentity(f,nearZero,g,query,0,weight),false);
        assert.equal(await proveInitialHiddenIdentity(f,path,g,query,0,
          async(name,index)=>name==="down"?NaN:weight(name,index)),false);
      }
    }
    await s.close();
  }finally{await rm(dir,{recursive:true,force:true});}
});
