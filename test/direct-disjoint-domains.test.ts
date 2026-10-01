import assert from "node:assert/strict";
import {test} from "node:test";
import {mkdtemp,rm,readFile,appendFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {createHash} from "node:crypto";
import {execFile} from "node:child_process";
import {promisify} from "node:util";
import {DirectFlatSubstitution,FlatConditions,type FlatProducer} from "../src/direct-flat-substitution.js";
import {DirectRustStream} from "../src/direct-rust-stream.js";
import {exactNumberRational as exact} from "../src/direct-round-preimage.js";
import {DirectBranchDomain,type Interval} from "../src/direct-branch-domain.js";
const run=promisify(execFile);

test("disjoint domains preserve strict holes, intersect both sides and remain immutable",()=>{
  const point=(x:number):Interval=>({lower:{value:exact(x),inclusive:true},upper:{value:exact(x),inclusive:true}});
  const base=new DirectBranchDomain();
  const separated=base.intersectRegions("x",[
    {upper:{value:exact(-2),inclusive:true}}, {lower:{value:exact(2),inclusive:true}},
  ])!;
  assert.equal(separated.split("x",">",exact(-1)).truth!.split("x","<",exact(1)).truth,undefined);
  assert.equal(separated.regions("x").length,2);
  assert.equal(base.regions("x").length,1);
  assert.equal(separated.intersectRegions("x",[point(-2)])!.regions("x").length,1);
  assert.equal(separated.intersectRegions("x",[point(2)])!.regions("x").length,1);
  assert.equal(separated.intersectRegions("x",[point(0)]),undefined);
  const missingZero=base.intersectRegions("x",[
    {upper:{value:exact(0),inclusive:false}},{lower:{value:exact(0),inclusive:false}},
  ])!;
  assert.equal(missingZero.regions("x").length,2);
  assert.equal(missingZero.intersectRegions("x",[point(-0)]),undefined);
  const coveredZero=base.intersectRegions("x",[
    {upper:{value:exact(0),inclusive:true}},{lower:{value:exact(0),inclusive:false}},
  ])!;
  assert.equal(coveredZero.regions("x").length,1);
  assert.equal(coveredZero.intersectRegions("x",[point(0)])!.regions("x").length,1);
});

test("square preimages discard the central impossible interval before its consumer",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-disjoint-square-"));
  try{
    const s=new DirectRustStream(join(dir,"unused.rs")),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    let impossible=0;
    const ranges:number[][]=[];
    await f.comparison(new FlatConditions(),(p,k)=>f.square(p,x,k),">=",exact(4),async p=>{
      await x(p,async(_p,v)=>{ranges.push([v.minimum,v.maximum]);});
      await f.comparison(p,x,">",exact(-1),async p=>{
        await f.comparison(p,x,"<",exact(1),async()=>{impossible++;},async()=>{});
      },async()=>{});
    },async()=>{});
    await s.close();assert.equal(impossible,0);
    assert.equal(ranges.some(([low,high])=>low!<2&&high!> -2),false);
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("joint finite proofs retain separated survivors before the next producer",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-disjoint-survivors-"));
  try{
    const s=new DirectRustStream(join(dir,"unused.rs")),f=new DirectFlatSubstitution(s),q=2**-24;
    const source="input_tokens[0][0]",hash=(text:string)=>createHash("sha256").update(text).digest("hex");
    let base=new FlatConditions().refine(hash(source),()=>s.write(source),{
      lower:{value:exact(0),inclusive:true},upper:{value:exact(4*q),inclusive:true}},true)!;
    const product=`(${source}*(${source}-${4*q}_f64))`;
    base=base.refine(hash(product),()=>s.write(product),{
      lower:{value:exact(0),inclusive:true},upper:{value:exact(0),inclusive:true}})!;
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write(source),k);
    const ranges:number[][]=[];
    await f.comparison(base,x,">=",exact(0),async p=>{
      await x(p,async(_p,v)=>{ranges.push([v.minimum,v.maximum]);});
    },async()=>{assert.fail("Nonnegative input took negative path");});
    await s.close();
    assert.equal(ranges.some(([low,high])=>low!<=2*q&&high!>=2*q),false);
    assert.equal(ranges.some(([low,high])=>low===0&&high===0),true);
    assert.equal(ranges.some(([low,high])=>low===4*q&&high===4*q),true);
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("mixed-sign addition and subtraction propagate endpoint preimages before expansion",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-signed-preimages-"));
  try{
    const s=new DirectRustStream(join(dir,"unused.rs")),f=new DirectFlatSubstitution(s);
    const hash=(text:string)=>createHash("sha256").update(text).digest("hex");
    for(const operator of ["+","-"] as const){
      let base=new FlatConditions();
      for(const [c,low,high] of [[0,-4,4],[1,-2,-1]]){
        const source=`input_tokens[0][${c}]`;
        base=base.refine(hash(source),()=>s.write(source),{
          lower:{value:exact(low!),inclusive:true},upper:{value:exact(high!),inclusive:true}},true)!;
      }
      const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
      const y:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][1]"),k);
      let reached=0;
      await f.comparison(base,(p,k)=>f.binary(p,x,y,operator,k),">",exact(1),async p=>{
        await x(p,async(_p,v)=>{reached++;assert.ok(v.minimum>(operator==="+"?2:-1));});
      },async()=>{});
      assert.ok(reached>0);
    }
    await s.close();
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("Rust executes square interval preimages bitwise over every finite half input",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-disjoint-rust-"));
  try{
    const file=join(dir,"square.rs"),s=new DirectRustStream(file),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    const square:FlatProducer=(p,k)=>f.square(p,x,k);
    await s.write("fn generated(input_tokens:&[[f64;1]])->f64 {'answer:{");s.beginReducedExpression();
    const outside=(p:FlatConditions)=>f.binary(p,x,(p,k)=>f.literal(p,0,k),"*",(p,v)=>f.leaf(p,"answer",v));
    await f.comparison(new FlatConditions(),square,">=",exact(4),p=>f.comparison(p,square,"<=",exact(9),
      p=>x(p,(p,v)=>f.leaf(p,"answer",v)),outside),outside);
    await f.finishRound();await s.write('panic!("outside input domain")}}');await s.close();
    const source=await readFile(file,"utf8");assert.doesNotMatch(source,/\blet\b|\b(?:while|for)\b|else|\bas\b/);
    await appendFile(file,'fn decode(b:u32)->f64{let sign=if b&32768==0{1.0}else{-1.0};let e=(b>>10)&31;let m=b&1023;sign*if e==0{m as f64*2_f64.powi(-24)}else{(1.0+m as f64/1024.0)*2_f64.powi(e as i32-15)}}fn main(){for bits in 0..65536{if bits&31744==31744{continue;}let x=decode(bits);let expected=if x*x>=4.0&&x*x<=9.0{x}else{x*0.0};for length in [1,2,4]{let rows=vec![[x];length];assert_eq!(generated(&rows).to_bits(),expected.to_bits(),"{} {}",bits,length);}}}');
    await run("rustc",["--edition=2021","-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("flat union guards do not fill holes or remove a condition using its hull",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-disjoint-guards-"));
  try{
    const file=join(dir,"guards.rs"),s=new DirectRustStream(file),f=new DirectFlatSubstitution(s);
    const key=createHash("sha256").update("x").digest("hex");
    const base=new FlatConditions().refine(key,()=>s.write("x"),{
      lower:{value:exact(-2),inclusive:true},upper:{value:exact(2),inclusive:true}})!;
    const separated=base.refineRegions(key,()=>s.write("x"),[
      {upper:{value:exact(-1),inclusive:false}},{lower:{value:exact(1),inclusive:true}},
    ])!;
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    await f.literal(separated,3,(p,v)=>f.leaf(p,"answer",v));
    await f.literal(base,7,(p,v)=>f.leaf(p,"answer",v));await f.finishRound();
    await s.write('panic!("outside domain")}}');
    await s.write("fn affine(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    const domain=new DirectBranchDomain().split("x",">=",exact(-2)).truth!.split("x","<=",exact(2)).truth!;
    const disjoint=domain.intersectRegions("x",[
      {upper:{value:exact(-1),inclusive:false}},{lower:{value:exact(1),inclusive:true}},
    ])!;
    await s.affineLeaf(disjoint,"answer",()=>s.write("3.0"),domain);
    await s.write("break 'answer 7.0;}}");await s.close();
    assert.match(await readFile(file,"utf8"),/\|\|/);
    await appendFile(file,'fn main(){for x in [-2.0,-1.00001,-1.0,-0.0,0.0,0.5,1.0,2.0]{let expected=if x< -1.0||x>=1.0{3.0}else{7.0};assert_eq!(generated(x),expected);assert_eq!(affine(x),expected);}}');
    await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});

test("signed endpoint preimages preserve every comparison and both zero signs in Rust",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-signed-preimages-rust-"));
  try{
    const file=join(dir,"signed.rs"),s=new DirectRustStream(file),f=new DirectFlatSubstitution(s);
    const hash=(text:string)=>createHash("sha256").update(text).digest("hex");
    let base=new FlatConditions();
    for(const [coordinate,limit] of [[0,4],[1,2]]){
      const name=`input_tokens[0][${coordinate}]`;
      base=base.refine(hash(name),()=>s.write(name),{
        lower:{value:exact(-limit!),inclusive:true},upper:{value:exact(limit!),inclusive:true}},true)!;
    }
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    const y:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][1]"),k);
    const cases:{name:string;operator:string;op:string;rhs:number}[]=[];
    for(const operator of ["+","-"] as const)for(const op of ["<","<=",">",">="] as const)
      for(const rhs of [-3,-(2**-24),0,2**-24,3]){
        const name=`case_${cases.length}`;cases.push({name,operator,op,rhs});
        await s.write(`fn ${name}(input_tokens:&[[f64;2]])->f64 {'answer:{`);s.beginReducedExpression();
        await f.comparison(base,(p,k)=>f.binary(p,x,y,operator,k),op,exact(rhs),
          (p)=>x(p,(p,v)=>f.leaf(p,"answer",v)),(p)=>y(p,(p,v)=>f.leaf(p,"answer",v)));
        await f.finishRound();await s.write('panic!("outside input domain")}}');
      }
    await s.close();
    const checks=cases.map(({name,operator,op,rhs})=>
      `assert_eq!(${name}(&[[x,y]]).to_bits(),(if x ${operator} y ${op} ${rhs}_f64{x}else{y}).to_bits(),"${name} {} {}",bits,y);`).join("");
    await appendFile(file,`fn decode(b:u32)->f64{let sign=if b&32768==0{1.0}else{-1.0};let e=(b>>10)&31;let m=b&1023;sign*if e==0{m as f64*2_f64.powi(-24)}else{(1.0+m as f64/1024.0)*2_f64.powi(e as i32-15)}}fn main(){for bits in 0..65536{if bits&31744==31744{continue;}let x=decode(bits);if x.abs()>4.0{continue;}for y in [-2.0,-1.0,-${2**-24}_f64,-0.0,0.0,${2**-24},1.0,2.0]{${checks}}}}`);
    await run("rustc",["--edition=2021","-Awarnings","-C","opt-level=1",file,"-o",join(dir,"run")]);
    await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
