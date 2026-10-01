import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, appendFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { DirectRustStream } from "../src/direct-rust-stream.js";
import { DirectFlatSubstitution, FlatConditions, type FlatProducer } from "../src/direct-flat-substitution.js";
import { rational } from "../src/direct-branch-domain.js";
import { substituteCpuArm64F32Sum } from "../src/direct-rust-mean.js";
import { emitRustSilu } from "../src/direct-rust-numeric.js";
import { substituteFlatProjection } from "../src/direct-flat-projection.js";
const run=promisify(execFile);

test("zero projection weights eliminate their dependencies before they are expanded",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-zero-weights-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    for(const width of [1,2,3,5,9]){
      let reads=0,visits=0;
      await substituteFlatProjection(f,new FlatConditions(),width,async c=>{reads++;return c%2?-0:0;},
        ()=>async()=>{throw new Error("Zero-weight dependency must be eliminated");},async(_p,v)=>{
          visits++;assert.ok(Object.is(v.literal,0));
        });
      assert.equal(reads,width);assert.equal(visits,1);
    }
    await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("SiLU enumeration starts inside the inherited positive or negative interval",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-silu-domain-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs"));
    for(const point of [0.5,-0.5]){
      let visits=0;
      await emitRustSilu(s,()=>s.write("x"),Math.abs(point),async run=>{
        visits++;assert.equal(run.minimum,point);assert.equal(run.maximum,point);
      },{minimum:point,maximum:point});
      assert.equal(visits,1);
    }
    assert.equal(s.bytes,0);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("later operand restrictions simplify the earlier operand before the next numerical expansion",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-refresh-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    const right:FlatProducer=(p,k)=>f.comparison(p,x,">=",rational(1n),
      p=>f.comparison(p,x,"<=",rational(1n),p=>f.literal(p,2,k),async()=>{}),async()=>{});
    let visited=0;
    await f.binary(new FlatConditions(),x,right,"+",async(_p,value)=>{
      visited++;assert.equal(value.literal,3);assert.equal(value.minimum,3);assert.equal(value.maximum,3);
    });
    assert.equal(visited,1);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("zero nonnegative reductions propagate to original squared inputs",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-zero-squares-"));
  try{
    const path=join(directory,"zero.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    const coordinate=(i:number):FlatProducer=>(p,k)=>f.f16Input(p,()=>s.write(`x[${i}]`),k);
    const squares:FlatProducer=(p,k)=>substituteCpuArm64F32Sum(f,p,2,i=>(p,k)=>f.square(p,coordinate(i),k),k);
    await s.write("fn generated(x:&[f64])->f64 {'answer:{");s.beginReducedExpression();
    await f.comparison(new FlatConditions(),squares,"<=",rational(0n),
      (p)=>f.literal(p,1,(p,v)=>f.leaf(p,"answer",v)),p=>f.literal(p,0,(p,v)=>f.leaf(p,"answer",v)));
    await f.finishRound();await s.write('panic!("outside domain")}}');await s.close();
    await appendFile(path,'fn main(){for a in [-2_f64.powi(-24),-0.0,0.0,2_f64.powi(-24),1.0]{for b in [-2_f64.powi(-24),-0.0,0.0,2_f64.powi(-24),1.0]{assert_eq!(generated(&[a,b]),if a==0.0&&b==0.0{1.0}else{0.0});}}}');
    await run("rustc",["-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
    const source=await readFile(path,"utf8");assert.match(source,/x\[0\]/);assert.match(source,/x\[1\]/);
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("consumer comparisons propagate through rounded and affine producers without manual normalization",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-preimages-"));
  try{
    const path=join(directory,"preimages.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.input(p,()=>s.write("x"),0,10,k);
    const four:FlatProducer=(p,k)=>f.literal(p,4,k);
    const rounded:FlatProducer=(p,k)=>f.round(p,(p,k)=>f.binary(p,x,four,"*",k),"f32",false,k);
    const result=(p:FlatConditions,value:number)=>f.literal(p,value,(p,v)=>f.leaf(p,"answer",v));
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    await f.comparison(new FlatConditions(),x,">",rational(3n),
      p=>f.comparison(p,rounded,"<",rational(5n),p=>result(p,999),p=>result(p,7)),
      p=>result(p,0));
    await f.finishRound();await s.write('panic!("outside domain")}}');await s.close();
    const source=await readFile(path,"utf8");assert.doesNotMatch(source,/999|4503599627370496/);
    await appendFile(path,'fn main(){for i in 0..=20000{let x=(i as f64)/2000.0;assert_eq!(generated(x),if x>3.0{7.0}else{0.0});}}');
    await run("rustc",["-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("activation substitution passes its branches to a rounded consumer without inline conditionals",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-activation-"));
  try{
    const path=join(directory,"activation.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.input(p,()=>s.write("x"),0.1,0.1001,(p,v)=>k(p,{...v,precision:"f16"}));
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    await f.round(new FlatConditions(),(p,k)=>f.silu(p,x,k),"f16",true,(p,v)=>f.leaf(p,"answer",v));
    await f.finishRound();await s.write('panic!("outside domain")}}');await s.close();
    await appendFile(path,'fn decode(b:u16)->f64{let e=b>>10;let m=b&1023;if e==0{(m as f64)*2_f64.powi(-24)}else{(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)}}fn main(){let data=std::fs::read(std::env::args().nth(1).unwrap()).unwrap();for b in 0..31744_u16{let x=decode(b);if x>=0.1&&x<=0.1001{let i=b as usize*2;let expected=decode(u16::from_le_bytes(data[i..i+2].try_into().unwrap()));assert_eq!(generated(x).to_bits(),expected.to_bits());}}}');
    await run("rustc",["--edition=2021","-Awarnings",path,"-o",join(directory,"run")]);
    await run(join(directory,"run"),[new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin",import.meta.url).pathname]);
    assert.ok(s.eliminatedBranches>0);
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("nested rounded producers substitute into flat consumers before source emission",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-consumers-"));
  try{
    const path=join(directory,"flat.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    const left:FlatProducer=(p,k)=>f.round(p,(p,k)=>f.input(p,()=>s.write("x/3.0"),1/3,2/3,k),"f32",false,k);
    const right:FlatProducer=(p,k)=>f.round(p,(p,k)=>f.input(p,()=>s.write("x*7.0"),7,14,k),"f32",false,k);
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    await f.round(new FlatConditions(),(p,k)=>f.binary(p,left,right,"+",k),"f32",false,
      (p,value)=>f.leaf(p,"answer",value));
    await f.finishRound();await s.write('panic!("outside domain")}}');await s.close();
    const source=await readFile(path,"utf8");
    assert.doesNotMatch(source,/\blet\b|\b(?:sqrt|exp|f32|f16|while|for|as)\b/);
    assert.match(source,/if/);assert.doesNotMatch(source,/else/);
    await appendFile(path,'fn main(){let mut state=97_u32;for _ in 0..20000{state=state.wrapping_mul(1664525).wrapping_add(1013904223);let x=1.0+(state as f64)/(u32::MAX as f64);let expected=(((x/3.0) as f32)+((x*7.0) as f32)) as f64;assert_eq!(generated(x).to_bits(),expected.to_bits(),"{}",x);}}');
    await run("rustc",["-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("flat double rounding preserves signed zero, subnormals and half midpoint neighborhoods",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-half-"));
  try{
    const path=join(directory,"half.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    await f.round(new FlatConditions(),(p,k)=>f.input(p,()=>s.write("x"),-65519,65519,k),"f16",false,
      (p,value)=>f.leaf(p,"answer",value));
    await f.finishRound();await s.write('panic!("outside domain")}}');await s.close();
    await appendFile(path,'fn decode(b:u32)->f64{let e=b>>10;let m=b&1023;if e==0{(m as f64)*2_f64.powi(-24)}else{(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)}}fn main(){assert_eq!(generated(-0.0).to_bits(),(-0.0_f64).to_bits());assert_eq!(generated(0.0).to_bits(),0.0_f64.to_bits());for b in 0..31743{let lo=decode(b);let hi=decode(b+1);let m=(lo+hi)/2.0;let code=(m as f32).to_bits();let cut=if b&1==0{m+(f32::from_bits(code+1) as f64-m)/2.0}else{m-(m-f32::from_bits(code-1) as f64)/2.0};for (x,y) in [(f64::from_bits(cut.to_bits()-1),lo),(cut,if b&1==0{lo}else{hi}),(f64::from_bits(cut.to_bits()+1),hi)]{for sign in [1.0,-1.0]{assert_eq!(generated(sign*x).to_bits(),(sign*y).to_bits(),"{} {}",b,sign*x);}}}}');
    await run("rustc",["-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
  }finally{await rm(directory,{recursive:true,force:true});}
});
