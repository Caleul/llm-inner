import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { DirectRustStream } from "../src/direct-rust-stream.js";
import { DirectBranchDomain, rational } from "../src/direct-branch-domain.js";
import { emitRustExp, emitRustSilu, emitRustSqrt } from "../src/direct-rust-numeric.js";
const run = promisify(execFile);
test("propagated activation bounds eliminate unreachable scalar branches exactly",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-activation-range-"));
  try{
    const file=join(dir,"activation.rs"),s=new DirectRustStream(file);
    await s.write("fn full_exp(x:f64)->f64 {");await emitRustExp(s,()=>s.write("x"),true);
    await s.write("}fn narrow_exp(x:f64)->f64 {");await emitRustExp(s,()=>s.write("x"),true,0.34);
    await s.write("}fn full_silu(x:f64)->f64 {");await emitRustSilu(s,()=>s.write("x"));
    await s.write("}fn narrow_silu(x:f64)->f64 {");await emitRustSilu(s,()=>s.write("x"),0.1);
    await s.write("}fn main(){for i in 0..=20000 {let x=((i as f64)*(-0.3399)/20000.0) as f32 as f64;assert_eq!(full_exp(x).to_bits(),narrow_exp(x).to_bits());}for bits in 0_u32..31744 {let e=bits>>10;let m=bits&1023;let x=if e==0 {(m as f64)*2_f64.powi(-24)}else{(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)};if x>0.1 {break;}for sign in [-1.0,1.0]{assert_eq!(full_silu(sign*x).to_bits(),narrow_silu(sign*x).to_bits());}}}");
    await s.close();assert.ok(s.eliminatedBranches>0);
    const source=await readFile(file,"utf8");const narrow=source.split("fn narrow_exp")[1]!.split("fn full_silu")[0]!;
    assert.doesNotMatch(narrow,/let (?:mut )?(?:exponent|scaled|remainder|first|second):f64/);
    await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("inline numeric scratch survives recursive producers and independent consumers",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-inline-scratch-"));
  try{
    const file=join(dir,"numeric.rs"),s=new DirectRustStream(file);
    const expression=async()=>{
      await s.write("(");
      await emitRustSilu(s,()=>s.round("f16",()=>emitRustSqrt(s,()=>emitRustExp(s,()=>s.write("x")))));
      await s.write("+");await emitRustSilu(s,()=>s.round("f16",()=>emitRustExp(s,()=>s.write("-x"))));
      await s.write(")");
    };
    await s.write("fn scoped(x:f64)->f64 {");await expression();await s.write("}fn reused(x:f64)->f64 {");
    await s.declareRoundingScratch();await s.declareInlineNumericScratch();await expression();
    await s.write("}fn main(){for i in -1000..=1000 {let x=(i as f64)/200.0;assert_eq!(scoped(x).to_bits(),reused(x).to_bits(),\"input {}\",x);}}");
    await s.close();await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("inline half rounding preserves every finite adjacent midpoint and ties to even",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-half-midpoint-"));
  try{
    const file=join(dir,"half.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");await s.declareRoundingScratch();
    await s.round("f16",()=>s.write("x"),true);
    await s.write("}fn decode(bits:u32)->f64 {let e=bits>>10;let m=bits&1023;if e==0 {(m as f64)*2_f64.powi(-24)} else {(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)}}fn main(){for bits in 0_u32..31743 {let lower=decode(bits);let upper=decode(bits+1);let midpoint=(lower+upper)/2.0;let even=if bits&1==0 {lower} else {upper};for (x,expected) in [(lower,lower),(midpoint,even),(f64::from_bits(midpoint.to_bits()-1),lower),(f64::from_bits(midpoint.to_bits()+1),upper)] {for sign in [1.0,-1.0] {assert_eq!(generated(sign*x).to_bits(),(sign*expected).to_bits());}}}assert_eq!(generated(65520.0),f64::INFINITY);assert_eq!(generated(-65520.0),f64::NEG_INFINITY);}");
    await s.close();await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("Rust executes flattened exact rounding over all finite half values and midpoint neighbors", async () => {
  const dir = await mkdtemp(join(tmpdir(), "direct-rust-"));
  try {
    const file = join(dir, "round.rs"), stream = new DirectRustStream(file);
    await stream.write("fn generated(x:f64)->f64 {");
    await stream.round("f32", () => stream.write("x"));
    await stream.write("}\nfn main(){for bits in 0_u32..65536 {let sign=if bits&32768!=0 {-1.0} else {1.0};let e=(bits>>10)&31;let m=bits&1023;if e==31 {continue;}let x=if e==0 {sign*(m as f64)*2_f64.powi(-24)} else {sign*(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)};for y in [x,x+2_f64.powi(-25),x-2_f64.powi(-25)] {assert_eq!(generated(y).to_bits(),((y as f32) as f64).to_bits());}}for y in [-0.0,0.0,1.0000000596046448,1.0000001788139343,3.4028235677973366e38,1.401298464324817e-45,7.006492321624085e-46] {assert_eq!(generated(y).to_bits(),((y as f32) as f64).to_bits());}println!(\"rounding ok\");}");
    await stream.close();
    const source = await readFile(file,"utf8");
    assert.doesNotMatch(source.split("fn main")[0]!, /\b(?:f16|f32Bits|exp|sqrt|round|floor|powi)\(/);
    await run("rustc", ["--edition=2021", "-Awarnings", file, "-o", join(dir,"run")]);
    assert.equal((await run(join(dir,"run"))).stdout.trim(), "rounding ok");
  } finally { await rm(dir,{recursive:true,force:true}); }
});
test("inherited affine conditions eliminate impossible and redundant Rust branches before emission", async () => {
  const dir = await mkdtemp(join(tmpdir(), "direct-domain-"));
  try {
    const file=join(dir,"branch.rs"), stream=new DirectRustStream(file);
    await stream.write("fn generated(x:f64)->i32 {");
    await stream.exactAffineBranch(new DirectBranchDomain(),"x",rational(1n),rational(0n),">",rational(3n),
      domain => stream.exactAffineBranch(domain,"x",rational(4n),rational(0n),"<",rational(5n),
        () => stream.write("999"),
        narrowed => stream.exactAffineBranch(narrowed,"x",rational(-2n),rational(1n),"<",rational(-5n),
          () => stream.write("7"), () => stream.write("888"))),
      () => stream.write("0"));
    await stream.write("}\nfn main(){assert_eq!(generated(4.0),7);assert_eq!(generated(3.0),0);}");
    await stream.close();
    const source=await readFile(file,"utf8");
    assert.doesNotMatch(source,/999|888/);assert.equal((source.match(/\bif\b/g)??[]).length,1);
    assert.equal(stream.eliminatedBranches,2);
    await run("rustc",[file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  } finally {await rm(dir,{recursive:true,force:true});}
});
test("reachable transformed branches flatten into reduced leaf conditions",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-flat-"));
  try {
    const file=join(dir,"flat.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->i32 {'result:{");
    const leaf=(domain:DirectBranchDomain,value:string)=>s.affineLeaf(domain,"result",()=>s.write(value));
    await s.flattenedAffinePaths(new DirectBranchDomain(),"x",rational(1n),rational(0n),">",rational(3n),
      domain=>s.flattenedAffinePaths(domain,"x",rational(4n),rational(0n),"<",rational(100n),
        inner=>leaf(inner,"7"),inner=>leaf(inner,"8")),domain=>leaf(domain,"0"));
    await s.write("panic!(\"finite input required\")}}fn main(){for (x,v) in [(2.0,0),(4.0,7),(25.0,8),(100.0,8)] {assert_eq!(generated(x),v);}}");
    await s.close();const source=await readFile(file,"utf8");
    assert.match(source,/x>=\(25\.0_f64\/1\.0_f64\)/);
    assert.doesNotMatch(source,/else/);
    await run("rustc",[file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("operator scratch preserves nested rounding without retaining producer results",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-scratch-"));
  try{
    const file=join(dir,"scratch.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");await s.declareRoundingScratch();
    await s.round("f32",async()=>{await s.round("f32",()=>s.write("x/3.0"));await s.write("+");await s.round("f32",()=>s.write("x*7.0"));});
    await s.write("}fn main(){for x in [-0.0,0.0,-1.0,1.0,65504.0,0.000000059604644775390625] {let expected=(((x/3.0) as f32)+((x*7.0) as f32)) as f64;assert_eq!(generated(x).to_bits(),expected.to_bits());}}");
    await s.close();await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("positive interval facts remove impossible rounding branches without changing midpoint ties",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-round-range-"));
  try{
    const file=join(dir,"range.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");await s.declareRoundingScratch();
    await s.round("f32",()=>s.write("x"),false,{minimum:2**-12,maximum:2});
    await s.write("}fn main(){let mut state=97_u32;for _ in 0..20000 {state=state.wrapping_mul(1664525).wrapping_add(1013904223);let bits=0x39800000+(state%(0x40000000-0x39800000-1));let lower=f32::from_bits(bits) as f64;let upper=f32::from_bits(bits+1) as f64;let midpoint=(lower+upper)/2.0;for x in [lower,midpoint,f64::from_bits(midpoint.to_bits()-1),f64::from_bits(midpoint.to_bits()+1)] {assert_eq!(generated(x).to_bits(),((x as f32) as f64).to_bits());}}}");
    await s.close();const source=(await readFile(file,"utf8")).split("fn main")[0]!;
    assert.doesNotMatch(source,/round_negative\s*=|round_normal=|round_exponent=|INFINITY/);
    await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
