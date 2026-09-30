import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { DirectRustStream } from "../src/direct-rust-stream.js";
import { DirectBranchDomain, rational } from "../src/direct-branch-domain.js";
import { DirectReductionGate, IncompleteDirectReduction } from "../src/direct-reduction-gate.js";
import { foldCpuArm64F32Sum } from "../src/direct-rust-mean.js";
import { emitRustExp, emitRustSilu, emitRustSqrt } from "../src/direct-rust-numeric.js";
const run = promisify(execFile);
test("normalization sum preserves declared CPU vector, cascade and tail order",async()=>{
  const expected=new Map([[1,999256192],[2,1141247376],[3,1042039202],[4,1167816342],
    [5,1157906436],[7,1111965965],[8,1157503443],[16,1202506296],[17,1210558463],
    [63,1203298570],[64,1212626310],[65,1224028867],[257,1233566697],
    [1024,1250178836],[4096,1270696017],[16385,1286460092]]);
  // Golden F32 sums measured independently with torch 2.12.1 CPU arm64.
  // Inputs are reproducible, have mixed magnitudes and cross cascade/tail
  // boundaries; ordinary sequential/pairwise sums do not satisfy this gate.
  let seed=97531;const bits=new DataView(new ArrayBuffer(4));
  for(const [size,reference] of expected){
    const input:number[]=[];
    for(let i=0;i<size;i++){
      seed=(Math.imul(seed,1664525)+1013904223)>>>0;
      const x=(seed&1023)*2**((seed>>>10)%20-20);input.push(Math.fround(x*x));
    }
    const result=await foldCpuArm64F32Sum(size,async i=>input[i]!);
    bits.setFloat32(0,result,true);assert.equal(bits.getUint32(0,true),reference,`width ${size}`);
  }
});
test("square-root substitution emits only exact affine leaves on its F32 input domain",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-root-substitution-"));
  try{
    const file=join(dir,"root.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");s.beginReducedExpression();
    await emitRustSqrt(s,()=>s.write("x"),{minimum:1,maximum:1.001});await s.write("}");await s.close();
    const source=await readFile(file,"utf8");assert.doesNotMatch(source,/\blet\b|\b(?:while|for|sqrt)\b/);
    const {appendFile}=await import("node:fs/promises");
    await appendFile(file,"fn main(){for bits in 0x3f800000_u32..=0x3f8020c4 {let x=f32::from_bits(bits);if (x as f64)>1.001{break;}assert_eq!(generated(x as f64).to_bits(),(x.sqrt() as f64).to_bits(),\"input {}\",x);}}");
    await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("fully substituted SiLU matches every finite F16 backend-policy input",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-silu-substitution-"));
  try{
    const file=join(dir,"activation.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");s.beginReducedExpression();await emitRustSilu(s,()=>s.write("x"));await s.write("}");await s.close();
    const source=await readFile(file,"utf8");assert.doesNotMatch(source,/\blet\b|\b(?:while|for)\b|\bbits\b|half_decode|half_encode/);
    const {appendFile}=await import("node:fs/promises");
    await appendFile(file,"fn decode(bits:u16)->f64{let sign=if bits&32768!=0{-1.0}else{1.0};let e=(bits>>10)&31;let m=bits&1023;sign*if e==0{(m as f64)*2_f64.powi(-24)}else{(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)}}fn main(){let bytes=std::fs::read(std::env::args().nth(1).unwrap()).unwrap();for bits in 0_u32..65536{if bits&31744==31744{continue;}let at=(bits as usize)*2;let expected=decode(u16::from_le_bytes(bytes[at..at+2].try_into().unwrap()));assert_eq!(generated(decode(bits as u16)).to_bits(),expected.to_bits(),\"input code {}\",bits);}}");
    await run("rustc",["--edition=2021","-Awarnings",file,"-o",join(dir,"run")]);
    await run(join(dir,"run"),[new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin",import.meta.url).pathname]);
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("reduction admission rejects intermediaries and loops across streaming chunk boundaries",()=>{
  for(const word of ["let","mut","while","for","round_input","root","as"]){
    if(word==="root")continue;
    const gate=new DirectReductionGate();assert.throws(()=>{for(const c of word+" ")gate.accept(c);gate.finish();},IncompleteDirectReduction);
  }
  for(const source of ["if x>0 {if y>0 {1}}", "if ('value:{if y>0 {break 'value 1;}0})>0 {1}"]){
    const gate=new DirectReductionGate();assert.throws(()=>{for(const c of source)gate.accept(c);gate.finish();},IncompleteDirectReduction);
  }
  const gate=new DirectReductionGate();for(const c of '/* let mut */ if x>0.0 {break \'value 1.0;} panic!("while")}')gate.accept(c);gate.finish();
});
test("composed F64 to F32 to F16 is substituted through every half midpoint preimage",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-composed-half-"));
  try{
    const file=join(dir,"composed.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");s.beginReducedExpression();await s.round("f16",()=>s.write("x"));
    await s.write("}");await s.close();
    const source=await readFile(file,"utf8");assert.doesNotMatch(source,/\blet\b|\b(?:while|for)\b|round_/);
    const {appendFile}=await import("node:fs/promises");
    await appendFile(file,"fn decode(bits:u32)->f64{let e=bits>>10;let m=bits&1023;if e==0{(m as f64)*2_f64.powi(-24)}else{(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)}}fn main(){for bits in 0_u32..31743{let lo=decode(bits);let hi=decode(bits+1);let midpoint=(lo+hi)/2.0;let m=(midpoint as f32).to_bits();let cut=if bits&1==0{midpoint+((f32::from_bits(m+1) as f64)-midpoint)/2.0}else{midpoint-(midpoint-(f32::from_bits(m-1) as f64))/2.0};let at=if bits&1==0{lo}else{hi};for (x,y) in [(f64::from_bits(cut.to_bits()-1),lo),(cut,at),(f64::from_bits(cut.to_bits()+1),hi)]{for sign in [1.0,-1.0]{assert_eq!(generated(sign*x).to_bits(),(sign*y).to_bits(),\"half code {} input {}\",bits,sign*x);}}}assert_eq!(generated(-0.0).to_bits(),(-0.0_f64).to_bits());for x in [65519.998046875,65520.0,1e300]{assert_eq!(generated(x),f64::INFINITY);assert_eq!(generated(-x),f64::NEG_INFINITY);}}");
    await run("rustc",["-Awarnings","-C","opt-level=2",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("exponential substitution removes numerical temporaries before Rust compilation",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-exp-substitution-"));
  try{
    const file=join(dir,"exponential.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");s.beginReducedExpression();
    await emitRustExp(s,()=>s.write("x"),true,undefined,{minimum:-0.10001,maximum:-0.1});
    await s.write("}fn lattice(x:f64)->f64 {");
    await emitRustExp(s,()=>s.write("x"),true,0.00001,undefined,2**-24);
    await s.write("}fn central(x:f64)->f64 {");
    await emitRustExp(s,()=>s.write("x"),true,undefined,{minimum:-(2**-25),maximum:2**-24});
    await s.write("}");await s.close();
    const source=await readFile(file,"utf8");assert.doesNotMatch(source,/\blet\b|\b(?:while|for|exp|sqrt)\b/);
    const {appendFile}=await import("node:fs/promises");
    await appendFile(file,`fn reference(x:f32)->f32{let scaled=x*1.4426950408889634_f32;let q=scaled.round_ties_even();let r=((q as f64)*(-0.693145751953125_f64)+(x as f64)) as f32;let r=((q as f64)*((-1.428606765330187e-6_f32) as f64)+(r as f64)) as f32;let mut p=0.000198527617612853646278381_f32;for c in [0.00139304355252534151077271_f32,0.00833336077630519866943359,0.0416664853692054748535156,0.166666671633720397949219,0.5]{p=((p as f64)*(r as f64)+(c as f64)) as f32;}let squared=r*r;let tail=((squared as f64)*(p as f64)+(r as f64)) as f32;let result=1.0+tail;let half=(q/2.0).floor();((result as f64)*2_f64.powi(half as i32)*2_f64.powi((q-half) as i32)) as f32}fn main(){for bits in 0xbdcccccd_u32..=0xbdccd20b {let x=f32::from_bits(bits);if (x as f64)< -0.10001{break;}assert_eq!(generated(x as f64).to_bits(),(reference(x) as f64).to_bits());}for k in -167..=0{let x=(k as f64)*2_f64.powi(-24);assert_eq!(lattice(x).to_bits(),(reference(x as f32) as f64).to_bits());}for x in [-2_f32.powi(-25),-2_f32.powi(-26),-0.0,0.0,2_f32.powi(-25),2_f32.powi(-24)]{assert_eq!(central(x as f64).to_bits(),(reference(x) as f64).to_bits());}}`);
    await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("inline half rounding preserves every finite adjacent midpoint and ties to even",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-half-midpoint-"));
  try{
    const file=join(dir,"half.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");
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
test("substitution emits composed rounding without local variables or numerical loops",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-scratch-"));
  try{
    const file=join(dir,"scratch.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");
    await s.round("f32",async()=>{await s.round("f32",()=>s.write("x/3.0"),false,{minimum:0.25,maximum:1});await s.write("+");await s.round("f32",()=>s.write("x*7.0"),false,{minimum:7,maximum:14});},false,{minimum:7,maximum:15});
    await s.write("}fn main(){for x in [1.0,1.5,2.0] {let expected=(((x/3.0) as f32)+((x*7.0) as f32)) as f64;assert_eq!(generated(x).to_bits(),expected.to_bits());}}");
    await s.close();assert.doesNotMatch((await readFile(file,"utf8")).split("fn main")[0]!,/\blet\b|\b(?:while|for)\b/);await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
test("positive interval facts remove impossible rounding branches without changing midpoint ties",async()=>{
  const dir=await mkdtemp(join(tmpdir(),"direct-round-range-"));
  try{
    const file=join(dir,"range.rs"),s=new DirectRustStream(file);
    await s.write("fn generated(x:f64)->f64 {");
    await s.round("f32",()=>s.write("x"),false,{minimum:2**-12,maximum:2});
    await s.write("}fn main(){let mut state=97_u32;for _ in 0..20000 {state=state.wrapping_mul(1664525).wrapping_add(1013904223);let bits=0x39800000+(state%(0x40000000-0x39800000-1));let lower=f32::from_bits(bits) as f64;let upper=f32::from_bits(bits+1) as f64;let midpoint=(lower+upper)/2.0;for x in [lower,midpoint,f64::from_bits(midpoint.to_bits()-1),f64::from_bits(midpoint.to_bits()+1)] {assert_eq!(generated(x).to_bits(),((x as f32) as f64).to_bits());}}}");
    await s.close();const source=(await readFile(file,"utf8")).split("fn main")[0]!;
    assert.doesNotMatch(source,/round_negative\s*=|round_normal=|round_exponent=|INFINITY/);
    await run("rustc",["-Awarnings",file,"-o",join(dir,"run")]);await run(join(dir,"run"));
  }finally{await rm(dir,{recursive:true,force:true});}
});
