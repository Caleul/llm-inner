import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { DirectRustStream } from "../src/direct-rust-stream.js";
import { DirectBranchDomain, rational } from "../src/direct-branch-domain.js";
const run = promisify(execFile);
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
