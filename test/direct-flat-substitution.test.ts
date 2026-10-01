import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, appendFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { createHash } from "node:crypto";
import { DirectRustStream } from "../src/direct-rust-stream.js";
import { DirectFlatSubstitution, FlatConditions, type FlatProducer } from "../src/direct-flat-substitution.js";
import { rational } from "../src/direct-branch-domain.js";
import { exactNumberRational as exact } from "../src/direct-round-preimage.js";
import { substituteCpuArm64F32Sum } from "../src/direct-rust-mean.js";
import { emitRustSilu } from "../src/direct-rust-numeric.js";
import { substituteFlatProjection } from "../src/direct-flat-projection.js";
import { substituteCpuArm64SoftmaxSum } from "../src/direct-rust-softmax.js";
const run=promisify(execFile);

test("Rust softmax scalar substitution preserves the length-dependent SIMD reduction order",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-softmax-"));
  try{
    const path=join(directory,"softmax.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    let main="fn main(){";
    for(const [index,[active,full]] of ([[3,false],[3,true],[5,true],[8,true],[9,true]] as const).entries()){
      await s.write(`fn sum${index}(x:&[f64])->f64 {'answer:{`);s.beginReducedExpression();
      await substituteCpuArm64SoftmaxSum(f,new FlatConditions(),active,full,i=>(p,k)=>
        f.input(p,()=>s.write(`x[${i}]`),0.91,0.99,(p,v)=>k(p,{...v,precision:"f32",quantum:2**-24})),
        (p,v)=>f.leaf(p,"answer",v));
      await f.finishRound();await s.write("}}");
      main+=`for seed in 0..100_u32{let x:Vec<f64>=(0..${active}).map(|i|((0.911+((seed*13+i*7)%78) as f64*0.001) as f32) as f64).collect();let mut a=[0_f32;4];for (i,v) in x.iter().enumerate(){a[i%4]+=*v as f32;}let expected=${full?"(a[0]+a[2])+(a[1]+a[3])":"(a[0]+a[1])+a[2]"};assert_eq!(sum${index}(&x).to_bits(),(expected as f64).to_bits(),"case ${index}: {}",seed);}`;
    }
    await s.close();await appendFile(path,main+"}");
    await run("rustc",["--edition=2021","-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("product endpoint preimages retain every valid sign and threshold case",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-product-bounds-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    const y:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][1]"),k);
    const product:FlatProducer=(p,k)=>f.binary(p,x,y,"*",k);
    for(const sign of [-1,1])for(const op of [">=","<="] as const)for(const threshold of [-1,0,1]){
      const low=sign>0?1:-2,high=sign>0?2:-1;
      await f.comparison(new FlatConditions(),y,">=",rational(BigInt(low)),p=>
        f.comparison(p,y,"<=",rational(BigInt(high)),p=>
          f.comparison(p,product,op,rational(BigInt(threshold)),p=>x(p,async(_p,value)=>{
            for(const a of [-4,-2,-1,-0.5,0,0.5,1,2,4])for(const b of [low,(low+high)/2,high]){
              if(op===">="?a*b>=threshold:a*b<=threshold)
                assert.ok(a>=value.minimum&&a<=value.maximum,`${a}*${b} ${op} ${threshold}`);
            }
          }),async()=>{}),async()=>{}),async()=>{});
    }
    await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("inherited sum comparisons survive zero substitution without deleting their guards",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-inherited-sum-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const hash=(source:string)=>createHash("sha256").update(source).digest("hex");
    const source="(input_tokens[0][0]*input_tokens[0][0])+(input_tokens[0][1]*input_tokens[0][1])";
    let path=new FlatConditions().refine(hash(source),()=>s.write(source),{lower:{value:rational(4n),inclusive:true}})!;
    path=path.refine(hash("input_tokens[0][0]"),()=>s.write("input_tokens[0][0]"),
      {lower:{value:rational(0n),inclusive:true},upper:{value:rational(0n),inclusive:true}},true)!;
    const y:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][1]"),k);
    const square:FlatProducer=(p,k)=>f.square(p,y,k);
    let visits=0;
    await f.binary(path,square,(p,k)=>f.literal(p,0,k),"+",async(p,value)=>{
      visits++;assert.equal(value.minimum,4);assert.equal(p.guards.length,2);
    });
    assert.equal(visits,1);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("normalized comparison consequences propagate into scalar subexpressions to a fixed point",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-fixed-point-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const source="(input_tokens[0][0]*input_tokens[0][0])+(input_tokens[0][1]*input_tokens[0][1])";
    const path=new FlatConditions().refine(createHash("sha256").update(source).digest("hex"),
      ()=>s.write(source),{lower:{value:rational(4n),inclusive:true}})!;
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    const y:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][1]"),k);
    const square:FlatProducer=(p,k)=>f.square(p,y,k);
    let visits=0;
    await f.comparison(path,x,">=",rational(0n),p=>f.comparison(p,x,"<=",rational(0n),p=>
      f.binary(p,square,(p,k)=>f.literal(p,2,k),"/",async(p,value)=>{
        visits++;assert.equal(value.minimum,2);assert.equal(p.guards.length,2);
      }),async()=>{}),async()=>{});
    assert.equal(visits,1);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("compound comparisons reach a fixed point before their next consumer without a new input binding",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-compound-round-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const hash=(value:string)=>createHash("sha256").update(value).digest("hex");
    const sum="(input_tokens[0][0]*input_tokens[0][0])+(input_tokens[0][1]*input_tokens[0][1])";
    let path=new FlatConditions().refine(hash(sum),()=>s.write(sum),{lower:{value:rational(4n),inclusive:true}})!;
    path=path.refine(hash("input_tokens[0][0]"),()=>s.write("input_tokens[0][0]"),
      {lower:{value:rational(0n),inclusive:true},upper:{value:rational(0n),inclusive:true}},true)!;
    const square="(input_tokens[0][1]*input_tokens[0][1])";
    const squared:FlatProducer=(p,k)=>f.input(p,()=>s.write(square),0,65504**2,k);
    const nested:FlatProducer=(p,k)=>f.input(p,()=>s.write(`(${square}+1.0)`),1,65504**2+1,k);
    let visits=0;
    await f.comparison(path,squared,"<=",rational(9n),p=>
      f.binary(p,nested,(p,k)=>f.literal(p,2,k),"/",async(_p,value)=>{
        visits++;assert.equal(value.minimum,2.5);assert.equal(value.maximum,5);
      }),async()=>{});
    assert.equal(visits,1);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("square magnitude bounds propagate through the sign of a positive variable product",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-magnitude-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const field=(i:number):FlatProducer=>(p,k)=>f.f16Input(p,()=>s.write(`input_tokens[0][${i}]`),k);
    const x=field(0),y=field(1),square:FlatProducer=(p,k)=>f.square(p,x,k),product:FlatProducer=(p,k)=>f.binary(p,x,y,"*",k);
    let positive=0,negative=0;
    await f.comparison(new FlatConditions(),y,">=",rational(1n),p=>f.comparison(p,y,"<=",rational(2n),
      p=>f.comparison(p,square,">=",rational(4n),p=>f.comparison(p,product,">=",rational(0n),
        p=>x(p,async(_p,v)=>{positive++;assert.equal(v.minimum,2);}),
        p=>x(p,async(_p,v)=>{negative++;assert.equal(v.maximum,-2);})),async()=>{}),async()=>{}),async()=>{});
    assert.equal(positive,1);assert.equal(negative,1);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("rounded-affine SiLU substitution preserves all small F16 policy inputs and signed zeros",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-small-silu-"));
  try{
    const path=join(directory,"silu.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    await f.silu(new FlatConditions(),(p,k)=>f.input(p,()=>s.write("x"),-(2**-13),2**-13,
      (p,v)=>k(p,{...v,precision:"f16"})),(p,v)=>f.leaf(p,"answer",v));
    await f.finishRound();await s.write('panic!("outside SiLU domain")}}');await s.close();
    await appendFile(path,'fn decode(b:u16)->f64{let s=if b&32768==0{1.0}else{-1.0};let e=(b>>10)&31;let m=b&1023;s*if e==0{(m as f64)*2_f64.powi(-24)}else{(1.0+(m as f64)/1024.0)*2_f64.powi(e as i32-15)}}fn main(){let data=std::fs::read(std::env::args().nth(1).unwrap()).unwrap();for b in 0..=2048_u16{for sign in [0,32768]{let bits=b|sign;let i=bits as usize*2;let expected=decode(u16::from_le_bytes(data[i..i+2].try_into().unwrap()));assert_eq!(generated(decode(bits)).to_bits(),expected.to_bits(),"{}",bits);}}}');
    await run("rustc",["--edition=2021","-Awarnings",path,"-o",join(directory,"run")]);
    await run(join(directory,"run"),[new URL("../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin",import.meta.url).pathname]);
    assert.ok(s.bytes<20000,`Small SiLU did not reduce: ${s.bytes} bytes`);
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("new fundamental constraints discard inherited contradictions before the next consumer",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-early-prune-"));
  try{
    const s=new DirectRustStream(join(directory,"unused.rs")),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("input_tokens[0][0]"),k);
    const squared:FlatProducer=(p,k)=>f.binary(p,x,x,"*",k);
    const difference:FlatProducer=(p,k)=>f.binary(p,squared,(p,k)=>f.literal(p,1,k),"+",k);
    let consumers=0;
    await f.comparison(new FlatConditions(),difference,"<",rational(0n),p=>
      f.comparison(p,x,"<=",rational(1n),async()=>{consumers++;},async()=>{consumers++;}),async()=>{});
    assert.equal(consumers,0);assert.ok(s.eliminatedBranches>=2);await s.close();
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("equal-result leaves merge across a gap containing no valid fundamental F16 input",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-discrete-union-"));
  try{
    const path=join(directory,"union.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s),base=new FlatConditions();
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write("x"),k);
    await s.write("fn generated(x:f64)->f64 {'answer:{");s.beginReducedExpression();
    for(const point of [1,1+2**-10]){
      await f.comparison(base,x,">=",exact(point),p=>f.comparison(p,x,"<=",exact(point),
        p=>f.literal(p,7,(p,v)=>f.leaf(p,"answer",v)),async()=>{}),async()=>{});
    }
    await f.finishRound();await s.write("break 'answer 0.0;}}");await s.close();
    const source=await readFile(path,"utf8");assert.equal(source.match(/break 'answer 7/g)?.length,1);
    await appendFile(path,'fn main(){assert_eq!(generated(1.0),7.0);assert_eq!(generated(1.0+2_f64.powi(-10)),7.0);assert_eq!(generated(1.0-2_f64.powi(-11)),0.0);assert_eq!(generated(1.0+2_f64.powi(-9)),0.0);}');
    await run("rustc",["-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
  }finally{await rm(directory,{recursive:true,force:true});}
});

test("direct root chords correct every F32 value across binades, small variances and subnormal inputs",async()=>{
  const directory=await mkdtemp(join(tmpdir(),"direct-flat-root-chords-"));
  try{
    const path=join(directory,"root.rs"),s=new DirectRustStream(path),f=new DirectFlatSubstitution(s);
    const max=Math.fround(3.4028234663852886e38),ranges=[[1,1.001],[3.999,4.001],
      [1e-6,1.001e-6],[2**-149,16*2**-149],[max-2**105,max]];
    const data=new DataView(new ArrayBuffer(4));
    const bits=(x:number)=>{data.setFloat32(0,x,true);return data.getUint32(0,true);};
    let main="fn main(){";
    for(const [i,range] of ranges.entries()){
      const minimum=Math.fround(range[0]!),maximum=Math.fround(range[1]!);
      await s.write(`fn root${i}(x:f64)->f64 {'answer:{`);s.beginReducedExpression();
      await f.sqrt(new FlatConditions(),(p,k)=>f.input(p,()=>s.write("x"),minimum,maximum,
        (p,v)=>k(p,{...v,precision:"f32"})),(p,v)=>f.leaf(p,"answer",v));
      await f.finishRound();await s.write('panic!("outside root domain")}}');
      main+=`for bits in ${bits(minimum)}..=${bits(maximum)}{let x=f32::from_bits(bits);assert_eq!(root${i}(x as f64).to_bits(),(x.sqrt() as f64).to_bits(),"window ${i}: {}",bits);}`;
    }
    await s.close();await appendFile(path,main+"}");
    await run("rustc",["-Awarnings",path,"-o",join(directory,"run")]);await run(join(directory,"run"));
  }finally{await rm(directory,{recursive:true,force:true});}
});

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
