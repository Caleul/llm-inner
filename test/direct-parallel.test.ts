import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, writeFile, appendFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { directCompilationUnits, directUnitConditions, finiteF16Count, validateDirectUnit } from '../src/direct-compilation-units.js';
import { directPoolBudget, runDirectCompilationPool, type DirectUnitJob } from '../src/direct-compilation-pool.js';
import { assembleDirectFragments } from '../src/direct-parallel-rust-model.js';
import { DirectRustStream } from '../src/direct-rust-stream.js';
import { DirectFlatSubstitution, FlatConditions, type FlatProducer } from '../src/direct-flat-substitution.js';
import { DirectWeightPages } from '../src/direct-weight-pages.js';
import { withDirectCompilationLease } from '../src/direct-compilation-lease.js';
import { DirectSourceDigest } from '../src/direct-source-digest.js';
import { directCheckpointIdentity, verifyDirectCheckpointIdentity } from '../src/direct-checkpoint-identity.js';
import { DirectSourceIdentities } from '../src/direct-source-identity.js';
import { f16BitsToDyadic } from '../src/fixed-f16-projection.js';
import type { TensorInfo } from '../src/types.js';
const run=promisify(execFile),fixtureURL=new URL('./direct-pool-fixture.js',import.meta.url);

test('F16 partitions cover every finite input exactly once and keep signed zeros together',()=>{
  for(const count of [1,2,3,4,7,16,finiteF16Count]){
    const units=[...directCompilationUnits(1,3,count,2)];
    if(count===1){assert.equal(units[0]!.partition,undefined);continue;}
    const membership=new Uint8Array(finiteF16Count);let zeroOwners=0;
    for(const unit of units){const p=unit.partition!;
      assert.equal(p.coordinate,2);for(let i=p.first;i<=p.last;i++)membership[i]=membership[i]!+1;
      if(p.first<=31743&&p.last>=31743)zeroOwners++;
    }
    assert.ok(membership.every(n=>n===1));assert.equal(zeroOwners,1);
  }
});
test('compilation units preserve the n=3 versus n>=4 softmax policy boundary',()=>{
  const units=[...directCompilationUnits(4,2,1)];
  assert.deepEqual(units.map(u=>[u.position,u.fullVectorSoftmax]),[[0,false],[1,false],[2,false],[2,true],[3,true]]);
  assert.throws(()=>validateDirectUnit({position:1,fullVectorSoftmax:true},4,2),/Invalid/);
  for(const count of [0,finiteF16Count+1])assert.throws(()=>[...directCompilationUnits(4,2,count)],/Invalid/);
  assert.throws(()=>[...directCompilationUnits(4,2,2,2)],/Invalid/);
});
test('pure identity cache preserves SHA256 exactly and never retains unbounded keys',()=>{
  const cache=new DirectSourceIdentities(2);
  for(const values of [['+','a','b'],['number','-0'],['number','0'],['input_tokens[12][34]']]){
    const expected=createHash('sha256').update(values.join('|')).digest('hex');
    assert.equal(cache.key(...values),expected);assert.equal(cache.key(...values),expected);
    assert.ok(cache.size<=2);
  }
  assert.ok(cache.hits>0);cache.key('x'.repeat(600));assert.equal(cache.size,2);
  const disabled=new DirectSourceIdentities(0);disabled.key('a');disabled.key('a');assert.equal(disabled.hits,0);
});
test('paged weights decode every finite F16 bit pattern exactly including negative zero',async()=>{
  const bytes=Buffer.alloc(65536*2);for(let bits=0;bits<65536;bits++)bytes.writeUInt16LE(bits,bits*2);
  const tensor:TensorInfo={name:'weight',shard:'one',byteOffset:0,byteLength:bytes.length,
    storageDtype:'F16',storageShape:[65536],logicalShape:[65536]};
  const reader={readTensorBytesRange:async(_t:TensorInfo,offset:number,length:number)=>Buffer.from(bytes.subarray(offset,offset+length))};
  for(const budget of [0,4096]){
    const pages=new DirectWeightPages(reader,budget,2048);
    for(let bits=0;bits<65536;bits++){
      if((bits&0x7c00)===0x7c00)continue;
      const d=f16BitsToDyadic(bits),expected=bits===32768?-0:Number(d.coefficient)*2**d.exponent;
      assert.ok(Object.is(await pages.read(tensor,bits),expected),`bits=${bits} budget=${budget}`);
      assert.ok(pages.retainedBytes<=budget);
    }
    if(budget){assert.ok(pages.hits>60000);assert.ok(pages.reads<100);}
    await assert.rejects(pages.read(tensor,0x7c00),/Nonfinite/);
    await assert.rejects(pages.read(tensor,65536),/Invalid/);
    pages.clear();assert.equal(pages.retainedBytes,0);
  }
});
test('weight pages evict bounded bytes without confusing shards or tensor offsets',async()=>{
  const reader={readTensorBytesRange:async(t:TensorInfo,_offset:number,length:number)=>{
    const b=Buffer.alloc(length);b.writeUInt16LE(t.shard==='a'?0x3c00:0xc000);return b;}};
  const pages=new DirectWeightPages(reader,4,4);
  const a:TensorInfo={name:'a',shard:'a',byteOffset:0,byteLength:4,storageDtype:'F16',storageShape:[2],logicalShape:[2]};
  assert.equal(await pages.read(a,0),1);assert.equal(await pages.read({...a,shard:'b'},0),-2);
  assert.equal(await pages.read(a,0),1);assert.equal(pages.reads,3);assert.equal(pages.retainedBytes,4);
});
test('worker budget bounds admission instead of treating more workers as more available RAM',()=>{
  assert.equal(directPoolBudget({workers:14,memoryMiB:1024}).workers,2);
  assert.throws(()=>directPoolBudget({memoryMiB:128}),/cannot admit/);
  assert.throws(()=>directPoolBudget({workers:0}),/Invalid/);
});
test('dynamic pool assembles canonical bytes despite out-of-order worker completion',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-pool-order-'));
  try{
    const sources=[];
    for(const workers of [1,4]){
      const jobs=Array.from({length:9},(_,id):DirectUnitJob=>({id,unit:{position:id,fullVectorSoftmax:false},path:join(dir,`${workers}-${id}`)}));
      const results=await runDirectCompilationPool(fixtureURL,{},jobs,directPoolBudget({workers}));
      const path=join(dir,`assembled-${workers}`);await assembleDirectFragments(path,'header\n','footer\n',results,1000);
      sources.push(await readFile(path));
    }
    assert.deepEqual(sources[0],sources[1]);
    assert.equal(sources[0]!.toString(),'header\n'+Array.from({length:9},(_,i)=>`unit ${i}\n`).join('')+'footer\n');
  }finally{await rm(dir,{recursive:true,force:true});}
});
test('pool stops on worker failure, unexpected clean exit, cancellation and duplicate IDs',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-pool-failure-'));
  const jobs=Array.from({length:3},(_,id):DirectUnitJob=>({id,unit:{position:0,fullVectorSoftmax:false},path:join(dir,`${id}`)}));
  try{
    const budget=directPoolBudget({workers:2});
    await assert.rejects(runDirectCompilationPool(fixtureURL,{fail:0},jobs,budget),/fixture worker failure/);
    await assert.rejects(runDirectCompilationPool(fixtureURL,{exit:0},jobs,budget),/worker exited/);
    const controller=new AbortController();setTimeout(()=>controller.abort(),30);
    await assert.rejects(runDirectCompilationPool(fixtureURL,{delay:1000},jobs,budget,controller.signal),/aborted/);
    await assert.rejects(runDirectCompilationPool(fixtureURL,{},[jobs[0]!,jobs[0]!],budget),/Duplicate/);
    const already=new AbortController();already.abort();
    await assert.rejects(runDirectCompilationPool(fixtureURL,{},jobs,budget,already.signal),/aborted/);
  }finally{await rm(dir,{recursive:true,force:true});}
});
test('fragment assembly rejects corruption, omissions and output over budget',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-fragment-integrity-'));
  try{
    const jobs=[{id:0,unit:{position:0,fullVectorSoftmax:false},path:join(dir,'unit')}];
    const results=await runDirectCompilationPool(fixtureURL,{},jobs,directPoolBudget({workers:1}));
    await assert.rejects(assembleDirectFragments(join(dir,'limit'),'head','tail',results,3),/byte limit/);
    await assert.rejects(assembleDirectFragments(join(dir,'missing'),'','',[{...results[0]!,id:1}],1000),/Missing/);
    await writeFile(jobs[0]!.path,'tampered');
    await assert.rejects(assembleDirectFragments(join(dir,'corrupt'),'','',results,1000),/integrity/);
  }finally{await rm(dir,{recursive:true,force:true});}
});
test('shared output budget counts only emitted bytes across independent streams',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-shared-byte-limit-'));
  const a=new DirectRustStream(join(dir,'a')),b=new DirectRustStream(join(dir,'b'));
  try{
    const counter=new SharedArrayBuffer(8);a.setSharedByteLimit(counter,10);b.setSharedByteLimit(counter,10);
    await a.inspectExpression(()=>a.write('x'.repeat(100)),()=>{});assert.equal(Atomics.load(new BigInt64Array(counter),0),0n);
    await a.write('12345');await b.write('12345');await assert.rejects(b.write('x'),/byte limit/);
    await a.close();await b.close();
  }finally{a.destroy();b.destroy();await rm(dir,{recursive:true,force:true});}
});
test('partitioned Rust preserves disjoint branch results bitwise for every finite F16 input',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-partitions-rust-'));
  try{
    const file=join(dir,'partition.rs'),s=new DirectRustStream(file),f=new DirectFlatSubstitution(s);
    const x:FlatProducer=(p,k)=>f.f16Input(p,()=>s.write('input_tokens[0][0]'),k);
    for(const partitions of [1,3,4,7]){
      await s.write(`fn generated_${partitions}(input_tokens:&[[f64;1]],t:usize)->f64 {'answer:{`);
      s.beginReducedExpression();
      for(const unit of directCompilationUnits(1,1,partitions)){
        const path=directUnitConditions(unit,1,1,s);
        const square:FlatProducer=(p,k)=>f.square(p,x,k);
        const outside=(p:FlatConditions)=>f.binary(p,x,(p,k)=>f.literal(p,0,k),'*',(p,v)=>f.leaf(p,'answer',v));
        const {exactNumberRational}=await import('../src/direct-round-preimage.js');
        await f.comparison(path,square,'>=',exactNumberRational(4),
          p=>f.comparison(p,square,'<=',exactNumberRational(9),p=>x(p,(p,v)=>f.leaf(p,'answer',v)),outside),outside);
        await f.finishRound();
      }
      await s.write('panic!("outside domain")}}');
    }
    await s.close();
    const checks=[1,3,4,7].map(n=>`assert_eq!(generated_${n}(&rows,0).to_bits(),expected.to_bits(),"partition ${n} bits {}",bits);`).join('');
    await appendFile(file,`fn decode(b:u32)->f64{let sign=if b&32768==0{1.0}else{-1.0};let e=(b>>10)&31;let m=b&1023;sign*if e==0{m as f64*2_f64.powi(-24)}else{(1.0+m as f64/1024.0)*2_f64.powi(e as i32-15)}}fn main(){for bits in 0..65536{if bits&31744==31744{continue;}let x=decode(bits);let expected=if x*x>=4.0&&x*x<=9.0{x}else{x*0.0};for length in [1,2,4]{let rows=vec![[x];length];${checks}}}}`);
    await run('rustc',['--edition=2021','-Awarnings',file,'-o',join(dir,'run')]);await run(join(dir,'run'));
  }finally{await rm(dir,{recursive:true,force:true});}
});

test('batched source digest preserves per-fragment bytes including UTF16 surrogate boundaries',()=>{
  for(const chunks of [['x','+','1.0'],Array.from({length:3000},(_,i)=>String(i)),['x'.repeat(20000),'á','\ud83d','\ude00'],[]]){
    const reference=createHash('sha256'),digest=new DirectSourceDigest();
    for(const chunk of chunks){reference.update(chunk);digest.update(chunk);}
    assert.equal(digest.digest('hex'),reference.digest('hex'));
  }
});
test('checkpoint identity rejects changed config, replaced payload and new shards',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-checkpoint-identity-'));
  try{
    await writeFile(join(dir,'config.json'),'{}');await writeFile(join(dir,'model.safetensors'),'payload');
    let identity=await directCheckpointIdentity(dir);await verifyDirectCheckpointIdentity(dir,identity);
    await writeFile(join(dir,'config.json'),'{} ');await assert.rejects(verifyDirectCheckpointIdentity(dir,identity),/changed/);
    identity=await directCheckpointIdentity(dir);await writeFile(join(dir,'model.safetensors'),'replacement');
    await assert.rejects(verifyDirectCheckpointIdentity(dir,identity),/changed/);
    identity=await directCheckpointIdentity(dir);await writeFile(join(dir,'extra.safetensors'),'extra');
    await assert.rejects(verifyDirectCheckpointIdentity(dir,identity),/changed/);
  }finally{await rm(dir,{recursive:true,force:true});}
});

test('publication lease rejects concurrent compilation and releases ownership after failure',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-compilation-lease-')),path=join(dir,'result.rs');
  try{
    await withDirectCompilationLease(path,async()=>{
      await assert.rejects(withDirectCompilationLease(path,async()=>{}),/already locked/);
    });
    await assert.rejects(withDirectCompilationLease(path,async()=>{throw new Error('fixture compilation failure');}),/fixture/);
    await withDirectCompilationLease(path,async()=>{});
  }finally{await rm(dir,{recursive:true,force:true});}
});
test('stream file errors reject completion without an unhandled asynchronous error',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-stream-io-error-')),stream=new DirectRustStream(join(dir,'missing','output'));
  try{await stream.write('x');await assert.rejects(stream.close(),/ENOENT/);}
  finally{stream.destroy();await rm(dir,{recursive:true,force:true});}
});
test('pool fails when measured RSS exceeds the requested budget',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-pool-rss-'));
  try{
    const budget=directPoolBudget({workers:1});budget.memoryMiB=1;
    await assert.rejects(runDirectCompilationPool(fixtureURL,{delay:2000},
      [{id:0,unit:{position:0,fullVectorSoftmax:false},path:join(dir,'unit')}],budget),/memory budget/);
  }finally{await rm(dir,{recursive:true,force:true});}
});
