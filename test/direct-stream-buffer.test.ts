import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DirectRustStream } from '../src/direct-rust-stream.js';
test('bounded output batching preserves bytes, inspection isolation and admission',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'direct-stream-buffer-'));
 try{
  const sources=[];
  for(const limit of [0,32,65536]){
   const file=join(dir,`${limit}.rs`),s=new DirectRustStream(file,limit);
   await s.write('fn generated(x:f64)->f64 {');s.beginReducedExpression();
   let inspected='';await s.inspectExpression(async()=>{await s.write('(x*2.0)');},chunk=>{inspected+=chunk;});
   assert.equal(inspected,'(x*2.0)');
   for(let i=0;i<100;i++){await s.write('if x>');await s.write(`${i}.0`);await s.write(' {return x+1.0;}');}
   await s.write('/* '+ 'á'.repeat(40000)+' */');await s.write('x}');await s.close();
   const bytes=await readFile(file);assert.equal(s.bytes,bytes.length);sources.push(bytes);
   if(limit===65536)assert.ok(s.outputWrites<10);
  }
  assert.deepEqual(sources[0],sources[1]);assert.deepEqual(sources[0],sources[2]);
  const cancelled=new DirectRustStream(join(dir,'cancelled.rs'),32);cancelled.cancel('stop');
  await assert.rejects(cancelled.write('x'),/stop/);cancelled.destroy();
 }finally{await rm(dir,{recursive:true,force:true});}
});
