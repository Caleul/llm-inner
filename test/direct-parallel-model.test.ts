import assert from 'node:assert/strict';
import test from 'node:test';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { writeDirectFlatRustModel } from '../src/direct-flat-rust-model.js';
import { writeParallelDirectFlatRustModel } from '../src/direct-parallel-rust-model.js';
const run=promisify(execFile);

test('real compiler preserves serial bytes, partitioned worker bytes and complete PyTorch logits',async(context)=>{
  const python=process.env.LLM_INNER_DIRECT_PYTHON;
  if(!python||process.arch!=='arm64'){context.skip('Set LLM_INNER_DIRECT_PYTHON to PyTorch 2.12.1 CPU arm64');return;}
  const dir=await mkdtemp(join(tmpdir(),'direct-parallel-model-'));
  try{
    const checkpoint=join(dir,'checkpoint');
    // A complete zero-output checkpoint exercises discovery, geometry proofs,
    // every position/length variant and CLI parity. It deliberately does not
    // claim nonzero attention/MLP final-logit parity.
    await run(python,['-c',`import torch\nfrom transformers import LlamaConfig,LlamaForCausalLM\nc=LlamaConfig(hidden_size=2,intermediate_size=1,num_hidden_layers=1,num_attention_heads=1,num_key_value_heads=1,head_dim=2,vocab_size=4,max_position_embeddings=4,rms_norm_eps=1e-6)\nwith torch.no_grad():\n m=LlamaForCausalLM(c).half()\n for name,p in m.named_parameters():\n  p.fill_(1 if 'norm' in name else 0)\nm.save_pretrained(${JSON.stringify(checkpoint)})`],{maxBuffer:1024*1024});
    const serial=join(dir,'serial.rs');await writeDirectFlatRustModel(checkpoint,python,2,serial,{weightCacheBytes:0});
    const natural=join(dir,'natural.rs');
    await writeParallelDirectFlatRustModel(checkpoint,python,2,natural,{workers:4,partitions:1,maxOutputMiB:1});
    assert.deepEqual(await readFile(serial),await readFile(natural));
    const one=join(dir,'one.rs'),four=join(dir,'four.rs');
    await writeParallelDirectFlatRustModel(checkpoint,python,2,one,{workers:1,partitions:4,maxOutputMiB:1});
    await writeParallelDirectFlatRustModel(checkpoint,python,2,four,{workers:4,partitions:4,maxOutputMiB:1});
    assert.deepEqual(await readFile(one),await readFile(four));
    for(const path of [serial,one,four]){
      const metadata=JSON.parse(await readFile(path+'.reduction.json','utf8'));
      assert.equal(metadata.status,'emitted');assert.equal(metadata.finalParity,false);
      const main=`\nfn main(){let rows:Vec<[f64;2]>=std::env::args().skip(1).map(|r|r.split(',').map(|x|x.parse().unwrap()).collect::<Vec<f64>>().try_into().unwrap()).collect();for t in 0..rows.len(){println!("{}",compiled_dimension(&rows,t).to_bits());}}`;
      const executable=path+'.executable';await writeFile(path,(await readFile(path,'utf8'))+main);
      await run('rustc',['--edition=2021','-Awarnings',path,'-o',executable]);
      const helper=new URL('../../helpers/validate_direct_flat_rust.py',import.meta.url).pathname;
      await run(python,[helper,checkpoint,executable,'2','--report',path+'.parity.json'],{maxBuffer:1024*1024});
      const parity=JSON.parse(await readFile(path+'.parity.json','utf8'));
      assert.equal(parity.exactFinalParity,true);assert.ok(parity.comparisons.length>20);
    }
    const target=join(dir,'preserved.rs');await writeFile(target,'existing complete source');
    const signal=new AbortController();signal.abort();
    await assert.rejects(writeParallelDirectFlatRustModel(checkpoint,python,2,target,
      {workers:2,maxOutputMiB:1,signal:signal.signal}),/aborted/);
    assert.equal(await readFile(target,'utf8'),'existing complete source');
    assert.equal((await readdir(dir)).some(name=>name.startsWith('.direct-parallel-')),false);
  }finally{await rm(dir,{recursive:true,force:true});}
});
