import test from 'node:test';
import assert from 'node:assert/strict';
import {captureDirectModelSnapshot,validateDirectModelSnapshot,type DirectModelSnapshot} from '../src/direct-model-snapshot.js';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import type {DirectModelDiscovery} from '../src/direct-flat-rust-model.js';

test('Remote target discovery rejects wrong numerical policies before reading or trusting state',async()=>{
  await assert.rejects(validateDirectModelSnapshot('/missing',{schema:'wrong'} as unknown as DirectModelSnapshot),/Incompatible/);
  await assert.rejects(validateDirectModelSnapshot('/missing',{
    schema:'direct-target-discovery-v1',target:'cuda',discovery:{output:{torch:'2.12.1'}}
  } as unknown as DirectModelSnapshot),/Incompatible/);
});

test('Compile-only target snapshots reject changed checkpoint, compiler and discovery identities',
  {skip:process.arch!=='arm64'},async()=>{
    const root=await mkdtemp(join(tmpdir(),'direct-target-snapshot-'));
    try{
      await writeFile(join(root,'config.json'),'{}');await writeFile(join(root,'model.safetensors'),'fixture');
      const discovery={output:{torch:'2.12.1'}} as unknown as DirectModelDiscovery;
      const snapshot=await captureDirectModelSnapshot(root,discovery);
      assert.equal(await validateDirectModelSnapshot(root,snapshot),discovery);
      await assert.rejects(validateDirectModelSnapshot(root,{...snapshot,discoverySha256:'0'.repeat(64)}),/changed target/);
      await assert.rejects(validateDirectModelSnapshot(root,{...snapshot,compiler:{}}),/compiler identity/);
      await writeFile(join(root,'model.safetensors'),'changed fixture');
      await assert.rejects(validateDirectModelSnapshot(root,snapshot),/checkpoint identity/);
    }finally{await rm(root,{recursive:true,force:true});}
  });
