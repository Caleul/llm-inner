import {createHash} from 'node:crypto';
import {readdir,readFile} from 'node:fs/promises';
import {join} from 'node:path';
import type {DirectModelDiscovery} from './direct-flat-rust-model.js';

/** Compile-only target discovery, never model weights or runtime activations.
 * A remote compiler keeps the source numerical target; it does not pretend
 * that an x64 CUDA host is the original CPU arm64 forward implementation. */
export interface DirectModelSnapshot {
  schema:'direct-target-discovery-v1';target:'pytorch-2.12.1-cpu-arm64';
  checkpoint:Record<string,string>;compiler:Record<string,string>;
  discovery:DirectModelDiscovery;discoverySha256:string;
}
const digest=(bytes:Uint8Array|string)=>createHash('sha256').update(bytes).digest('hex');
async function checkpointIdentity(directory:string):Promise<Record<string,string>>{
  const files=(await readdir(directory)).filter(name=>name==='config.json'||name.endsWith('.safetensors')||name.endsWith('.safetensors.index.json')).sort();
  if(!files.includes('config.json')||!files.some(name=>name.endsWith('.safetensors')))throw new Error('Snapshot checkpoint files missing');
  return Object.fromEntries(await Promise.all(files.map(async name=>[name,digest(await readFile(join(directory,name)))])));
}
async function compilerIdentity():Promise<Record<string,string>>{
  const root=new URL('../../',import.meta.url),files:string[]=[];
  for(const [directory,suffixes] of [['dist/src',['.js']],['helpers',['.py','.mjs']],['numeric-profiles',['.bin']]] as const)
    for(const name of (await readdir(new URL(directory+'/',root))).sort())if(suffixes.some(suffix=>name.endsWith(suffix)))files.push(directory+'/'+name);
  return Object.fromEntries(await Promise.all(files.map(async name=>[name,digest(await readFile(new URL(name,root)))])));
}
export async function captureDirectModelSnapshot(directory:string,discovery:DirectModelDiscovery):Promise<DirectModelSnapshot>{
  if(process.arch!=='arm64'||discovery.output.torch.split('+')[0]!=='2.12.1')throw new Error('Target discovery must be captured on its original CPU arm64 host');
  return {schema:'direct-target-discovery-v1',target:'pytorch-2.12.1-cpu-arm64',
    checkpoint:await checkpointIdentity(directory),compiler:await compilerIdentity(),
    discovery,discoverySha256:digest(JSON.stringify(discovery))};
}
export async function validateDirectModelSnapshot(directory:string,snapshot:DirectModelSnapshot):Promise<DirectModelDiscovery>{
  if(snapshot.schema!=='direct-target-discovery-v1'||snapshot.target!=='pytorch-2.12.1-cpu-arm64'||
    snapshot.discovery?.output.torch.split('+')[0]!=='2.12.1'||digest(JSON.stringify(snapshot.discovery))!==snapshot.discoverySha256)
    throw new Error('Incompatible or changed target discovery snapshot');
  for(const [label,expected,actual] of [['checkpoint',snapshot.checkpoint,await checkpointIdentity(directory)],
    ['compiler',snapshot.compiler,await compilerIdentity()]] as const)
    if(JSON.stringify(expected)!==JSON.stringify(actual))throw new Error(`Incompatible ${label} identity; saved discovery cannot be reused`);
  return snapshot.discovery;
}
