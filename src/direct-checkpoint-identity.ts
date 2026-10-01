import { readFile, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

/** Content identity for config/index and stat identity for payloads. Detects
 * ordinary concurrent replacement/modification without reading the checkpoint
 * twice. This is an immutability guard, not a payload checksum or savepoint.
 */
export async function directCheckpointIdentity(directory:string):Promise<string>{
  const files=(await readdir(directory)).filter(name=>name==='config.json'||
    name.endsWith('.safetensors')||name.endsWith('.safetensors.index.json')).sort();
  if(!files.includes('config.json')||!files.some(name=>name.endsWith('.safetensors')))
    throw new Error('Missing checkpoint config or Safetensors payload');
  const entries=[];
  for(const name of files){
    const info=await stat(join(directory,name),{bigint:true});
    entries.push({name,size:String(info.size),dev:String(info.dev),ino:String(info.ino),
      mtimeNs:String(info.mtimeNs),ctimeNs:String(info.ctimeNs),
      ...(name.endsWith('.json')?{sha256:createHash('sha256').update(await readFile(join(directory,name))).digest('hex')}:{})});
  }
  return createHash('sha256').update(JSON.stringify(entries)).digest('hex');
}
export async function verifyDirectCheckpointIdentity(directory:string,expected:string):Promise<void>{
  if(await directCheckpointIdentity(directory)!==expected)throw new Error('Checkpoint changed during compilation');
}
