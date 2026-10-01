import { mkdir, open, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

/** Exclusive ownership of the publication path. A crashed owner leaves a lock
 * requiring explicit removal after inspecting it; never steal an uncertain run.
 */
export async function withDirectCompilationLease<T>(path:string,body:()=>Promise<T>):Promise<T>{
  const target=resolve(path),lock=target+'.compile.lock';
  await mkdir(dirname(target),{recursive:true});
  const handle=await open(lock,'wx').catch(error=>{
    if((error as NodeJS.ErrnoException).code==='EEXIST')throw new Error(`Compilation target already locked: ${lock}`);
    throw error;
  });
  try{
    await handle.writeFile(JSON.stringify({pid:process.pid,target,startedAt:new Date().toISOString()})+'\n');
    return await body();
  }finally{await handle.close();await rm(lock,{force:true});}
}
