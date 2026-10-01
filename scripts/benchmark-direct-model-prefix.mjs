/** Fixed-work diagnostic: cancels a separate compilation, never compiles a draft. */
import { pathToFileURL } from 'node:url';
import { resolve,join } from 'node:path';
import { readFile,writeFile } from 'node:fs/promises';
const [buildRoot,checkpoint,python,rawDimension,output,rawBytes='450000']=process.argv.slice(2);
if(!output)throw new Error('Usage: benchmark-direct-model-prefix BUILD CHECKPOINT PYTHON DIMENSION OUTPUT [BYTE_TARGET]');
const target=Number(rawBytes);
if(!Number.isSafeInteger(target)||target<1)throw new Error('Invalid byte target');
const {DirectRustStream}=await import(pathToFileURL(join(resolve(buildRoot),'src/direct-rust-stream.js')));
const original=DirectRustStream.prototype.write;
DirectRustStream.prototype.write=function(source){
 if(this.bytes>=target)this.cancel('Fixed-work benchmark complete; draft is not an executable artifact');
 return original.call(this,source);
};
const {writeDirectFlatRustModel}=await import(pathToFileURL(join(resolve(buildRoot),'src/direct-flat-rust-model.js')));
try{await writeDirectFlatRustModel(checkpoint,python,Number(rawDimension),output);throw new Error('Workload unexpectedly completed before byte target');}
catch(error){if(!String(error).includes('Fixed-work benchmark complete'))throw error;}
const progress=JSON.parse(await readFile(output+'.progress.json','utf8'));
if(progress.finalParity||progress.rustCompilationAdmitted)throw new Error('Diagnostic draft incorrectly admitted');
await writeFile(output+'.benchmark.json',JSON.stringify({scope:'fixed-work model compilation prefix, not final parity',target,...progress},null,2)+'\n');
console.log(JSON.stringify(progress));
