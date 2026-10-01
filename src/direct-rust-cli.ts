import { writeDirectFlatRustModel } from "./direct-flat-rust-model.js";
import { appendFile,readFile,writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { writeParallelDirectFlatRustModel } from "./direct-parallel-rust-model.js";
import type { DirectPoolOptions } from "./direct-compilation-pool.js";
const [directory,python,rawDimension,path,...flags]=process.argv.slice(2);
if(!directory||!python||!rawDimension||!path)
  throw new Error("Usage: direct-rust-cli CHECKPOINT PYTHON DIMENSION OUTPUT.rs [--validate] [--workers N] [--partitions N] [--memory-mib N] [--weight-cache-mib N] [--max-output-mib N] [--serial]");
let validation=false,serial=false;
const options:DirectPoolOptions={};
const numericFlags:Record<string,keyof DirectPoolOptions>={"--workers":"workers","--partitions":"partitions",
  "--memory-mib":"memoryMiB","--heap-mib":"heapMiB","--weight-cache-mib":"weightCacheMiB",
  "--partition-coordinate":"partitionCoordinate","--max-output-mib":"maxOutputMiB"};
for(let i=0;i<flags.length;i++){
  const flag=flags[i]!;
  if(flag==="--validate")validation=true;
  else if(flag==="--serial")serial=true;
  else{
    const key=numericFlags[flag],raw=flags[++i];
    if(!key||raw===undefined||!/^\d+$/.test(raw))throw new Error(`Invalid compiler option: ${flag}`);
    (options as Record<string,unknown>)[key]=Number(raw);
  }
}
if(serial){
  if(options.workers!==undefined||options.partitions!==undefined||options.partitionCoordinate!==undefined||
    options.memoryMiB!==undefined||options.heapMiB!==undefined)throw new Error("Pool options cannot be used with --serial");
  await writeDirectFlatRustModel(directory,python,Number(rawDimension),path,{
    ...(options.weightCacheMiB!==undefined?{weightCacheBytes:options.weightCacheMiB*1024*1024}:{}),
    ...(options.maxOutputMiB!==undefined?{maxOutputBytes:options.maxOutputMiB*1024*1024}:{})});
}else await writeParallelDirectFlatRustModel(directory,python,Number(rawDimension),path,options);
const metadata=JSON.parse(await readFile(path+".reduction.json","utf8"));
if(!Number.isSafeInteger(metadata.inputWidth)||metadata.inputWidth<1)throw new Error("Missing discovered embedding width");
await appendFile(path,`\nfn main(){let tokens:Vec<[f64;${metadata.inputWidth}]>=std::env::args().skip(1).map(|row|row.split(',').map(|x|x.parse().expect("F16 embedding value widened to f64")).collect::<Vec<f64>>().try_into().expect("Discovered embedding width")).collect();for t in 0..tokens.len(){println!("{}",compiled_dimension(&tokens,t).to_bits());}}\n`);
if(validation){
  const run=promisify(execFile),executable=path+".executable",report=path+".parity.json";
  try{
    // The emitter admits only a complete flat expression. No draft is compiled.
    if(metadata.status!=="emitted")throw new Error("Complete expression was not admitted");
    console.log("Complete expression emitted; compiling Rust.");
    await run("rustc",["--edition=2021","-Awarnings","-C","opt-level=0",path,"-o",executable],{maxBuffer:1024*1024});
    metadata.executableProduced=true;
    console.log("Rust executable produced; comparing final logit bits with PyTorch.");
    const helper=new URL("../../helpers/validate_direct_flat_rust.py",import.meta.url).pathname;
    const {stdout}=await run(python,[helper,directory,executable,rawDimension,"--report",report],{maxBuffer:1024*1024});
    metadata.finalParity=true;metadata.parityReport=report;console.log(stdout.trim());
  }catch(error){
    metadata.finalParity=false;metadata.validationFailure=error instanceof Error?error.message:String(error);throw error;
  }finally{await writeFile(path+".reduction.json",JSON.stringify(metadata,null,2)+"\n");}
}
