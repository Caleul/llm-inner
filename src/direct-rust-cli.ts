import { writeDirectFlatRustModel } from "./direct-flat-rust-model.js";
import { appendFile,readFile,writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
const [directory,python,rawDimension,path,validation]=process.argv.slice(2);
if(!directory||!python||!rawDimension||!path||(validation&&validation!=="--validate"))
  throw new Error("Usage: direct-rust-cli CHECKPOINT PYTHON DIMENSION OUTPUT.rs [--validate]");
await writeDirectFlatRustModel(directory,python,Number(rawDimension),path);
await appendFile(path,"\nfn main(){let tokens:Vec<Vec<f64>>=std::env::args().skip(1).map(|row|row.split(',').map(|x|x.parse().expect(\"F16 embedding value widened to f64\")).collect()).collect();for t in 0..tokens.len(){println!(\"{}\",compiled_dimension(&tokens,t).to_bits());}}\n");
if(validation){
  const run=promisify(execFile),executable=path+".executable",report=path+".parity.json";
  const metadata=JSON.parse(await readFile(path+".reduction.json","utf8"));
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
