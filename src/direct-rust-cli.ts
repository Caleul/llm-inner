import { writeDirectRustModel } from "./direct-rust-model.js";
import { appendFile } from "node:fs/promises";
const [directory,python,rawDimension,path]=process.argv.slice(2);
if(!directory||!python||!rawDimension||!path)throw new Error("Usage: direct-rust-cli CHECKPOINT PYTHON DIMENSION OUTPUT.rs");
await writeDirectRustModel(directory,python,Number(rawDimension),path);
await appendFile(path,"\nfn main(){let tokens:Vec<usize>=std::env::args().skip(1).map(|x|x.parse().expect(\"integer token ID\")).collect();for t in 0..tokens.len(){println!(\"{}\",compiled_dimension(&tokens,t).to_bits());}}\n");
