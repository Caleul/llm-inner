import { resolve } from "node:path";
import { bindGemma4LiteralRuntimeIndex } from "./gemma4-compiled-bundle.js";

const arguments_ = process.argv.slice(2);
if (arguments_.length !== 2 || arguments_[0] !== "--bundle" || !arguments_[1]) throw new Error("Uso: --bundle <diretório-compilado>.");
const manifest = await bindGemma4LiteralRuntimeIndex(resolve(arguments_[1]));
process.stdout.write(`${JSON.stringify({ bundle: resolve(arguments_[1]), runtimeIndex: manifest.runtimeIndex }, null, 2)}\n`);
