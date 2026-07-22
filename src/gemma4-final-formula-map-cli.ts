import { resolve } from "node:path";
import { bindGemma4FinalFormulaMap } from "./gemma4-compiled-bundle.js";

const values = new Map<string, string>();
for (let index = 2; index < process.argv.length; index += 2) { const flag = process.argv[index], value = process.argv[index + 1]; if (flag !== "--bundle" || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
const bundle = values.get("--bundle"); if (!bundle) throw new Error("--bundle é obrigatório.");
const manifest = await bindGemma4FinalFormulaMap(resolve(bundle));
process.stdout.write(`${JSON.stringify(manifest.finalFormulaMap, null, 2)}\n`);
