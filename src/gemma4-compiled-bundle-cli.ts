import { resolve } from "node:path";
import { createGemma4CompiledBundle } from "./gemma4-compiled-bundle.js";

const values = new Map<string, string>();
for (let index = 2; index < process.argv.length; index += 2) { const flag = process.argv[index], value = process.argv[index + 1]; if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`); values.set(flag, value); }
for (const flag of values.keys()) if (!["--graph", "--global-ssa", "--runtime-model", "--constants", "--tokenizer-directory", "--output"].includes(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
const required = (flag: string): string => { const value = values.get(flag); if (!value) throw new Error(`${flag} é obrigatório.`); return resolve(value); };
const manifest = await createGemma4CompiledBundle({ graph: required("--graph"), globalSsa: required("--global-ssa"), ...(values.get("--runtime-model") ? { runtimeModel: resolve(values.get("--runtime-model")!) } : {}), constantArtifact: required("--constants"), tokenizerDirectory: required("--tokenizer-directory"), outputDirectory: required("--output") });
process.stdout.write(`${JSON.stringify(manifest, null, 2)}\n`);
