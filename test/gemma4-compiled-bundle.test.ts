import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createGemma4CompiledBundle } from "../src/gemma4-compiled-bundle.js";

test("empacota grafo fechado, constant pool e tokenizer sem caminhos externos", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-compiled-bundle-")), tokenizer = join(directory, "tokenizer"), output = join(directory, "bundle");
  const { mkdir } = await import("node:fs/promises"); await mkdir(tokenizer);
  const graph = join(directory, "graph.json"), constants = join(directory, "constants.json"), globalSsa = join(directory, "global.json");
  await writeFile(graph, '{"kind":"gemma4-parametric-reverse-algebraic-composition","schemaVersion":1,"output":{"family":"terminal_logit","dimension":0,"parameters":{}},"graph":{"nodes":[{"id":"sha256:a"}]},"root":"sha256:a","steps":[],"remainingFunctionCalls":[],"inputVector":{"tensor":"x","length":3}}\n');
  await writeFile(constants, "constant-pool");
  await writeFile(globalSsa, `{"constantMemory":{"kind":"authenticated-gemma4-literal-artifact","artifact":"${constants}"},"outputs":[{"assignment":"calc_terminal_logit_0"},{"assignment":"calc_terminal_logit_1"}]}\n`);
  for (const file of ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json"]) await writeFile(join(tokenizer, file), `{${JSON.stringify(file)}:true}`);
  try {
    const manifest = await createGemma4CompiledBundle({ graph, globalSsa, constantArtifact: constants, tokenizerDirectory: tokenizer, outputDirectory: output });
    assert.equal(manifest.formula.root, "sha256:a"); assert.equal(manifest.formula.inputLength, 3); assert.equal(manifest.formula.expressionNodes, 1);
    assert.deepEqual(manifest.runtimeLowering, { engine: "paged-literal-vectorized", directlyExecutesGlobalFormula: false, globalFormulaRole: "algebraic-source-and-scalar-reference" });
    assert.equal(manifest.files.length, 7); assert.ok(manifest.files.every((entry) => !entry.file.includes(directory)));
    assert.deepEqual(manifest.globalProgram, { file: "global-formulas.ssa.json", terminalLogits: 2, constantPool: "constants.literal.json" });
    assert.match(await readFile(join(output, "global-formulas.ssa.json"), "utf8"), /"artifact":"constants\.literal\.json"/);
    assert.deepEqual(JSON.parse(await readFile(join(output, "manifest.json"), "utf8")), manifest);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
