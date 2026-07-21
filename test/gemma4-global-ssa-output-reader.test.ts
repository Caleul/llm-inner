import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readGemma4GlobalSsaOutput } from "../src/gemma4-global-ssa-output-reader.js";

const root = (digit: string) => `sha256:${digit.repeat(64)}`;

test("lê uma única dimensão final sem depender do tamanho das seções anteriores", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-ssa-output-reader-")), path = join(directory, "global.json");
  const outputs = [0, 1, 2].map((dimension) => ({ assignment: `calc_terminal_logit_${dimension}`, value: root(String(dimension + 1)), parameters: ["batch", "sequence"], coordinate: [root("a"), root("b"), root(String(dimension + 1))], finalQuantization: "BF16-round-to-nearest-ties-to-even" }));
  try {
    await writeFile(path, JSON.stringify({ kind: "fixture", statements: [{ text: "outputs:[{not the marker}" }], functions: [{ nested: { braces: "{}" } }], outputs, coverage: {} }));
    assert.deepEqual(await readGemma4GlobalSsaOutput(path, 1), outputs[1]);
    await assert.rejects(() => readGemma4GlobalSsaOutput(path, 3), /não possui output no ordinal 3/);
    await assert.rejects(() => readGemma4GlobalSsaOutput(path, -1), /inteiro não negativo/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("rejeita binding final sem raiz autenticada", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-ssa-output-invalid-")), path = join(directory, "global.json");
  try {
    await writeFile(path, JSON.stringify({ outputs: [{ assignment: "calc_terminal_logit_0", value: "root externo", parameters: [], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" }] }));
    await assert.rejects(() => readGemma4GlobalSsaOutput(path, 0), /contrato inválido/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
