import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { writeGemma4FinalFormulaMap } from "../src/gemma4-final-formula-map.js";

test("materializa todas as dimensões finais como calc_final_n = F_n(x)", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-final-map-")), source = join(directory, "global.json"), output = join(directory, "final.json");
  const binding = (assignment: string, value: string) => ({ assignment, value, parameters: ["batch", "sequence"], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" });
  await writeFile(source, JSON.stringify({ functions: [{ id: "ignored" }], outputs: [binding("calc_final_hidden_dimension_0", "hidden"), binding("calc_terminal_logit_0", "sha256:" + "a".repeat(64)), binding("calc_terminal_logit_1", "sha256:" + "b".repeat(64))] }));
  try {
    assert.deepEqual(await writeGemma4FinalFormulaMap(source, output, 1, 2), { functions: 2 });
    assert.deepEqual(JSON.parse(await readFile(output, "utf8")), {
      calc_final_0: `BF16_RNE(EVAL_EXACT_DAG("sha256:${"a".repeat(64)}", x))`,
      calc_final_1: `BF16_RNE(EVAL_EXACT_DAG("sha256:${"b".repeat(64)}", x))`,
    });
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("gerador falha fechado para binding ou cardinalidade divergentes", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-final-map-invalid-")), source = join(directory, "global.json");
  const binding = { assignment: "calc_terminal_logit_1", value: "root", parameters: [], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" };
  await writeFile(source, JSON.stringify({ outputs: [binding] }));
  try {
    await assert.rejects(writeGemma4FinalFormulaMap(source, join(directory, "bad.json"), 0, 1), /terminal_logit\[0\] inválido/);
    await assert.rejects(writeGemma4FinalFormulaMap(source, join(directory, "missing.json"), 1, 2), /0\/2 fórmulas/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
