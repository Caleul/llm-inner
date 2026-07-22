import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { validateGemma4FinalFormulaMapFile, writeGemma4FinalFormulaMap } from "../src/gemma4-final-formula-map.js";

test("materializa todas as dimensões finais como calc_final_n = F_n(x)", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-final-map-")), source = join(directory, "global.json"), output = join(directory, "final.json");
  const binding = (assignment: string, value: string) => ({ assignment, value, parameters: ["batch", "sequence"], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" });
  await writeFile(source, JSON.stringify({ functions: [{ id: "ignored" }], outputs: [binding("calc_final_hidden_dimension_0", "hidden"), binding("calc_terminal_logit_0", "sha256:" + "a".repeat(64)), binding("calc_terminal_logit_1", "sha256:" + "b".repeat(64))] }));
  try {
    const roots = [`sha256:${"a".repeat(64)}`, `sha256:${"b".repeat(64)}`];
    const orderedRootsSha256 = createHash("sha256").update(roots.map((root, dimension) => `${dimension}:${root}\n`).join(""), "utf8").digest("hex");
    assert.deepEqual(await writeGemma4FinalFormulaMap(source, output, 1, 2), { functions: 2, orderedRootsSha256 });
    assert.deepEqual(JSON.parse(await readFile(output, "utf8")), {
      calc_final_0: `BF16_RNE(EVAL_EXACT_DAG("sha256:${"a".repeat(64)}", x))`,
      calc_final_1: `BF16_RNE(EVAL_EXACT_DAG("sha256:${"b".repeat(64)}", x))`,
    });
    const bytes = await readFile(output);
    assert.deepEqual(await validateGemma4FinalFormulaMapFile(output, { fileSha256: createHash("sha256").update(bytes).digest("hex"), functions: 2, orderedRootsSha256 }), { functions: 2, fileSha256: createHash("sha256").update(bytes).digest("hex"), orderedRootsSha256 });
    await writeFile(output, `${bytes.toString("utf8").replace("calc_final_1", "calc_final_2")}`);
    await assert.rejects(validateGemma4FinalFormulaMapFile(output, { fileSha256: createHash("sha256").update(await readFile(output)).digest("hex"), functions: 2, orderedRootsSha256 }), /calc_final_1/);
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
