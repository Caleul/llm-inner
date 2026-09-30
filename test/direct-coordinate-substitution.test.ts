import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { substituteDirectCoordinates } from "../src/direct-coordinate-substitution.js";

test("scalar coordinate substitution streams repeated producer source", async () => {
  const directory = await mkdtemp(join(tmpdir(), "direct-coordinate-"));
  try {
    const input = join(directory, "input"), output = join(directory, "output");
    const producer = [join(directory, "p0"), join(directory, "p1")];
    await Promise.all([
      writeFile(input, "residualInput[t][0]+residualInput[t][1]+residualInput[t][0]"),
      writeFile(producer[0]!, "(embeddings[t][0]*2)"),
      writeFile(producer[1]!, "(embeddings[t][1]*3)"),
    ]);
    const result = await substituteDirectCoordinates(input, output, "residualInput", "t", producer);
    assert.equal(result.replacements, 3);
    assert.equal(await readFile(output, "utf8"),
      "(embeddings[t][0]*2)+(embeddings[t][1]*3)+(embeddings[t][0]*2)");
    await assert.rejects(substituteDirectCoordinates(input, join(directory, "limited"),
      "residualInput", "t", producer, 10), RangeError);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
