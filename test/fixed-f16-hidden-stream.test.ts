import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { inlineFixedF16HiddenCall, rewriteFixedF16HiddenFile } from "../src/fixed-f16-hidden-inline.js";

test("streaming substitutes hidden calls across read boundaries without changing surrounding text", async () => {
  const dir = await mkdtemp(join(tmpdir(), "f16-hidden-stream-"));
  try {
    const input = join(dir, "input.txt"), output = join(dir, "output.txt");
    const source = `before${"x".repeat(251)}+hidden_0(t,31)+hidden_1(t,7)${"y".repeat(273)}after`;
    await writeFile(input, source);
    const result = await rewriteFixedF16HiddenFile(input, output, 7);
    const expected = source.replace(/\bhidden_(\d+)\(t,(\d+)\)/g,
      (_, layer: string, dimension: string) => inlineFixedF16HiddenCall(layer, dimension));
    assert.equal(await readFile(output, "utf8"), expected);
    assert.deepEqual(result, { replacements: 2, bytes: Buffer.byteLength(expected) });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
