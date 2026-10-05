import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { jsonConstant as constant, jsonInput, jsonOperation as operation, type JsonExpression } from "../src/direct-json-expression.js";
import { createJsonModelLowerer } from "../src/direct-json-lower-model.js";
import { evaluateJsonExpression } from "../src/direct-json-evaluator.js";
import { auditJsonExpression, writeJsonScalarUnits, type JsonScalarHeader } from "../src/direct-json-stream.js";
import { decodeIeeeF16ToF32 } from "../src/utils.js";

// Independent ordered F32 reference: no stored intermediate answers or JSON
// evaluator is used to construct the expected result.
function reference(x: number, y: number): [number, number] {
  const f = Math.fround;
  const mean = f(f(x + y) / 2);
  const h0 = f(f(f(x * f(3.456812134)) + f(y * f(-0.125))) + f(0.75));
  const h1 = f(f(f(x * f(-0.5)) + f(y * f(1.25))) + f(-0.25));
  const a = f(h0 * mean), b = f(h1 * mean);
  return [f(f(f(a * f(0.25)) + f(b * f(2))) + f(-0.5)),
    f(f(f(a * f(-1.5)) + f(b * f(0.5))) + f(0.125))];
}

function bits(value: number): bigint {
  const word = new DataView(new ArrayBuffer(8));
  word.setFloat64(0, value);
  return word.getBigUint64(0);
}

test("compiled scalar substitution emits input-only expressions and reproduces ordered F32 bits", async () => {
  const directory = await mkdtemp(join(tmpdir(), "compiled-scalar-example-"));
  try {
    const x = operation("widen", "f32", jsonInput("f16", "X1"));
    const y = operation("widen", "f32", jsonInput("f16", "X2"));
    const add = (a: JsonExpression, b: JsonExpression) => operation("add", "f32", a, b);
    const multiply = (a: JsonExpression, coefficient: number) => operation("mul", "f32", a, constant("f32", coefficient));
    const mean = operation("div", "f32", add(x, y), constant("f32", 2));
    const h0 = add(add(multiply(x, 3.456812134), multiply(y, -0.125)), constant("f32", 0.75));
    const h1 = add(add(multiply(x, -0.5), multiply(y, 1.25)), constant("f32", -0.25));
    const a = operation("mul", "f32", h0, mean), b = operation("mul", "f32", h1, mean);
    const outputs = [add(add(multiply(a, 0.25), multiply(b, 2)), constant("f32", -0.5)),
      add(add(multiply(a, -1.5), multiply(b, 0.5)), constant("f32", 0.125))];
    const facts = { halfSources: new WeakMap(), positiveNormalRoots: new WeakSet(),
      exponentialBounds: new WeakMap(), activationBounds: new WeakMap() };
    let substitutions = 0;
    const lowerer = createJsonModelLowerer(facts, new WeakMap(), {
      incremental: true, onSubstitution: () => substitutions++,
    });
    const closed = outputs.map((output) => lowerer.lower(output));
    assert.ok(substitutions > 10);
    const header: JsonScalarHeader = { schema: "direct-scalar-json-v1", inputWidth: 2,
      context: 1, outputWidth: 2, inputs: {
        X1: { dtype: "f64", inputDtype: "f16", tokenPosition: 0, coordinate: 0 },
        X2: { dtype: "f64", inputDtype: "f16", tokenPosition: 0, coordinate: 1 },
      } };
    const path = join(directory, "compiled.jsonl");
    async function* units() {
      for (const [dimension, expression] of closed.entries()) yield { position: 0, dimension, expression };
    }
    const report = await writeJsonScalarUnits(path, header, units());
    assert.equal(report.units, 2);
    const records = (await readFile(path, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    const emitted: JsonExpression[] = records.filter((record) => record.kind === "scalar").map((record) => record.expression);
    assert.equal(emitted.length, 2);
    for (const expression of emitted) auditJsonExpression(expression, header.inputs);
    const finite = [0, -0, 2 ** -24, -(2 ** -24), 2 ** -14, -(2 ** -14),
      0.125, -0.5, 1, -1, 65504, -65504];
    const corpus: [number, number][] = finite.flatMap((left) => finite.map((right): [number, number] => [left, right]));
    for (let i = 0; i < 256; i++) {
      const left = decodeIeeeF16ToF32((i * 251) & 0xffff);
      const right = decodeIeeeF16ToF32((i * 509) & 0xffff);
      if (Number.isFinite(left) && Number.isFinite(right)) corpus.push([left, right]);
    }
    for (const [left, right] of corpus) {
      const expected = reference(left, right);
      for (const [dimension, expression] of emitted.entries()) {
        const actual = Number(evaluateJsonExpression(expression, { X1: left, X2: right }, { allowPendingPrimitives: false }));
        assert.equal(bits(actual), bits(expected[dimension]!), `coordinate=${dimension} inputs=${left},${right}`);
      }
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
