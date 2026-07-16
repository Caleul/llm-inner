import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

interface Assignment {
  output: string;
  formula: string;
  substitutedConstants?: number[];
  value: number;
}

interface Example {
  inputs: { x0: [number, number] };
  layers: { assignments: Assignment[] }[];
  expectedOutputs: { x2: [number, number] };
}

test("scalar-substitution example contains concrete learned values and reproduces its F32 result", async () => {
  const artifact = JSON.parse(await readFile(path.join(process.cwd(), "docs/examples/literal-scalar-substitution.example.json"), "utf8")) as Example;
  const assignments = new Map(artifact.layers.flatMap((layer) => layer.assignments).map((assignment) => [assignment.output, assignment]));
  const value = (name: string): number => assignments.get(name)?.value ?? assert.fail(`Missing assignment ${name}`);
  const f = Math.fround;
  const [x00, x01] = artifact.inputs.x0;

  assert.equal(value("mean_0"), f(f(x00 + x01) / 2));
  assert.equal(value("h0[0]"), f(f(x00 * f(3.456812134)) + f(x01 * f(-0.125)) + f(0.75)));
  assert.equal(value("h0[1]"), f(f(x00 * f(-0.5)) + f(x01 * f(1.25)) + f(-0.25)));
  assert.equal(value("x1[0]"), f(value("h0[0]") * value("mean_0")));
  assert.equal(value("x1[1]"), f(value("h0[1]") * value("mean_0")));
  assert.equal(value("mean_1"), f(f(value("x1[0]") + value("x1[1]")) / 2));
  assert.equal(value("x2[0]"), f(f(value("x1[0]") * f(0.25)) + f(value("x1[1]") * f(2)) + f(-0.5)));
  assert.equal(value("x2[1]"), f(f(value("x1[0]") * f(-1.5)) + f(value("x1[1]") * f(0.5)) + f(0.125)));
  assert.deepEqual(artifact.expectedOutputs.x2, [value("x2[0]"), value("x2[1]")]);

  for (const assignment of assignments.values()) {
    assert.doesNotMatch(assignment.formula, /weight\[/, `${assignment.output} must not hide a learned scalar behind weight[index]`);
    if (assignment.substitutedConstants) assert.ok(assignment.substitutedConstants.every(Number.isFinite));
  }
});
