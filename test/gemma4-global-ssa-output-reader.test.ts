import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readGemma4GlobalSsaOutput, resolveGemma4GlobalSsaRoot } from "../src/gemma4-global-ssa-output-reader.js";

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

test("resolve EVAL_EXACT_DAG até a chamada e o corpo matemático autenticado", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-ssa-root-resolution-")), path = join(directory, "global.json");
  const node = <T extends Record<string, unknown>>(payload: T) => ({ id: `sha256:${createHash("sha256").update(JSON.stringify(payload)).digest("hex")}`, ...payload });
  const parameter = node({ kind: "integer-parameter", name: "operation:softcap:output_feature" });
  const scale = node({ kind: "rational", value: { numerator: "30", denominator: "1" } });
  const body = node({ kind: "multiply", arguments: [parameter.id, scale.id] });
  const call = node({ kind: "function-call", functionId: "operation:softcap", arguments: [parameter.id] });
  try {
    await writeFile(path, JSON.stringify({
      statements: [{ target: parameter.id, expression: parameter.name, node: parameter }, { target: scale.id, expression: "30/1", node: scale }, { target: body.id, expression: `(${parameter.id} * ${scale.id})`, node: body }, { target: call.id, expression: `operation:softcap(${parameter.id})`, node: call }],
      functions: [{ functionId: "operation:softcap", operationId: "softcap", ordinal: 1, output: "logits", parameters: [{ name: "output_feature", node: parameter.id }], root: body.id, predecessorFunctions: [], closureKind: "global-output-closure", operandBoundaries: [] }],
      outputs: [],
    }));
    const resolved = await resolveGemma4GlobalSsaRoot(path, call.id);
    assert.equal(resolved.root.expression, `operation:softcap(${parameter.id})`);
    assert.deepEqual(resolved.dependencies, [parameter.id]);
    assert.equal(resolved.calledFunction?.operationId, "softcap");
    assert.equal(resolved.calledFunction?.body.expression, `(${parameter.id} * ${scale.id})`);
    assert.deepEqual(resolved.calledFunction?.bodyDependencies, [parameter.id, scale.id]);
    assert.equal(resolved.calledFunction?.bodyFragment.expandedExpression, `((operation:softcap:output_feature) * (30/1))`);
    assert.equal(resolved.calledFunction?.bodyFragment.truncated, false);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
