import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createGemma4CompiledBundle } from "../src/gemma4-compiled-bundle.js";

test("empacota grafo fechado, constant pool e tokenizer sem caminhos externos", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-compiled-bundle-")), tokenizer = join(directory, "tokenizer"), output = join(directory, "bundle");
  const { mkdir } = await import("node:fs/promises"); await mkdir(tokenizer);
  const graph = join(directory, "graph.json"), constants = join(directory, "constants.json"), globalSsa = join(directory, "global.json");
  const lowering = join(directory, "lowering.json");
  await writeFile(graph, '{"kind":"gemma4-parametric-reverse-algebraic-composition","schemaVersion":1,"output":{"family":"terminal_logit","dimension":0,"parameters":{}},"graph":{"nodes":[{"id":"sha256:a"}]},"root":"sha256:a","steps":[],"remainingFunctionCalls":[],"inputVector":{"tensor":"x","length":3}}\n');
  await writeFile(constants, "constant-pool");
  await writeFile(globalSsa, `{"constantMemory":{"kind":"authenticated-gemma4-literal-artifact","artifact":"${constants}"},"outputs":[{"assignment":"calc_terminal_logit_0","value":"root:output","parameters":[],"coordinate":[],"finalQuantization":"BF16-round-to-nearest-ties-to-even"}]}\n`);
  const functionBindings = [{ functionId: "op:0", operationId: "operation_0", ordinal: 0, output: "value_0", operation: "activation", kernel: "mlx-real-activation", root: "root:0", predecessorFunctions: [] }];
  const functionBindingsSha256 = createHash("sha256").update(JSON.stringify(functionBindings), "utf8").digest("hex");
  const orderedDispatchSha256 = createHash("sha256").update(JSON.stringify([{ id: "operation_0", op: "activation", output: "value_0" }]), "utf8").digest("hex");
  const compiledOutputProgram = { id: "gemma4-text-real-final-vectors", semantics: "shared-dag-parametric-output-functions-v1", logicalDispatchesPerForward: 1, operationFunctions: 1, outputFunctions: 1, firstOperationId: "operation_0", terminalOperationId: "operation_0", orderedDispatchSha256 };
  const outputBinding = { name: "terminal_logit", operationId: "operation_0", fixedDimension: 0, coordinate: [], parameters: [], root: "root:output", finalQuantization: "BF16-round-to-nearest-ties-to-even" };
  const outputBindingsSha256 = createHash("sha256").update(JSON.stringify([outputBinding]), "utf8").digest("hex");
  const standaloneSsaOutputsSha256 = createHash("sha256").update(JSON.stringify([{ assignment: "calc_terminal_logit_0", value: "root:output", parameters: [], coordinate: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" }]), "utf8").digest("hex");
  await writeFile(lowering, JSON.stringify({
    kind: "gemma4-vectorized-real-lowering-plan", schemaVersion: 3,
    contract: {
      kind: "gemma4-vectorized-real-lowering-contract", schemaVersion: 3, semantics: "gemma4-exact-real-simplified-v1",
      execution: { engine: "mlx-f32-real-decoder-stack-v1", mode: "compiled-parametric-output-program", intermediateBf16Boundaries: 0, finalQuantization: "BF16-round-to-nearest-ties-to-even", directlyLoadsStandaloneSsaFile: false, sourceProgramValidated: true, directlyExecutesGlobalFormula: true },
      source: { artifactIntegritySha256: "a".repeat(64), realSimplifiedProgramSha256: "b".repeat(64), sourceAssignments: 1, expressionNodes: 1, operationFunctions: 1, globalClosureFunctions: 1, standaloneRuntimeReductions: 0, outputFunctions: 1, outputBindingsSha256, standaloneSsaOutputsSha256, outputFamilies: { terminal_logit: { dimensions: 1, operationId: "operation_0", parameters: [], finalQuantization: "BF16-round-to-nearest-ties-to-even" } } },
      coverage: { unresolvedRuntimeReductions: 0, intermediateIeeeRoundingNodes: 0, unsupportedOperations: 0, globalClosuresDependingOnStandaloneReductions: 0, kernels: { activation: 1, elementwise: 0, linear: 0, reshape_heads: 0, rms_norm: 0, rotary_embedding: 0, scaled_dot_product_attention: 0, select_per_layer: 0, tensor_scale: 0 } },
      functionBindingsSha256, compiledOutputProgram,
    },
    functionBindings,
  }));
  for (const file of ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "config.json"]) await writeFile(join(tokenizer, file), `{${JSON.stringify(file)}:true}`);
  try {
    const manifest = await createGemma4CompiledBundle({ graph, globalSsa, realLoweringPlan: lowering, constantArtifact: constants, tokenizerDirectory: tokenizer, outputDirectory: output, createRuntimeIndex: false });
    assert.equal(manifest.formula.root, "sha256:a"); assert.equal(manifest.formula.inputLength, 3); assert.equal(manifest.formula.expressionNodes, 1);
    assert.deepEqual(manifest.runtimeLowering, { engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: true, executesPersistedLoweringPlan: true, plan: "vectorized-real-lowering.json", functionBindingsSha256, outputBindingsSha256, standaloneSsaOutputsSha256, outputFunctions: 1, realSimplifiedProgramSha256: "b".repeat(64), globalFormulaRole: "compiled-executable-shared-dag", compiledOutputProgram });
    assert.equal(manifest.schemaVersion, 2); assert.equal(manifest.files.length, 9); assert.ok(manifest.files.every((entry) => !entry.file.includes(directory)));
    assert.deepEqual(manifest.globalProgram, { file: "global-formulas.ssa.json", terminalLogits: 1, constantPool: "constants.literal.json" });
    assert.deepEqual(manifest.finalFormulaMap, { file: "final-formulas.json", functions: 1, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", globalFormulaSha256: manifest.files.find((entry) => entry.role === "global-formulas")!.sha256 });
    assert.deepEqual(JSON.parse(await readFile(join(output, "final-formulas.json"), "utf8")), { calc_final_0: "BF16_RNE(EVAL_EXACT_DAG(\"root:output\", x))" });
    assert.match(await readFile(join(output, "global-formulas.ssa.json"), "utf8"), /"artifact":"constants\.literal\.json"/);
    assert.deepEqual(JSON.parse(await readFile(join(output, "vectorized-real-lowering.json"), "utf8")), JSON.parse(await readFile(lowering, "utf8")));
    assert.deepEqual(JSON.parse(await readFile(join(output, "manifest.json"), "utf8")), manifest);
    await writeFile(globalSsa, `{"constantMemory":{"kind":"authenticated-gemma4-literal-artifact","artifact":"${constants}"},"outputs":[{"assignment":"calc_terminal_logit_0","value":"tampered","parameters":[],"coordinate":[],"finalQuantization":"BF16-round-to-nearest-ties-to-even"}]}\n`);
    await assert.rejects(() => createGemma4CompiledBundle({ graph, globalSsa, realLoweringPlan: lowering, constantArtifact: constants, tokenizerDirectory: tokenizer, outputDirectory: `${output}-tampered`, createRuntimeIndex: false }), /SSA global não corresponde às saídas finais/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
