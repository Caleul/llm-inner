import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { GEMMA4_FINAL_FORMULA_NODE_SEMANTICS, validateGemma4FinalFormulaRuntimeFile, writeGemma4FinalFormulaRuntime, type Gemma4FinalFormulaRuntimeContract } from "../src/gemma4-final-formula-runtime.js";

const hash = (digit: string) => digit.repeat(64);

function contract(): Gemma4FinalFormulaRuntimeContract {
  return {
    kind: "gemma4-final-formula-runtime", schemaVersion: 1, evaluator: "EVAL_EXACT_DAG", semantics: "gemma4-exact-real-simplified-v1", publicFormula: "BF16_RNE(EVAL_EXACT_DAG(root,x))",
    input: { tensor: "x", length: 79_872, onlyFreeInput: true }, output: { family: "terminal_logit", functions: 262_144, finalQuantization: "BF16-round-to-nearest-ties-to-even" },
    artifacts: {
      formulaMap: { file: "final-formulas.json", sha256: hash("1"), orderedRootsSha256: hash("2") },
      globalSsa: { file: "global-formulas.ssa.json", sha256: hash("3"), standaloneOutputsSha256: hash("4") },
      constantPool: { file: "constants.literal.json", sha256: hash("5") },
      loweringPlan: { file: "vectorized-real-lowering.json", sha256: hash("6"), functionBindingsSha256: hash("7"), outputBindingsSha256: hash("8"), realSimplifiedProgramSha256: hash("9") },
    },
    evaluation: { rootResolution: "calc_final_n -> ordered terminal_logit output root -> dependency-ordered SSA node", functionCalls: "lexical parameter binding followed by evaluation of the referenced operation-function root", reductionOrder: "ascending integer index; arguments retain serialized order", intermediateIeeeRounding: "none", nodeSemantics: GEMMA4_FINAL_FORMULA_NODE_SEMANTICS },
    execution: { engine: "mlx-f32-real-decoder-stack-v1", mode: "compiled-parametric-output-program", parallelism: "metal-vectorized-output-dimensions", compiledOutputProgram: { id: "gemma4-text-real-final-vectors", semantics: "shared-dag-parametric-output-functions-v1", logicalDispatchesPerForward: 1, operationFunctions: 1_221, outputFunctions: 264_704, firstOperationId: "embedding", terminalOperationId: "final_logit_softcap", orderedDispatchSha256: hash("a") } },
  };
}

test("publica uma definição autenticada e executável de EVAL_EXACT_DAG", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-final-runtime-")), path = join(directory, "final-formulas.runtime.json");
  try {
    const descriptor = await writeGemma4FinalFormulaRuntime(path, contract());
    const opened = await validateGemma4FinalFormulaRuntimeFile(path, descriptor.sha256);
    assert.equal(opened.output.functions, 262_144); assert.equal(opened.input.onlyFreeInput, true); assert.equal(opened.evaluation.nodeSemantics["function-call"], "evaluate(operationFunctions[functionId].root, bind(parameters, arguments))");
    await assert.rejects(() => validateGemma4FinalFormulaRuntimeFile(path, hash("f")), /SHA-256 autenticado/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("rejeita semântica de redução ou arredondamento não publicada", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-final-runtime-invalid-")), path = join(directory, "invalid.json");
  try {
    const invalid = contract() as unknown as { evaluation: { intermediateIeeeRounding: string } }; invalid.evaluation.intermediateIeeeRounding = "por camada";
    const bytes = Buffer.from(JSON.stringify(invalid)); await writeFile(path, bytes);
    await assert.rejects(() => validateGemma4FinalFormulaRuntimeFile(path, createHash("sha256").update(bytes).digest("hex")), /contrato incompleto/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
