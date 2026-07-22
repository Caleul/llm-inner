import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import type { Gemma4VectorizedRealLoweringPlan } from "./gemma4-vectorized-real-lowering.js";

export const GEMMA4_FINAL_FORMULA_NODE_SEMANTICS = {
  "integer-constant": "value",
  "integer-parameter": "environment[name]",
  "integer-bound-index": "bound_environment[name]",
  "integer-add": "left + right",
  "integer-subtract": "left - right",
  "integer-multiply": "left * right",
  "integer-floor-divide": "floor(left / right)",
  "integer-modulo": "euclidean_modulo(left, right)",
  "tensor-axis": "shape(input[tensor])[axis]",
  "stable-true-prefix-rank": "row_major_rank_of_prior_elements_equal_to(tensor, equals, batch, sequence)",
  rational: "integer(value.numerator) / integer(value.denominator)",
  boolean: "value",
  "negative-infinity": "-Infinity",
  "positive-infinity": "+Infinity",
  "input-element": "input[tensor][coordinates]",
  "learned-rational-element": "decode_ieee_little_endian(constant_pool[tensor], storageDtype, coordinates, decoderId)",
  "function-call": "evaluate(operationFunctions[functionId].root, bind(parameters, arguments))",
  add: "sum(arguments)",
  multiply: "product(arguments)",
  minimum: "min(arguments)",
  maximum: "max(arguments)",
  divide: "numerator / denominator",
  "integer-power": "base ** exponent",
  power: "base ** exponent",
  "unary-function": "function(argument)",
  compare: "compare(comparison, left, right)",
  select: "condition ? whenTrue : whenFalse",
  "finite-sum": "sum(index=startInclusive..endExclusive-1, predicate ? body : 0)",
  "finite-maximum": "max(index=startInclusive..endExclusive-1, predicate ? body : -Infinity)",
} as const;

export interface Gemma4FinalFormulaRuntimeContract {
  kind: "gemma4-final-formula-runtime";
  schemaVersion: 1;
  evaluator: "EVAL_EXACT_DAG";
  semantics: "gemma4-exact-real-simplified-v1";
  publicFormula: "BF16_RNE(EVAL_EXACT_DAG(root,x))";
  input: { tensor: "x"; length: number; onlyFreeInput: true };
  output: { family: "terminal_logit"; functions: number; finalQuantization: "BF16-round-to-nearest-ties-to-even" };
  artifacts: {
    formulaMap: { file: "final-formulas.json"; sha256: string; orderedRootsSha256: string };
    globalSsa: { file: "global-formulas.ssa.json"; sha256: string; standaloneOutputsSha256: string };
    constantPool: { file: "constants.literal.json"; sha256: string };
    loweringPlan: { file: "vectorized-real-lowering.json"; sha256: string; functionBindingsSha256: string; outputBindingsSha256: string; realSimplifiedProgramSha256: string };
  };
  evaluation: {
    rootResolution: "calc_final_n -> ordered terminal_logit output root -> dependency-ordered SSA node";
    functionCalls: "lexical parameter binding followed by evaluation of the referenced operation-function root";
    reductionOrder: "ascending integer index; arguments retain serialized order";
    intermediateIeeeRounding: "none";
    nodeSemantics: typeof GEMMA4_FINAL_FORMULA_NODE_SEMANTICS;
  };
  execution: {
    engine: "mlx-f32-real-decoder-stack-v1";
    mode: "compiled-parametric-output-program";
    parallelism: "metal-vectorized-output-dimensions";
    compiledOutputProgram: Gemma4VectorizedRealLoweringPlan["contract"]["compiledOutputProgram"];
  };
}

export interface Gemma4FinalFormulaRuntimeDescriptor {
  file: "final-formulas.runtime.json";
  schemaVersion: 1;
  evaluator: "EVAL_EXACT_DAG";
  sha256: string;
}

export async function writeGemma4FinalFormulaRuntime(
  path: string,
  contract: Gemma4FinalFormulaRuntimeContract,
): Promise<Gemma4FinalFormulaRuntimeDescriptor> {
  validateGemma4FinalFormulaRuntime(contract);
  const bytes = Buffer.from(`${JSON.stringify(contract, null, 2)}\n`, "utf8");
  await writeFile(path, bytes, { flag: "wx" });
  return { file: "final-formulas.runtime.json", schemaVersion: 1, evaluator: "EVAL_EXACT_DAG", sha256: createHash("sha256").update(bytes).digest("hex") };
}

export async function validateGemma4FinalFormulaRuntimeFile(
  path: string,
  expectedSha256: string,
): Promise<Gemma4FinalFormulaRuntimeContract & { fileSha256: string }> {
  if (!/^[0-9a-f]{64}$/.test(expectedSha256)) throw new Error("SHA-256 esperado do runtime de fórmulas é inválido.");
  const info = await stat(path);
  if (!info.isFile() || info.size < 1 || info.size > 1024 * 1024) throw new Error("Runtime de fórmulas deve ser JSON não vazio de até 1 MiB.");
  const bytes = await readFile(path), fileSha256 = createHash("sha256").update(bytes).digest("hex");
  if (fileSha256 !== expectedSha256) throw new Error("Runtime de fórmulas diverge do SHA-256 autenticado.");
  const contract = JSON.parse(bytes.toString("utf8")) as unknown;
  validateGemma4FinalFormulaRuntime(contract);
  return { ...contract, fileSha256 };
}

export function validateGemma4FinalFormulaRuntime(value: unknown): asserts value is Gemma4FinalFormulaRuntimeContract {
  if (!isRecord(value)) throw new Error("Runtime de fórmulas deve ser objeto JSON.");
  const contract = value as unknown as Gemma4FinalFormulaRuntimeContract;
  const hashes = [contract.artifacts?.formulaMap?.sha256, contract.artifacts?.formulaMap?.orderedRootsSha256, contract.artifacts?.globalSsa?.sha256,
    contract.artifacts?.globalSsa?.standaloneOutputsSha256, contract.artifacts?.constantPool?.sha256, contract.artifacts?.loweringPlan?.sha256,
    contract.artifacts?.loweringPlan?.functionBindingsSha256, contract.artifacts?.loweringPlan?.outputBindingsSha256, contract.artifacts?.loweringPlan?.realSimplifiedProgramSha256];
  if (contract.kind !== "gemma4-final-formula-runtime" || contract.schemaVersion !== 1 || contract.evaluator !== "EVAL_EXACT_DAG" ||
    contract.semantics !== "gemma4-exact-real-simplified-v1" || contract.publicFormula !== "BF16_RNE(EVAL_EXACT_DAG(root,x))" ||
    contract.input?.tensor !== "x" || contract.input.onlyFreeInput !== true || !Number.isSafeInteger(contract.input.length) || contract.input.length < 1 ||
    contract.output?.family !== "terminal_logit" || !Number.isSafeInteger(contract.output.functions) || contract.output.functions < 1 || contract.output.finalQuantization !== "BF16-round-to-nearest-ties-to-even" ||
    contract.artifacts?.formulaMap?.file !== "final-formulas.json" || contract.artifacts?.globalSsa?.file !== "global-formulas.ssa.json" || contract.artifacts?.constantPool?.file !== "constants.literal.json" || contract.artifacts?.loweringPlan?.file !== "vectorized-real-lowering.json" ||
    hashes.some((hash) => typeof hash !== "string" || !/^[0-9a-f]{64}$/.test(hash)) ||
    contract.evaluation?.rootResolution !== "calc_final_n -> ordered terminal_logit output root -> dependency-ordered SSA node" || contract.evaluation.functionCalls !== "lexical parameter binding followed by evaluation of the referenced operation-function root" ||
    contract.evaluation.reductionOrder !== "ascending integer index; arguments retain serialized order" || contract.evaluation.intermediateIeeeRounding !== "none" ||
    JSON.stringify(contract.evaluation.nodeSemantics) !== JSON.stringify(GEMMA4_FINAL_FORMULA_NODE_SEMANTICS) ||
    contract.execution?.engine !== "mlx-f32-real-decoder-stack-v1" || contract.execution.mode !== "compiled-parametric-output-program" || contract.execution.parallelism !== "metal-vectorized-output-dimensions" || !isRecord(contract.execution.compiledOutputProgram)) {
    throw new Error("Runtime de fórmulas possui contrato incompleto ou desconhecido.");
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
