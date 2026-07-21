import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type { Gemma4LiteralArtifactIntegrityManifest } from "./gemma4-literal-artifact-integrity.js";
import type { Gemma4LiteralCalculationGraph } from "./gemma4-literal-calculation-graph.js";
import type { Gemma4ParametricExactRealProgram } from "./gemma4-parametric-global-real-program.js";

const KERNEL_BY_OPERATION = {
  activation: "mlx-real-activation",
  elementwise: "mlx-real-elementwise",
  linear: "mlx-real-linear",
  reshape_heads: "mlx-real-head-layout",
  rms_norm: "mlx-real-rms-norm",
  rotary_embedding: "mlx-real-rope",
  scaled_dot_product_attention: "mlx-real-attention",
  select_per_layer: "mlx-real-layer-select",
  tensor_scale: "mlx-real-tensor-scale",
} as const;

export type Gemma4VectorizedRealOperation = keyof typeof KERNEL_BY_OPERATION;
export type Gemma4VectorizedRealKernel = typeof KERNEL_BY_OPERATION[Gemma4VectorizedRealOperation];

export interface Gemma4VectorizedRealFunctionBinding {
  functionId: string;
  operationId: string;
  ordinal: number;
  output: string;
  operation: Gemma4VectorizedRealOperation;
  kernel: Gemma4VectorizedRealKernel;
  root: string;
  predecessorFunctions: string[];
}

export interface Gemma4VectorizedRealLoweringContract {
  kind: "gemma4-vectorized-real-lowering-contract";
  schemaVersion: 1;
  semantics: "gemma4-exact-real-simplified-v1";
  execution: {
    engine: "mlx-f32-real-decoder-stack-v1";
    mode: "architectural-vector-lowering";
    intermediateBf16Boundaries: 0;
    finalQuantization: "BF16-round-to-nearest-ties-to-even";
    directlyLoadsStandaloneSsaFile: false;
    sourceProgramValidated: true;
  };
  source: {
    artifactIntegritySha256: string;
    realSimplifiedProgramSha256: string;
    sourceAssignments: number;
    expressionNodes: number;
    operationFunctions: number;
    globalClosureFunctions: number;
    standaloneRuntimeReductions: number;
    outputFunctions: number;
  };
  coverage: {
    unresolvedRuntimeReductions: 0;
    intermediateIeeeRoundingNodes: 0;
    unsupportedOperations: 0;
    globalClosuresDependingOnStandaloneReductions: 0;
    kernels: Record<Gemma4VectorizedRealOperation, number>;
  };
  functionBindingsSha256: string;
}

export interface Gemma4VectorizedRealLoweringPlan {
  kind: "gemma4-vectorized-real-lowering-plan";
  schemaVersion: 1;
  contract: Gemma4VectorizedRealLoweringContract;
  functionBindings: Gemma4VectorizedRealFunctionBinding[];
}

export interface Gemma4VectorizedRealDispatchedOperation {
  id: string;
  op: string;
  output: string;
}

export interface Gemma4VectorizedRealExecutionReceipt {
  kind: "gemma4-vectorized-real-execution-receipt";
  schemaVersion: 1;
  completeGlobalClosure: true;
  dispatchedFunctions: number;
  firstOrdinal: number;
  lastOrdinal: number;
  functionBindingsSha256: string;
  orderedDispatchSha256: string;
}

export interface Gemma4VectorizedRealExecutionSummary extends Gemma4VectorizedRealExecutionReceipt {
  authorizedDispatches: number;
}

/**
 * Authorizes the exact operation sequence crossing into the fused Metal hot
 * path. Startup validation proves the persisted plan matches the artifact;
 * this guard additionally proves that the actual dispatch consumes the whole
 * global closure, in order, without an omitted or substituted operation.
 */
export class Gemma4VectorizedRealExecutionGuard {
  readonly #plan: Gemma4VectorizedRealLoweringPlan;
  #authorizedDispatches = 0;
  #lastReceipt?: Gemma4VectorizedRealExecutionReceipt;

  constructor(plan: Gemma4VectorizedRealLoweringPlan) {
    validateGemma4VectorizedRealLoweringPlan(plan);
    this.#plan = plan;
  }

  authorize(operations: readonly Gemma4VectorizedRealDispatchedOperation[]): Gemma4VectorizedRealExecutionReceipt {
    if (operations.length !== this.#plan.functionBindings.length) {
      throw new Error(`Despacho real cobre ${operations.length} operações, mas o plano exige ${this.#plan.functionBindings.length}.`);
    }
    for (let index = 0; index < operations.length; index += 1) {
      const operation = operations[index]!, binding = this.#plan.functionBindings[index]!;
      if (operation.id !== binding.operationId || operation.op !== binding.operation || operation.output !== binding.output ||
        (index > 0 && binding.ordinal <= this.#plan.functionBindings[index - 1]!.ordinal)) {
        throw new Error(`Despacho real diverge do binding persistido na posição ${index}: ${operation.id}.`);
      }
    }
    const first = this.#plan.functionBindings[0]!, last = this.#plan.functionBindings.at(-1)!;
    const receipt: Gemma4VectorizedRealExecutionReceipt = {
      kind: "gemma4-vectorized-real-execution-receipt",
      schemaVersion: 1,
      completeGlobalClosure: true,
      dispatchedFunctions: operations.length,
      firstOrdinal: first.ordinal,
      lastOrdinal: last.ordinal,
      functionBindingsSha256: this.#plan.contract.functionBindingsSha256,
      orderedDispatchSha256: createHash("sha256").update(JSON.stringify(operations), "utf8").digest("hex"),
    };
    this.#authorizedDispatches += 1;
    this.#lastReceipt = receipt;
    return receipt;
  }

  summary(): Gemma4VectorizedRealExecutionSummary | undefined {
    return this.#lastReceipt ? { ...this.#lastReceipt, authorizedDispatches: this.#authorizedDispatches } : undefined;
  }
}

/**
 * Fail-closed certificate connecting the authenticated scalar/parametric
 * program to the operation families implemented by the vectorized MLX stack.
 * It deliberately does not claim that the standalone SSA JSON is parsed at
 * runtime: the artifact-embedded program and its manifest commitment are the
 * executable source identity used here.
 */
export function buildGemma4VectorizedRealLoweringContract(
  program: Gemma4ParametricExactRealProgram,
  graph: Gemma4LiteralCalculationGraph,
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest,
): Gemma4VectorizedRealLoweringContract {
  return buildGemma4VectorizedRealLoweringPlan(program, graph, integrityManifest).contract;
}

/** Produces the compact, persistable dispatch plan consumed at worker startup. */
export function buildGemma4VectorizedRealLoweringPlan(
  program: Gemma4ParametricExactRealProgram,
  graph: Gemma4LiteralCalculationGraph,
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest,
): Gemma4VectorizedRealLoweringPlan {
  if (program.semantics !== "gemma4-exact-real-simplified-v1" ||
    program.coverage.unresolvedRuntimeReductions !== 0 ||
    program.coverage.intermediateIeeeRoundingNodes !== 0) {
    throw new Error("Lowering vetorizado real requer programa global sem reduções ou arredondamentos intermediários não resolvidos.");
  }
  const realSection = integrityManifest.sections.find((section) => section.name === "realSimplifiedProgram");
  if (!realSection || !/^[0-9a-f]{64}$/.test(realSection.sha256) || !/^[0-9a-f]{64}$/.test(integrityManifest.rootSha256)) {
    throw new Error("Lowering vetorizado real requer compromisso autenticado do programa simplificado.");
  }
  const assignments = new Map(graph.assignments.map((assignment) => [assignment.operationId, assignment]));
  const functions = new Map(program.operationFunctions.map((entry) => [entry.functionId, entry]));
  const global = program.operationFunctions.filter((entry) => entry.closureKind === "global-output-closure");
  const standalone = program.operationFunctions.filter((entry) => entry.closureKind === "standalone-runtime-reduction");
  const kernelCounts = Object.fromEntries(Object.keys(KERNEL_BY_OPERATION).map((operation) => [operation, 0])) as Record<Gemma4VectorizedRealOperation, number>;
  const bindings: Gemma4VectorizedRealFunctionBinding[] = global.map((entry) => {
    const assignment = assignments.get(entry.operationId);
    if (!assignment || assignment.ordinal !== entry.ordinal || assignment.output !== entry.output) {
      throw new Error(`${entry.functionId}: função real não corresponde ao grafo literal autenticado.`);
    }
    if (entry.predecessorFunctions.some((functionId) => functions.get(functionId)?.closureKind === "standalone-runtime-reduction")) {
      throw new Error(`${entry.functionId}: closure global ainda depende de redução standalone.`);
    }
    if (!(assignment.operation in KERNEL_BY_OPERATION)) {
      throw new Error(`${entry.functionId}: operação sem lowering vetorizado: ${assignment.operation}.`);
    }
    const operation = assignment.operation as Gemma4VectorizedRealOperation;
    kernelCounts[operation] += 1;
    return {
      functionId: entry.functionId,
      operationId: entry.operationId,
      ordinal: entry.ordinal,
      output: entry.output,
      operation,
      kernel: KERNEL_BY_OPERATION[operation],
      root: entry.root,
      predecessorFunctions: entry.predecessorFunctions,
    };
  });
  if (bindings.length === 0 || Object.values(kernelCounts).some((count) => count === 0)) {
    throw new Error("Lowering vetorizado real não cobre todas as famílias de kernel obrigatórias.");
  }
  if (program.outputFunctions.some((output) => output.finalQuantization !== "BF16-round-to-nearest-ties-to-even")) {
    throw new Error("Lowering vetorizado real requer BF16 somente nas saídas públicas finais.");
  }
  const contract: Gemma4VectorizedRealLoweringContract = {
    kind: "gemma4-vectorized-real-lowering-contract",
    schemaVersion: 1,
    semantics: "gemma4-exact-real-simplified-v1",
    execution: {
      engine: "mlx-f32-real-decoder-stack-v1",
      mode: "architectural-vector-lowering",
      intermediateBf16Boundaries: 0,
      finalQuantization: "BF16-round-to-nearest-ties-to-even",
      directlyLoadsStandaloneSsaFile: false,
      sourceProgramValidated: true,
    },
    source: {
      artifactIntegritySha256: integrityManifest.rootSha256,
      realSimplifiedProgramSha256: realSection.sha256,
      sourceAssignments: program.coverage.sourceAssignments,
      expressionNodes: program.expressionGraph.nodes.length,
      operationFunctions: program.operationFunctions.length,
      globalClosureFunctions: global.length,
      standaloneRuntimeReductions: standalone.length,
      outputFunctions: program.outputFunctions.length,
    },
    coverage: {
      unresolvedRuntimeReductions: 0,
      intermediateIeeeRoundingNodes: 0,
      unsupportedOperations: 0,
      globalClosuresDependingOnStandaloneReductions: 0,
      kernels: kernelCounts,
    },
    functionBindingsSha256: createHash("sha256").update(JSON.stringify(bindings), "utf8").digest("hex"),
  };
  return { kind: "gemma4-vectorized-real-lowering-plan", schemaVersion: 1, contract, functionBindings: bindings };
}

export function validateGemma4VectorizedRealLoweringPlan(value: unknown): asserts value is Gemma4VectorizedRealLoweringPlan {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Plano de lowering vetorizado real deve ser um objeto.");
  const plan = value as Partial<Gemma4VectorizedRealLoweringPlan>;
  const contract = plan.contract;
  if (plan.kind !== "gemma4-vectorized-real-lowering-plan" || plan.schemaVersion !== 1 || !contract ||
    contract.kind !== "gemma4-vectorized-real-lowering-contract" || contract.schemaVersion !== 1 ||
    contract.semantics !== "gemma4-exact-real-simplified-v1" || contract.execution?.engine !== "mlx-f32-real-decoder-stack-v1" ||
    contract.execution.intermediateBf16Boundaries !== 0 || contract.execution.finalQuantization !== "BF16-round-to-nearest-ties-to-even" ||
    contract.execution.directlyLoadsStandaloneSsaFile !== false || contract.execution.sourceProgramValidated !== true ||
    contract.coverage?.unresolvedRuntimeReductions !== 0 || contract.coverage.intermediateIeeeRoundingNodes !== 0 ||
    contract.coverage.unsupportedOperations !== 0 || contract.coverage.globalClosuresDependingOnStandaloneReductions !== 0 ||
    !Array.isArray(plan.functionBindings) || plan.functionBindings.length !== contract.source?.globalClosureFunctions) {
    throw new Error("Plano de lowering vetorizado real possui contrato ou cobertura incompleta.");
  }
  const seenFunctions = new Set<string>();
  const kernelCounts = Object.fromEntries(Object.keys(KERNEL_BY_OPERATION).map((operation) => [operation, 0])) as Record<Gemma4VectorizedRealOperation, number>;
  for (const binding of plan.functionBindings) {
    if (!binding || typeof binding !== "object" || typeof binding.functionId !== "string" || seenFunctions.has(binding.functionId) ||
      typeof binding.operationId !== "string" || !Number.isSafeInteger(binding.ordinal) || binding.ordinal < 0 || typeof binding.output !== "string" ||
      !(binding.operation in KERNEL_BY_OPERATION) || binding.kernel !== KERNEL_BY_OPERATION[binding.operation] || typeof binding.root !== "string" ||
      !Array.isArray(binding.predecessorFunctions) || binding.predecessorFunctions.some((entry) => typeof entry !== "string")) {
      throw new Error("Plano de lowering vetorizado real contém binding inválido ou duplicado.");
    }
    seenFunctions.add(binding.functionId);
    kernelCounts[binding.operation] += 1;
  }
  if (!isDeepStrictEqual(kernelCounts, contract.coverage.kernels) ||
    createHash("sha256").update(JSON.stringify(plan.functionBindings), "utf8").digest("hex") !== contract.functionBindingsSha256 ||
    !/^[0-9a-f]{64}$/.test(contract.source.artifactIntegritySha256) || !/^[0-9a-f]{64}$/.test(contract.source.realSimplifiedProgramSha256)) {
    throw new Error("Plano de lowering vetorizado real diverge de seus compromissos e contagens.");
  }
}

export function assertGemma4VectorizedRealLoweringPlanMatches(
  persisted: unknown,
  program: Gemma4ParametricExactRealProgram,
  graph: Gemma4LiteralCalculationGraph,
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest,
): asserts persisted is Gemma4VectorizedRealLoweringPlan {
  validateGemma4VectorizedRealLoweringPlan(persisted);
  const expected = buildGemma4VectorizedRealLoweringPlan(program, graph, integrityManifest);
  if (!isDeepStrictEqual(persisted, expected)) throw new Error("Plano de lowering persistido não corresponde ao programa real autenticado do artefato.");
}
