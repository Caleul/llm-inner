import { createHash } from "node:crypto";
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
  const bindings = global.map((entry) => {
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
  return {
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
}
