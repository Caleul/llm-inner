import { isDeepStrictEqual } from "node:util";
import type { Gemma4AuthoritativeExecutionContract } from "./gemma4-authoritative-runtime.js";
import type {
  Gemma4LiteralCalculationGraph,
  Gemma4LiteralInstantiatedCalculation,
} from "./gemma4-literal-calculation-graph.js";
import type { Gemma4LiteralCalculationScope } from "./gemma4-literal-domains.js";

export type Gemma4LiteralRuntimeReductionOperationClass =
  | "vision-attention-score"
  | "vision-attention-value"
  | "audio-content-attention-score"
  | "audio-position-attention-score"
  | "audio-attention-value";

export interface Gemma4LiteralUnresolvedRuntimeReduction {
  ordinal: number;
  operationId: string;
  definitionId: string;
  invocationId?: string;
  scope: Gemma4LiteralInstantiatedCalculation["scope"];
  operation: string;
  operationClass: Gemma4LiteralRuntimeReductionOperationClass;
  output: string;
  scalarCalculationPointer: string;
  outputCoordinatePointer: string;
}

/**
 * Machine-readable completion boundary derived from the canonical graph.
 * It never certifies an implementation session: an empty unresolved list only
 * makes the artifact eligible for independent source-removed/differential
 * certification.
 */
export interface Gemma4LiteralFidelityGate {
  kind: "gemma4-literal-fidelity-gate";
  schemaVersion: 1;
  status: "blocked-on-runtime-reduction" | "eligible-for-independent-certification";
  exactReplayClaim: "forbidden" | "not-certified";
  provider: Gemma4AuthoritativeExecutionContract["unresolvedNativeReduction"]["provider"];
  scalarSchedule: Gemma4AuthoritativeExecutionContract["unresolvedNativeReduction"]["scalarSchedule"];
  unresolvedNativeReductionCount: number;
  unresolvedNativeReductions: Gemma4LiteralUnresolvedRuntimeReduction[];
  completionRule: "exact replay remains forbidden while unresolvedNativeReductions is non-empty; an empty list requires fresh independent source-removed replay and authoritative differential certification";
}

export function buildGemma4LiteralFidelityGate(
  graph: Gemma4LiteralCalculationGraph,
  authority: Gemma4AuthoritativeExecutionContract,
): Gemma4LiteralFidelityGate {
  const declaredClasses = new Set(authority.unresolvedNativeReduction.operationClasses);
  const unresolvedNativeReductions = graph.assignments.flatMap((assignment): Gemma4LiteralUnresolvedRuntimeReduction[] => {
    const reduction = assignment.scalarCalculation.reduction;
    if (reduction?.order !== "runtime-defined") return [];
    if (assignment.scalarCalculation.reproducibility !== "fail-closed-runtime-reduction" ||
      assignment.scalarCalculation.dtypePolicy.accumulationDtype !== "runtime-defined" ||
      assignment.outputDomain.dtypePolicy?.accumulationDtype !== "runtime-defined" || reduction.domains.length !== 1) {
      throw new Error(`${assignment.operationId}: redução runtime-defined não está integralmente marcada como fail-closed.`);
    }
    const operationClass = gemma4LiteralRuntimeReductionOperationClass(assignment.scope, assignment.operation);
    if (!declaredClasses.has(operationClass)) {
      throw new Error(`${assignment.operationId}: classe BMM ${operationClass} ausente do contrato autoritativo.`);
    }
    return [{
      ordinal: assignment.ordinal,
      operationId: assignment.operationId,
      definitionId: assignment.definitionId,
      ...(assignment.invocationId ? { invocationId: assignment.invocationId } : {}),
      scope: assignment.scope,
      operation: assignment.operation,
      operationClass,
      output: assignment.output,
      scalarCalculationPointer: `/calculationGraph/assignments/${assignment.ordinal}/scalarCalculation/reduction`,
      outputCoordinatePointer: `/calculationGraph/assignments/${assignment.ordinal}/outputCoordinate`,
    }];
  });
  const blocked = unresolvedNativeReductions.length > 0;
  return {
    kind: "gemma4-literal-fidelity-gate",
    schemaVersion: 1,
    status: blocked ? "blocked-on-runtime-reduction" : "eligible-for-independent-certification",
    exactReplayClaim: blocked ? "forbidden" : "not-certified",
    provider: authority.unresolvedNativeReduction.provider,
    scalarSchedule: authority.unresolvedNativeReduction.scalarSchedule,
    unresolvedNativeReductionCount: unresolvedNativeReductions.length,
    unresolvedNativeReductions,
    completionRule: "exact replay remains forbidden while unresolvedNativeReductions is non-empty; an empty list requires fresh independent source-removed replay and authoritative differential certification",
  };
}

export function validateGemma4LiteralFidelityGate(
  gate: Gemma4LiteralFidelityGate,
  graph: Gemma4LiteralCalculationGraph,
  authority: Gemma4AuthoritativeExecutionContract,
): void {
  if (!isDeepStrictEqual(gate, buildGemma4LiteralFidelityGate(graph, authority))) {
    throw new Error("Artefato literal Gemma 4 possui gate de fidelidade incompleto ou divergente.");
  }
}

export function gemma4LiteralRuntimeReductionOperationClass(
  scope: Gemma4LiteralCalculationScope,
  operation: string,
): Gemma4LiteralRuntimeReductionOperationClass {
  if (scope === "vision" && operation === "attention-score-matmul") return "vision-attention-score";
  if (scope === "vision" && operation === "attention-value-matmul") return "vision-attention-value";
  if (scope === "audio" && operation === "chunked-attention-content-matmul") return "audio-content-attention-score";
  if (scope === "audio" && operation === "relative-attention-position-matmul") return "audio-position-attention-score";
  if (scope === "audio" && operation === "chunked-relative-attention-values") return "audio-attention-value";
  throw new Error(`Redução runtime-defined inesperada em ${scope}:${operation}; nenhuma semântica BMM pode ser inferida.`);
}
