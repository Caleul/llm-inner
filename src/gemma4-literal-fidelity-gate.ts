import { isDeepStrictEqual } from "node:util";
import type { Gemma4AuthoritativeExecutionContract } from "./gemma4-authoritative-runtime.js";
import type {
  Gemma4LiteralCalculationGraph,
  Gemma4LiteralInstantiatedCalculation,
} from "./gemma4-literal-calculation-graph.js";
import {
  gemma4RuntimeReductionInvocationProgram,
  type Gemma4LiteralRuntimeReductionOperationClass,
} from "./gemma4-runtime-reduction-invocation.js";

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
  invocationProgramId: Gemma4LiteralRuntimeReductionOperationClass;
  invocationProgramPointer: string;
  runtimeEnvironmentIdentityPointer: "/authoritativeExecution/unresolvedNativeReduction/executableReplay/runtimeEnvironmentIdentity";
}

/**
 * Machine-readable completion boundary derived from the canonical graph.
 * It never certifies an implementation session: an empty unresolved list only
 * makes the artifact eligible for independent source-removed/differential
 * certification.
 */
export interface Gemma4LiteralFidelityGate {
  kind: "gemma4-literal-fidelity-gate";
  schemaVersion: 2;
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
  const invocationPrograms = authority.unresolvedNativeReduction.executableReplay.invocationPrograms;
  const unresolvedNativeReductions = graph.assignments.flatMap((assignment): Gemma4LiteralUnresolvedRuntimeReduction[] => {
    const reduction = assignment.scalarCalculation.reduction;
    if (reduction?.order !== "runtime-defined") return [];
    if (assignment.scalarCalculation.reproducibility !== "fail-closed-runtime-reduction" ||
      assignment.scalarCalculation.dtypePolicy.accumulationDtype !== "runtime-defined" ||
      assignment.outputDomain.dtypePolicy?.accumulationDtype !== "runtime-defined" || reduction.domains.length !== 1) {
      throw new Error(`${assignment.operationId}: redução runtime-defined não está integralmente marcada como fail-closed.`);
    }
    const invocationProgram = gemma4RuntimeReductionInvocationProgram(invocationPrograms, assignment.scope, assignment.operation);
    const operationClass = invocationProgram.id;
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
      invocationProgramId: invocationProgram.id,
      invocationProgramPointer: `/authoritativeExecution/unresolvedNativeReduction/executableReplay/invocationPrograms/${invocationPrograms.indexOf(invocationProgram)}`,
      runtimeEnvironmentIdentityPointer: "/authoritativeExecution/unresolvedNativeReduction/executableReplay/runtimeEnvironmentIdentity",
    }];
  });
  const blocked = unresolvedNativeReductions.length > 0;
  return {
    kind: "gemma4-literal-fidelity-gate",
    schemaVersion: 2,
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
