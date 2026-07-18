import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import type { Gemma4LiteralNumericLiteral, Gemma4LiteralNumericLiteralUse } from "./gemma4-literal-numeric-literals.js";
import type { Gemma4LiteralLearnedOperandRole } from "./gemma4-literal-learned-operands.js";
import { listGemma4LiteralOperations } from "./gemma4-literal-multimodal-scalar-view.js";
import type { Gemma4LiteralOperationNavigation } from "./gemma4-literal-scalar-view.js";
import type { LiteralDenseStorageDecodeAssignment } from "./literal.js";
import type { TensorRef } from "./types.js";

export interface Gemma4LiteralCalculationSliceExternalInput {
  name: string;
  consumerOperationIds: string[];
}

export interface Gemma4LiteralCalculationSliceLearnedConsumer {
  operationId: string;
  role: Gemma4LiteralLearnedOperandRole;
  logicalIndices: string[];
}

export interface Gemma4LiteralCalculationSliceLearnedConstant {
  tensor: TensorRef;
  decoderId: string;
  decoderOperation: LiteralDenseStorageDecodeAssignment["operation"];
  consumers: Gemma4LiteralCalculationSliceLearnedConsumer[];
}

export interface Gemma4LiteralCalculationSliceNumericLiteral extends Omit<Gemma4LiteralNumericLiteral, "uses"> {
  uses: Gemma4LiteralNumericLiteralUse[];
}

/**
 * A target-rooted, source-independent subset of the artifact's canonical
 * calculation graph. Every producer needed by the target is retained in its
 * original dependency order; unrelated branches are omitted without changing
 * any formula, domain, reduction bound or storage declaration.
 */
export interface Gemma4LiteralCalculationSlice {
  kind: "gemma4-literal-calculation-slice";
  sourceCheckpointAccessed: false;
  targetOperationId: string;
  targetOutput: string;
  operationCount: number;
  firstOperationId: string;
  lastOperationId: string;
  operations: Gemma4LiteralOperationNavigation[];
  externalInputs: Gemma4LiteralCalculationSliceExternalInput[];
  learnedConstants: Gemma4LiteralCalculationSliceLearnedConstant[];
  numericLiterals: Gemma4LiteralCalculationSliceNumericLiteral[];
  reproducibility: {
    status: "literal" | "fail-closed-runtime-reduction";
    literalOperationCount: number;
    failClosedOperationIds: string[];
  };
}

/**
 * Resolves the complete transitive producer closure for any instantiated
 * forward assignment serialized in the literal artifact.
 */
export function buildGemma4LiteralCalculationSlice(
  artifact: OpenGemma4CompositeLiteralArtifact,
  targetOperationId: string,
): Gemma4LiteralCalculationSlice {
  const navigation = listGemma4LiteralOperations(artifact);
  const byId = new Map(navigation.map((operation) => [operation.operationId, operation]));
  const target = byId.get(targetOperationId);
  if (!target) throw new Error(`Atribuição Gemma 4 literal não encontrada para slice: ${targetOperationId}.`);

  const selectedIds = new Set<string>();
  const visit = (operation: Gemma4LiteralOperationNavigation): void => {
    if (selectedIds.has(operation.operationId)) return;
    selectedIds.add(operation.operationId);
    for (const predecessor of operation.predecessors) {
      if (!predecessor.producerOperationId) continue;
      const producer = byId.get(predecessor.producerOperationId);
      if (!producer) {
        throw new Error(`${operation.operationId}: slice referencia produtor ausente ${predecessor.producerOperationId}.`);
      }
      if (producer.ordinal >= operation.ordinal) {
        throw new Error(`${operation.operationId}: slice referencia produtor não anterior ${producer.operationId}.`);
      }
      visit(producer);
    }
  };
  visit(target);

  const operations = navigation.filter((operation) => selectedIds.has(operation.operationId));
  const operationIdsByDefinition = new Map<string, string[]>();
  for (const operation of operations) {
    const key = definitionKey(operation.scope, operation.definitionId ?? operation.operationId);
    const ids = operationIdsByDefinition.get(key) ?? [];
    ids.push(operation.operationId);
    operationIdsByDefinition.set(key, ids);
  }

  const externalConsumers = new Map<string, string[]>();
  for (const operation of operations) for (const predecessor of operation.predecessors) {
    if (predecessor.producerOperationId) continue;
    const consumers = externalConsumers.get(predecessor.input) ?? [];
    if (!consumers.includes(operation.operationId)) consumers.push(operation.operationId);
    externalConsumers.set(predecessor.input, consumers);
  }
  const externalInputs = [...externalConsumers.entries()]
    .map(([name, consumerOperationIds]) => ({ name, consumerOperationIds }))
    .sort((left, right) => left.name.localeCompare(right.name, "en"));

  const decoderById = new Map(artifact.storageDecoders.map((decoder) => [decoder.id, decoder]));
  const learnedByTensor = new Map<string, Gemma4LiteralCalculationSliceLearnedConstant>();
  for (const operation of operations) for (const operand of operation.learnedOperands ?? []) {
    const decoder = decoderById.get(operand.decoderId);
    if (!decoder) throw new Error(`${operation.operationId}: decoder aprendido ausente ${operand.decoderId}.`);
    const existing = learnedByTensor.get(operand.tensor.name);
    if (existing && (existing.decoderId !== operand.decoderId || existing.decoderOperation !== decoder.operation)) {
      throw new Error(`${operand.tensor.name}: slice encontrou decoders aprendidos contraditórios.`);
    }
    const constant = existing ?? {
      tensor: structuredClone(operand.tensor),
      decoderId: operand.decoderId,
      decoderOperation: decoder.operation,
      consumers: [],
    };
    constant.consumers.push({
      operationId: operation.operationId,
      role: operand.role,
      logicalIndices: [...operand.logicalIndices],
    });
    learnedByTensor.set(operand.tensor.name, constant);
  }
  const learnedConstants = [...learnedByTensor.values()]
    .sort((left, right) => left.tensor.name.localeCompare(right.tensor.name, "en"));

  const numericLiterals = artifact.numericLiterals.literals.flatMap((literal): Gemma4LiteralCalculationSliceNumericLiteral[] => {
    const uses = literal.uses.filter((use) => use.section === "forward" &&
      operationIdsByDefinition.has(definitionKey(use.scope ?? "", use.definitionId)));
    return uses.length === 0 ? [] : [{ ...structuredClone(literal), uses }];
  });
  const failClosedOperationIds = operations
    .filter((operation) => operation.scalarCalculation.reproducibility === "fail-closed-runtime-reduction")
    .map((operation) => operation.operationId);

  return {
    kind: "gemma4-literal-calculation-slice",
    sourceCheckpointAccessed: false,
    targetOperationId,
    targetOutput: target.output,
    operationCount: operations.length,
    firstOperationId: operations[0]!.operationId,
    lastOperationId: target.operationId,
    operations,
    externalInputs,
    learnedConstants,
    numericLiterals,
    reproducibility: {
      status: failClosedOperationIds.length === 0 ? "literal" : "fail-closed-runtime-reduction",
      literalOperationCount: operations.length - failClosedOperationIds.length,
      failClosedOperationIds,
    },
  };
}

function definitionKey(scope: string, definitionId: string): string {
  return `${scope}\u0000${definitionId}`;
}
