import { isDeepStrictEqual } from "node:util";
import type { Gemma4LiteralGenerationAssignment } from "./gemma4-composite-literal.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import { listGemma4LiteralOperations } from "./gemma4-literal-multimodal-scalar-view.js";
import type { Gemma4LiteralOperationNavigation } from "./gemma4-literal-scalar-view.js";

export interface Gemma4LiteralGenerationPredecessor {
  input: string;
  /** Empty for declared caller inputs, embedded constants and control values. */
  producerOperationIds: string[];
}

export interface Gemma4LiteralGenerationForwardExpansion {
  mode: "multimodal-prefill" | "cached-incremental";
  sourceCheckpointAccessed: false;
  operationCount: number;
  firstOperationId: string;
  lastOperationId: string;
  inputContract: string;
}

export interface Gemma4LiteralGenerationNavigationPlan {
  kind: "gemma4-literal-generation-navigation-plan";
  sourceCheckpointAccessed: false;
  maxNewTokens: number;
  /** One shared expansion; generation forward entries reference its endpoints. */
  declaredForwardOperations: Gemma4LiteralOperationNavigation[];
  operations: Gemma4LiteralGenerationOperationNavigation[];
}

export interface Gemma4LiteralGenerationOperationNavigation {
  operationId: string;
  definitionId: string;
  operation: Gemma4LiteralGenerationAssignment["operation"];
  ordinal: number;
  step?: number;
  executionCondition: string;
  conditionProducerOperationIds: string[];
  inputs: string[];
  output: string;
  dtype: string;
  shape: string;
  predecessors: Gemma4LiteralGenerationPredecessor[];
  consumers: string[];
  previousOperationId?: string;
  nextOperationId?: string;
  forwardExpansion?: Gemma4LiteralGenerationForwardExpansion;
}

export interface Gemma4LiteralGenerationCalculationView {
  kind: "gemma4-literal-generation-calculation-view";
  sourceCheckpointAccessed: false;
  navigation: Gemma4LiteralGenerationOperationNavigation;
  formula: string;
  scalarAssignments: string[];
  forwardExpansion?: Gemma4LiteralGenerationForwardExpansion;
  declaredForwardOperations?: Gemma4LiteralOperationNavigation[];
}

export interface Gemma4LiteralGenerationCalculationPlan {
  kind: "gemma4-literal-generation-calculation-plan";
  sourceCheckpointAccessed: false;
  maxNewTokens: number;
  /** Concrete control assignments. The complete forward graph is shared once by the enclosing product view. */
  operations: Array<Omit<Gemma4LiteralGenerationCalculationView, "declaredForwardOperations">>;
}

interface InstantiatedAssignment {
  assignment: Gemma4LiteralGenerationAssignment;
  operationId: string;
  step?: number;
  executionCondition: string;
  inputs: string[];
  output: string;
}

/**
 * Instantiates the artifact's symbolic greedy loop into a finite, dependency-
 * ordered audit plan. This is a view of max_new_tokens, not a generation
 * limit: callers choose the exact bound they want to inspect.
 */
export function buildGemma4LiteralGenerationNavigation(
  artifact: OpenGemma4CompositeLiteralArtifact,
  maxNewTokens: number,
): Gemma4LiteralGenerationNavigationPlan {
  validateMaxNewTokens(maxNewTokens);
  const instantiated = instantiateAssignments(artifact, maxNewTokens);
  const producerByOutput = new Map(instantiated.map((entry) => [entry.output, entry.operationId]));
  const forwardStateProducers = instantiated
    .filter((entry) => entry.output.startsWith("forward_state["))
    .map((entry) => entry.operationId);
  const stopProducers = instantiated
    .filter((entry) => entry.output.startsWith("stop_after_step["))
    .map((entry) => entry.operationId);
  const consumers = new Map<string, string[]>();
  const predecessors = instantiated.map((entry) => entry.inputs.map((input) => {
    const producerOperationIds = producerIds(input, producerByOutput, forwardStateProducers, stopProducers);
    for (const producer of producerOperationIds) {
      const existing = consumers.get(producer) ?? [];
      if (!existing.includes(entry.operationId)) existing.push(entry.operationId);
      consumers.set(producer, existing);
    }
    return { input, producerOperationIds };
  }));
  const conditionProducers = instantiated.map((entry) => entry.step === undefined || entry.step === 0
    ? []
    : Array.from({ length: entry.step }, (_, step) => `generation_eos_stop[${step}]`));
  conditionProducers.forEach((producers, ordinal) => {
    for (const producer of producers) {
      const existing = consumers.get(producer) ?? [];
      if (!existing.includes(instantiated[ordinal]!.operationId)) existing.push(instantiated[ordinal]!.operationId);
      consumers.set(producer, existing);
    }
  });
  const declaredForward = listGemma4LiteralOperations(artifact);
  if (declaredForward.length === 0) throw new Error("Programa literal Gemma 4 não possui operações de forward navegáveis.");
  validateSerializedForwardOrder(artifact, declaredForward);
  const operations = instantiated.map((entry, ordinal) => {
    const forwardExpansion = entry.assignment.operation === "execute-declared-forward"
      ? expansion(declaredForward, "multimodal-prefill")
      : entry.assignment.operation === "execute-declared-incremental-forward"
        ? expansion(declaredForward, "cached-incremental")
        : undefined;
    return {
      operationId: entry.operationId,
      definitionId: entry.assignment.id,
      operation: entry.assignment.operation,
      ordinal,
      ...(entry.step === undefined ? {} : { step: entry.step }),
      executionCondition: entry.executionCondition,
      conditionProducerOperationIds: conditionProducers[ordinal]!,
      inputs: [...entry.inputs],
      output: entry.output,
      dtype: entry.assignment.dtype,
      shape: entry.assignment.shape,
      predecessors: predecessors[ordinal]!,
      consumers: consumers.get(entry.operationId) ?? [],
      ...(ordinal === 0 ? {} : { previousOperationId: instantiated[ordinal - 1]!.operationId }),
      ...(ordinal + 1 === instantiated.length ? {} : { nextOperationId: instantiated[ordinal + 1]!.operationId }),
      ...(forwardExpansion ? { forwardExpansion } : {}),
    };
  });
  return {
    kind: "gemma4-literal-generation-navigation-plan",
    sourceCheckpointAccessed: false,
    maxNewTokens,
    declaredForwardOperations: declaredForward,
    operations,
  };
}

/** Renders every generation-control operation as an exact indexed formula. */
export function renderGemma4LiteralGenerationCalculationView(
  artifact: OpenGemma4CompositeLiteralArtifact,
  maxNewTokens: number,
  operationId: string,
): Gemma4LiteralGenerationCalculationView {
  const plan = buildGemma4LiteralGenerationNavigation(artifact, maxNewTokens);
  const navigation = plan.operations
    .find((entry) => entry.operationId === operationId);
  if (!navigation) throw new Error(`Atribuição de geração Gemma 4 literal não encontrada: ${operationId}.`);
  return {
    ...generationCalculationView(artifact, navigation, maxNewTokens),
    ...(navigation.forwardExpansion ? { declaredForwardOperations: plan.declaredForwardOperations } : {}),
  };
}

/** Materializes every greedy-control formula once without duplicating the 2,709-operation forward graph per step. */
export function buildGemma4LiteralGenerationCalculationPlan(
  artifact: OpenGemma4CompositeLiteralArtifact,
  maxNewTokens: number,
): Gemma4LiteralGenerationCalculationPlan {
  const navigation = buildGemma4LiteralGenerationNavigation(artifact, maxNewTokens);
  return {
    kind: "gemma4-literal-generation-calculation-plan",
    sourceCheckpointAccessed: false,
    maxNewTokens,
    operations: navigation.operations.map((entry) => generationCalculationView(artifact, entry, maxNewTokens)),
  };
}

function generationCalculationView(
  artifact: OpenGemma4CompositeLiteralArtifact,
  navigation: Gemma4LiteralGenerationOperationNavigation,
  maxNewTokens: number,
): Omit<Gemma4LiteralGenerationCalculationView, "declaredForwardOperations"> {
  const calculation = artifact.generation.scalarCalculations.assignments
    .find((candidate) => candidate.definitionId === navigation.definitionId);
  if (!calculation || calculation.operation !== navigation.operation) {
    throw new Error(`${navigation.definitionId}: cálculo escalar de geração ausente ou incompatível.`);
  }
  const scalarAssignments = calculation.scalarAssignments.map((formula) => instantiateFormula(formula, navigation.step, maxNewTokens));
  return {
    kind: "gemma4-literal-generation-calculation-view",
    sourceCheckpointAccessed: false,
    navigation,
    formula: scalarAssignments.at(-1)!,
    scalarAssignments,
    ...(navigation.forwardExpansion ? { forwardExpansion: navigation.forwardExpansion } : {}),
  };
}

function instantiateAssignments(
  artifact: OpenGemma4CompositeLiteralArtifact,
  maxNewTokens: number,
): InstantiatedAssignment[] {
  const leading = artifact.generation.assignments.filter((assignment) => assignment.iteration === undefined).slice(0, 2);
  const loop = artifact.generation.assignments.filter((assignment) => assignment.iteration !== undefined);
  const terminal = artifact.generation.assignments.filter((assignment) => assignment.iteration === undefined).slice(2);
  if (leading.length !== 2 || loop.length !== 8 || terminal.length !== 2) {
    throw new Error("Programa literal Gemma 4 não possui a estrutura de geração validada 2+8+2.");
  }
  const result = leading.map((assignment) => instantiate(assignment, undefined, "always"));
  for (let step = 0; step < maxNewTokens; step += 1) {
    const condition = step === 0
      ? "max_new_tokens > 0"
      : `max_new_tokens > ${step} and every stop_after_step[0..${step - 1}] == false`;
    result.push(...loop.map((assignment) => instantiate(assignment, step, condition)));
  }
  result.push(...terminal.map((assignment) => instantiate(assignment, undefined, "after the final executed forward state")));
  return result;
}

function instantiate(
  assignment: Gemma4LiteralGenerationAssignment,
  step: number | undefined,
  executionCondition: string,
): InstantiatedAssignment {
  return {
    assignment,
    operationId: step === undefined ? assignment.id : `${assignment.id}[${step}]`,
    ...(step === undefined ? {} : { step }),
    executionCondition,
    inputs: assignment.inputs.map((input) => instantiateReference(input, step)),
    output: instantiateReference(assignment.output, step),
  };
}

function instantiateReference(value: string, step: number | undefined): string {
  if (step === undefined) return value;
  return value
    .replaceAll("[step+1]", `[${step + 1}]`)
    .replaceAll("[step-1]", `[${step - 1}]`)
    .replaceAll("[0..step-1]", step === 0 ? "[]" : `[0..${step - 1}]`)
    .replaceAll("[0..step]", `[0..${step}]`)
    .replaceAll("[step]", `[${step}]`);
}

function producerIds(
  input: string,
  producerByOutput: ReadonlyMap<string, string>,
  forwardStateProducers: string[],
  stopProducers: string[],
): string[] {
  if (input.startsWith("forward_state[executed_steps]")) return [...forwardStateProducers];
  if (input === "stop_after_step") return [...stopProducers];
  let best: { output: string; operationId: string } | undefined;
  for (const [output, operationId] of producerByOutput) {
    if (input === output || input.startsWith(`${output}.`) || input.startsWith(`${output}[`)) {
      if (!best || output.length > best.output.length) best = { output, operationId };
    }
  }
  return best ? [best.operationId] : [];
}

function expansion(
  operations: Gemma4LiteralOperationNavigation[],
  mode: Gemma4LiteralGenerationForwardExpansion["mode"],
): Gemma4LiteralGenerationForwardExpansion {
  return {
    mode,
    sourceCheckpointAccessed: false,
    operationCount: operations.length,
    firstOperationId: operations[0]!.operationId,
    lastOperationId: operations.at(-1)!.operationId,
    inputContract: mode === "multimodal-prefill"
      ? "input_ids plus supplied paired image/video/audio inputs and optional mm_token_type_ids; no caller cache"
      : "input_ids=[[selected_token]], position_ids=[[position]], exact prior post-RoPE cache; no modality input, mm_token_type_ids or caller attention mask",
  };
}

function validateSerializedForwardOrder(
  artifact: OpenGemma4CompositeLiteralArtifact,
  navigation: readonly Gemma4LiteralOperationNavigation[],
): void {
  const actual = navigation.map((entry) => ({
    operationId: entry.operationId,
    definitionId: entry.definitionId ?? entry.operationId,
    scope: entry.scope,
    ...(entry.invocationId ? { invocationId: entry.invocationId } : {}),
  }));
  if (!isDeepStrictEqual(actual, artifact.generation.forwardCalculation.operationOrder)) {
    const expected = artifact.generation.forwardCalculation.operationOrder;
    const mismatch = Array.from({ length: Math.max(actual.length, expected.length) }, (_, index) => index)
      .find((index) => !isDeepStrictEqual(actual[index], expected[index]));
    throw new Error(`Ordem forward serializada da geração diverge da navegação completa do artefato em ${mismatch ?? "comprimento"}: esperado=${JSON.stringify(expected[mismatch ?? -1])}, atual=${JSON.stringify(actual[mismatch ?? -1])}.`);
  }
}

function instantiateFormula(formula: string, step: number | undefined, maxNewTokens: number): string {
  if (formula.startsWith("executed_steps = max_new_tokens==0")) {
    return maxNewTokens === 0
      ? "executed_steps = 0 because max_new_tokens = 0"
      : `executed_steps = first s in 0..${maxNewTokens - 1} with stop_after_step[s] == true, mapped to s+1; otherwise ${maxNewTokens}`;
  }
  let instantiated = formula.replaceAll("max_new_tokens", String(maxNewTokens));
  if (step === undefined) return instantiated;
  instantiated = instantiated
    .replaceAll("[step+1]", `[${step + 1}]`)
    .replaceAll("[step-1]", `[${step - 1}]`)
    .replaceAll("[0..step-1]", step === 0 ? "[]" : `[0..${step - 1}]`)
    .replaceAll("[0..step]", `[0..${step}]`)
    .replaceAll("[step]", `[${step}]`)
    .replaceAll("step==0", `${step}==0`);
  return instantiated;
}

function validateMaxNewTokens(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("Navegação de geração Gemma 4 requer maxNewTokens inteiro não negativo.");
  const operations = 4 + value * 8;
  if (!Number.isSafeInteger(operations) || operations > 1_000_000) {
    throw new Error("Navegação de geração Gemma 4 excede um milhão de operações de controle materializadas.");
  }
}
