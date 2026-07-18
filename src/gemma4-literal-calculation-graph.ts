import { isDeepStrictEqual } from "node:util";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeAssignment, Gemma4CompositeProgram } from "./gemma4-composite.js";
import {
  buildGemma4LiteralCalculationDomains,
  instantiateGemma4LiteralValueDomain,
  type Gemma4LiteralCalculationScope,
  type Gemma4LiteralValueDomain,
} from "./gemma4-literal-domains.js";
import {
  buildGemma4LiteralLearnedOperandBindings,
  type Gemma4LiteralLearnedOperand,
} from "./gemma4-literal-learned-operands.js";
import {
  buildGemma4LiteralScalarCalculations,
  type Gemma4LiteralScalarCalculation,
} from "./gemma4-literal-scalar-calculations.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { Operation } from "./types.js";

type NonTextAssignment = Gemma4CompositeAssignment | Gemma4VisionAssignment | Gemma4AudioAssignment;

export interface Gemma4LiteralCalculationPredecessor {
  input: string;
  producerOperationId?: string;
}

/**
 * One fully instantiated assignment in the actual composite forward. Shared
 * tower definitions are bound at each call site here, not by reader behavior.
 */
export interface Gemma4LiteralInstantiatedCalculation {
  ordinal: number;
  operationId: string;
  definitionId: string;
  operation: string;
  scope: Exclude<Gemma4LiteralCalculationScope, "text-prelude">;
  layer?: number;
  invocationId?: string;
  orderedInputs: string[];
  output: string;
  outputDomain: Gemma4LiteralValueDomain;
  scalarCalculation: Gemma4LiteralScalarCalculation;
  learnedOperands?: Gemma4LiteralLearnedOperand[];
  predecessors: Gemma4LiteralCalculationPredecessor[];
  consumers: string[];
}

export interface Gemma4LiteralCalculationGraph {
  kind: "gemma4-literal-instantiated-calculation-graph";
  schemaVersion: 1;
  order: "dependency-order";
  assignments: Gemma4LiteralInstantiatedCalculation[];
}

interface Seed {
  operationId: string;
  definitionId: string;
  operation: string;
  scope: Gemma4LiteralInstantiatedCalculation["scope"];
  layer?: number;
  invocationId?: string;
  inputs: string[];
  output: string;
  outputDomain: Gemma4LiteralValueDomain;
  scalarCalculation: Gemma4LiteralScalarCalculation;
  learnedOperands?: Gemma4LiteralLearnedOperand[];
}

interface NonTextDefinition {
  assignment: NonTextAssignment;
  scope: "composite" | "vision" | "audio";
  operationId: string;
  invocationId?: string;
  bindings: ReadonlyMap<string, string>;
}

/** Builds the complete call-site-bound forward graph stored in the artifact. */
export function buildGemma4LiteralCalculationGraph(program: Gemma4CompositeProgram): Gemma4LiteralCalculationGraph {
  const domains = buildGemma4LiteralCalculationDomains(program);
  const calculations = buildGemma4LiteralScalarCalculations(program);
  const learned = buildGemma4LiteralLearnedOperandBindings(program);
  const seeds = calculationSeeds(program, domains, calculations, learned);
  const ids = new Set<string>();
  const producerByOutput = new Map<string, string>();
  for (const seed of seeds) {
    if (ids.has(seed.operationId)) throw new Error(`Grafo literal Gemma 4 repete operação instanciada ${seed.operationId}.`);
    if (producerByOutput.has(seed.output)) throw new Error(`Grafo literal Gemma 4 redeclara saída instanciada ${seed.output}.`);
    ids.add(seed.operationId);
    producerByOutput.set(seed.output, seed.operationId);
  }
  const consumersByOutput = new Map<string, string[]>();
  for (const seed of seeds) for (const input of seed.inputs) {
    const consumers = consumersByOutput.get(input) ?? [];
    consumers.push(seed.operationId);
    consumersByOutput.set(input, consumers);
  }
  const assignments = seeds.map((seed, ordinal): Gemma4LiteralInstantiatedCalculation => ({
    ordinal,
    operationId: seed.operationId,
    definitionId: seed.definitionId,
    operation: seed.operation,
    scope: seed.scope,
    ...(seed.layer === undefined ? {} : { layer: seed.layer }),
    ...(seed.invocationId ? { invocationId: seed.invocationId } : {}),
    orderedInputs: [...seed.inputs],
    output: seed.output,
    outputDomain: structuredClone(seed.outputDomain),
    scalarCalculation: structuredClone(seed.scalarCalculation),
    ...(seed.learnedOperands ? { learnedOperands: structuredClone(seed.learnedOperands) } : {}),
    predecessors: seed.inputs.map((input) => ({
      input,
      ...(producerByOutput.has(input) ? { producerOperationId: producerByOutput.get(input)! } : {}),
    })),
    consumers: [...(consumersByOutput.get(seed.output) ?? [])],
  }));
  for (const assignment of assignments) for (const predecessor of assignment.predecessors) {
    if (!predecessor.producerOperationId) continue;
    const producer = assignments.find((candidate) => candidate.operationId === predecessor.producerOperationId)!;
    if (producer.ordinal >= assignment.ordinal) {
      throw new Error(`${assignment.operationId}: predecessor ${producer.operationId} não antecede o consumidor no grafo literal.`);
    }
  }
  return { kind: "gemma4-literal-instantiated-calculation-graph", schemaVersion: 1, order: "dependency-order", assignments };
}

export function validateGemma4LiteralCalculationGraph(
  graph: Gemma4LiteralCalculationGraph,
  program: Gemma4CompositeProgram,
): void {
  if (graph.kind !== "gemma4-literal-instantiated-calculation-graph" || graph.schemaVersion !== 1 || graph.order !== "dependency-order") {
    throw new Error("Programa literal Gemma 4 possui cabeçalho de grafo de cálculo inválido.");
  }
  if (!isDeepStrictEqual(graph, buildGemma4LiteralCalculationGraph(program))) {
    throw new Error("Programa literal Gemma 4 possui grafo instanciado, bindings ou dependências ausentes ou divergentes.");
  }
}

/**
 * Binds source definition names without recursively rewriting overlapping
 * names such as `*_bd` and `*_bd_unshifted`.
 */
export function bindGemma4LiteralNames(value: string, bindings: ReadonlyMap<string, string>): string {
  const ordered = [...bindings.entries()].sort(([left], [right]) => right.length - left.length);
  const placeholders = ordered.map(([source, target], index) => ({ source, target, token: `\u0000binding:${index}\u0000` }));
  const tokenized = placeholders.reduce((current, binding) => current.split(binding.source).join(binding.token), value);
  return placeholders.reduce((current, binding) => current.split(binding.token).join(binding.target), tokenized);
}

function calculationSeeds(
  program: Gemma4CompositeProgram,
  domains: ReturnType<typeof buildGemma4LiteralCalculationDomains>,
  calculations: ReturnType<typeof buildGemma4LiteralScalarCalculations>,
  learned: ReturnType<typeof buildGemma4LiteralLearnedOperandBindings>,
): Seed[] {
  const seeds: Seed[] = [];
  for (const assignment of program.assignments) {
    if (assignment.operation === "vision-feature-program") {
      const definitions = program.visionProgram.assignments.filter((entry) => entry.operation !== "masked-scatter-image-features");
      seeds.push(...instantiateNonTextDefinitions(definitions, "vision", assignment, program.visionProgram.output)
        .map((entry) => seedFromNonText(entry, domains, calculations, learned)));
      continue;
    }
    if (assignment.operation === "audio-feature-program") {
      const definitions = program.audioProgram.assignments.filter((entry) => entry.operation !== "masked-scatter-audio-features");
      seeds.push(...instantiateNonTextDefinitions(definitions, "audio", assignment, program.audioProgram.output)
        .map((entry) => seedFromNonText(entry, domains, calculations, learned)));
      continue;
    }
    if (assignment.operation === "text-core") {
      const bindings = new Map([
        ["attention_mask:full_attention", "full_attention_mask"],
        ["attention_mask:sliding_attention", "sliding_attention_mask"],
      ]);
      for (const layer of program.textProgram.layers) for (const operation of layer.operations) {
        seeds.push(seedFromText(operation, "text-layer", assignment.id, bindings, domains, calculations, learned));
      }
      for (const operation of program.textProgram.epilogue) {
        seeds.push(seedFromText(operation, "text-epilogue", assignment.id, bindings, domains, calculations, learned));
      }
      continue;
    }
    seeds.push(seedFromNonText({ assignment, scope: "composite", operationId: assignment.id, bindings: new Map() }, domains, calculations, learned));
  }
  if (seeds.length === 0) throw new Error("Programa Gemma 4 não possui cálculos instanciados.");
  return seeds;
}

function instantiateNonTextDefinitions(
  definitions: NonTextAssignment[],
  scope: "vision" | "audio",
  invocation: Gemma4CompositeAssignment,
  terminalOutput: string,
): NonTextDefinition[] {
  const bindings = scope === "vision"
    ? new Map([["pixel_values", invocation.inputs[0]!], ["pixel_position_ids", invocation.inputs[1]!]])
    : new Map([["input_features", invocation.inputs[0]!], ["input_features_mask", invocation.inputs[1]!]]);
  for (const definition of definitions) {
    bindings.set(definition.output, definition.output === terminalOutput ? invocation.output : `${invocation.id}/${definition.output}`);
  }
  return definitions.map((assignment) => ({
    assignment,
    scope,
    operationId: `${invocation.id}/${assignment.id}`,
    invocationId: invocation.id,
    bindings,
  }));
}

function seedFromNonText(
  definition: NonTextDefinition,
  domains: ReturnType<typeof buildGemma4LiteralCalculationDomains>,
  calculations: ReturnType<typeof buildGemma4LiteralScalarCalculations>,
  learned: ReturnType<typeof buildGemma4LiteralLearnedOperandBindings>,
): Seed {
  const { assignment, bindings } = definition;
  const scalarCalculation = requiredCalculation(calculations, definition.scope, assignment.id);
  const boundInputs = assignment.inputs.map((input) => bindings.get(input) ?? input);
  const boundOutput = bindings.get(assignment.output) ?? assignment.output;
  const boundCalculation = bindCalculation(scalarCalculation, bindings, boundInputs, boundOutput);
  const domain = domains.assignments.find((candidate) => candidate.scope === definition.scope && candidate.definitionId === assignment.id);
  if (!domain) throw new Error(`${definition.scope}:${assignment.id}: domínio ausente no grafo literal.`);
  const operands = learned.assignments.find((candidate) => candidate.scope === definition.scope && candidate.definitionId === assignment.id)?.operands;
  return {
    operationId: definition.operationId,
    definitionId: assignment.id,
    operation: assignment.operation,
    scope: definition.scope,
    ...(definition.invocationId ? { invocationId: definition.invocationId } : {}),
    inputs: boundInputs,
    output: boundOutput,
    outputDomain: instantiateGemma4LiteralValueDomain(domain.domain, definition.invocationId),
    scalarCalculation: boundCalculation,
    ...(operands ? { learnedOperands: structuredClone(operands) } : {}),
  };
}

function seedFromText(
  operation: Operation,
  scope: "text-layer" | "text-epilogue",
  invocationId: string,
  bindings: ReadonlyMap<string, string>,
  domains: ReturnType<typeof buildGemma4LiteralCalculationDomains>,
  calculations: ReturnType<typeof buildGemma4LiteralScalarCalculations>,
  learned: ReturnType<typeof buildGemma4LiteralLearnedOperandBindings>,
): Seed {
  const inputs = textInputs(operation).map((input) => bindings.get(input) ?? input);
  const output = bindings.get(operation.output) ?? operation.output;
  const domain = domains.assignments.find((candidate) => candidate.scope === scope && candidate.definitionId === operation.id);
  if (!domain) throw new Error(`${scope}:${operation.id}: domínio ausente no grafo literal.`);
  const calculation = bindCalculation(requiredCalculation(calculations, scope, operation.id), bindings, inputs, output);
  const operands = learned.assignments.find((candidate) => candidate.scope === scope && candidate.definitionId === operation.id)?.operands;
  return {
    operationId: operation.id,
    definitionId: operation.id,
    operation: operation.op,
    scope,
    ...(operation.layer === undefined ? {} : { layer: operation.layer }),
    invocationId,
    inputs,
    output,
    outputDomain: structuredClone(domain.domain),
    scalarCalculation: calculation,
    ...(operands ? { learnedOperands: structuredClone(operands) } : {}),
  };
}

function bindCalculation(
  calculation: Gemma4LiteralScalarCalculation,
  bindings: ReadonlyMap<string, string>,
  orderedInputs: string[],
  output: string,
): Gemma4LiteralScalarCalculation {
  return {
    ...structuredClone(calculation),
    orderedInputs,
    output,
    formula: bindGemma4LiteralNames(calculation.formula, bindings),
  };
}

function requiredCalculation(
  calculations: ReturnType<typeof buildGemma4LiteralScalarCalculations>,
  scope: Gemma4LiteralCalculationScope,
  definitionId: string,
): Gemma4LiteralScalarCalculation {
  const calculation = calculations.assignments.find((candidate) => candidate.scope === scope && candidate.definitionId === definitionId);
  if (!calculation) throw new Error(`${scope}:${definitionId}: cálculo escalar ausente no grafo literal.`);
  return calculation;
}

function textInputs(operation: Operation): string[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.tokenInput];
    case "rms_norm": case "linear": case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": case "tensor_scale": return [operation.input];
    case "rotary_embedding": return [operation.input, operation.positionInput];
    case "scaled_dot_product_attention": return [operation.query, operation.key, operation.value, operation.maskInput];
    case "elementwise": return [...operation.inputs];
  }
}
