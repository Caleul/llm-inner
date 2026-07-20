import { isDeepStrictEqual } from "node:util";
import type { Gemma4AudioProgram } from "./gemma4-audio.js";
import type { Gemma4VisionProgram } from "./gemma4-vision.js";
import type { DifferentialOperationSample } from "./types.js";

type RuntimeReductionProgram = Gemma4VisionProgram | Gemma4AudioProgram;
interface RuntimeReductionAssignment {
  id: string;
  operation: string;
  inputs: string[];
  output: string;
  dtypePolicy?: { accumulationDtype?: string };
}

export interface Gemma4RuntimeReductionTraceOperand {
  input: string;
  producerOperationId: string;
  shape: number[];
}

export interface Gemma4RuntimeReductionTraceEntry {
  operationId: string;
  operation: string;
  output: string;
  outputShape: number[];
  orderedOperands: Gemma4RuntimeReductionTraceOperand[];
}

/**
 * Machine-checkable proof that a native-reduction trace contains the exact
 * tensors on both sides of every still-opaque BMM boundary. It does not infer
 * an accumulation schedule; it prevents probes from rebuilding post-RoPE or
 * blocked operands from an earlier, mathematically different checkpoint.
 */
export interface Gemma4RuntimeReductionTraceCoverage {
  kind: "gemma4-runtime-reduction-trace-coverage";
  schemaVersion: 1;
  entries: Gemma4RuntimeReductionTraceEntry[];
}

/** Exact artifact-declared execution order for every opaque native reduction. */
export function gemma4RuntimeReductionOperationIds(program: RuntimeReductionProgram): string[] {
  const operationIds = program.assignments
    .filter((assignment) => assignment.dtypePolicy?.accumulationDtype === "runtime-defined")
    .map((assignment) => assignment.id);
  if (operationIds.length === 0 || operationIds.some((operationId) => !operationId) ||
    new Set(operationIds).size !== operationIds.length) {
    throw new Error(`${program.kind}: ordem das reduções runtime-defined está vazia ou possui operationId inválido/duplicado.`);
  }
  return operationIds;
}

export function buildGemma4RuntimeReductionTraceCoverage(
  program: RuntimeReductionProgram,
  operations: readonly DifferentialOperationSample[],
): Gemma4RuntimeReductionTraceCoverage {
  const samplesByOperation = uniqueIndex(operations, (sample) => sample.operationId, "operationId do trace");
  const assignments: readonly RuntimeReductionAssignment[] = program.assignments;
  const producersByOutput = uniqueIndex(assignments, (assignment) => assignment.output, "output do programa");
  const entries = assignments.flatMap((assignment): Gemma4RuntimeReductionTraceEntry[] => {
    if (assignment.dtypePolicy?.accumulationDtype !== "runtime-defined") return [];
    if (assignment.inputs.length !== 2) throw new Error(`${assignment.id}: redução runtime-defined requer exatamente dois operandos ordenados.`);
    const outputSample = samplesByOperation.get(assignment.id);
    if (!outputSample || outputSample.output !== assignment.output) {
      throw new Error(`${assignment.id}: trace não contém a saída nativa ${assignment.output}.`);
    }
    const orderedOperands = assignment.inputs.map((input): Gemma4RuntimeReductionTraceOperand => {
      const producer = producersByOutput.get(input);
      if (!producer) throw new Error(`${assignment.id}: produtor de ${input} ausente do programa.`);
      const sample = samplesByOperation.get(producer.id);
      if (!sample || sample.output !== input) throw new Error(`${assignment.id}: trace não contém o operando nativo ${input} de ${producer.id}.`);
      return { input, producerOperationId: producer.id, shape: validShape(sample.tensor.shape, `${producer.id}:${input}`) };
    });
    return [{
      operationId: assignment.id,
      operation: assignment.operation,
      output: assignment.output,
      outputShape: validShape(outputSample.tensor.shape, `${assignment.id}:${assignment.output}`),
      orderedOperands,
    }];
  });
  const expectedOrder = gemma4RuntimeReductionOperationIds(program);
  if (!isDeepStrictEqual(entries.map((entry) => entry.operationId), expectedOrder)) {
    throw new Error(`${program.kind}: cobertura não preserva a ordem declarada das reduções runtime-defined.`);
  }
  return { kind: "gemma4-runtime-reduction-trace-coverage", schemaVersion: 1, entries };
}

export function validateGemma4RuntimeReductionTraceCoverage(
  coverage: Gemma4RuntimeReductionTraceCoverage,
  program: RuntimeReductionProgram,
  operations: readonly DifferentialOperationSample[],
): void {
  if (!isDeepStrictEqual(coverage, buildGemma4RuntimeReductionTraceCoverage(program, operations))) {
    throw new Error("Trace Gemma 4 possui cobertura de operandos BMM incompleta ou divergente.");
  }
}

function uniqueIndex<T>(values: readonly T[], key: (value: T) => string, label: string): Map<string, T> {
  const result = new Map<string, T>();
  for (const value of values) {
    const id = key(value);
    if (!id || result.has(id)) throw new Error(`${label} ausente ou duplicado: ${id}.`);
    result.set(id, value);
  }
  return result;
}

function validShape(shape: readonly number[], owner: string): number[] {
  if (shape.length === 0 || shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)) {
    throw new Error(`${owner}: shape de trace inválido.`);
  }
  return [...shape];
}
