import { isDeepStrictEqual } from "node:util";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { Operation, TensorRef } from "./types.js";

export type Gemma4LiteralLearnedOperandScope =
  | "composite"
  | "vision"
  | "audio"
  | "text-prelude"
  | "text-layer"
  | "text-epilogue";

export type Gemma4LiteralLearnedOperandRole =
  | "weight"
  | "bias"
  | "input-min"
  | "input-max"
  | "output-min"
  | "output-max"
  | "position-table"
  | "convolution-kernel"
  | "normalization-scale"
  | "per-dimension-scale"
  | "tensor-scale";

export interface Gemma4LiteralLearnedOperand {
  role: Gemma4LiteralLearnedOperandRole;
  tensor: TensorRef;
  /**
   * Expressions are evaluated against the scalar view's named output
   * coordinates. They bind a logical tensor address before the storage
   * decoder performs row-major index-to-byte mapping.
   */
  logicalIndices: string[];
  decoderId: string;
}

export interface Gemma4LiteralLearnedOperandBinding {
  scope: Gemma4LiteralLearnedOperandScope;
  definitionId: string;
  operands: Gemma4LiteralLearnedOperand[];
}

/**
 * Explicit bridge from semantic assignments to checkpoint-resident scalars.
 * Scalar readers must use these roles rather than rediscovering weight, bias,
 * clipping or normalization semantics from tensor names and shapes.
 */
export interface Gemma4LiteralLearnedOperandBindings {
  schemaVersion: 1;
  indexLanguage: "named-output-coordinate-expressions-v1";
  assignments: Gemma4LiteralLearnedOperandBinding[];
}

type MultimodalAssignment = Gemma4CompositeProgram["assignments"][number] | Gemma4VisionAssignment | Gemma4AudioAssignment;

export function buildGemma4LiteralLearnedOperandBindings(program: Gemma4CompositeProgram): Gemma4LiteralLearnedOperandBindings {
  const assignments: Gemma4LiteralLearnedOperandBinding[] = [];
  addMultimodal(assignments, "composite", program.assignments);
  addMultimodal(assignments, "vision", program.visionProgram.assignments);
  addMultimodal(assignments, "audio", program.audioProgram.assignments);
  addText(assignments, "text-prelude", program.textProgram.prelude);
  for (const layer of program.textProgram.layers) addText(assignments, "text-layer", layer.operations);
  addText(assignments, "text-epilogue", program.textProgram.epilogue);
  return {
    schemaVersion: 1,
    indexLanguage: "named-output-coordinate-expressions-v1",
    assignments,
  };
}

export function validateGemma4LiteralLearnedOperandBindings(
  bindings: Gemma4LiteralLearnedOperandBindings,
  program: Gemma4CompositeProgram,
): void {
  const expected = buildGemma4LiteralLearnedOperandBindings(program);
  if (!isDeepStrictEqual(bindings, expected)) {
    throw new Error("Programa literal Gemma 4 possui papéis ou índices de operandos aprendidos incompletos.");
  }
}

export function requiredGemma4LiteralLearnedOperand(
  bindings: Gemma4LiteralLearnedOperandBindings,
  scope: Gemma4LiteralLearnedOperandScope,
  definitionId: string,
  role: Gemma4LiteralLearnedOperandRole,
): Gemma4LiteralLearnedOperand {
  const assignment = bindings.assignments.find((candidate) =>
    candidate.scope === scope && candidate.definitionId === definitionId);
  const operand = assignment?.operands.find((candidate) => candidate.role === role);
  if (!operand) throw new Error(`${scope}:${definitionId}: operando aprendido ${role} não foi declarado.`);
  return structuredClone(operand);
}

export function optionalGemma4LiteralLearnedOperand(
  bindings: Gemma4LiteralLearnedOperandBindings,
  scope: Gemma4LiteralLearnedOperandScope,
  definitionId: string,
  role: Gemma4LiteralLearnedOperandRole,
): Gemma4LiteralLearnedOperand | undefined {
  const assignment = bindings.assignments.find((candidate) =>
    candidate.scope === scope && candidate.definitionId === definitionId);
  const operand = assignment?.operands.find((candidate) => candidate.role === role);
  return operand ? structuredClone(operand) : undefined;
}

function addMultimodal(
  output: Gemma4LiteralLearnedOperandBinding[],
  scope: "composite" | "vision" | "audio",
  assignments: readonly MultimodalAssignment[],
): void {
  for (const assignment of assignments) {
    const references = assignment.tensors ?? [];
    if (references.length === 0) continue;
    const operands = multimodalOperands(assignment, references);
    validateUniqueRoles(scope, assignment.id, operands);
    output.push({ scope, definitionId: assignment.id, operands });
  }
}

function multimodalOperands(
  assignment: MultimodalAssignment,
  references: readonly TensorRef[],
): Gemma4LiteralLearnedOperand[] {
  switch (assignment.operation) {
    case "linear":
      requireCount(assignment.id, references, 1, 2);
      return [operand("weight", references[0]!, ["output_feature", "input_feature"]),
        ...(references[1] ? [operand("bias", references[1], ["output_feature"])] : [])];
    case "clipped-linear":
      requireCount(assignment.id, references, 5);
      return [
        operand("weight", references[0]!, ["output_feature", "input_feature"]),
        operand("input-min", references[1]!, scalarLogicalIndices(references[1]!)),
        operand("input-max", references[2]!, scalarLogicalIndices(references[2]!)),
        operand("output-min", references[3]!, scalarLogicalIndices(references[3]!)),
        operand("output-max", references[4]!, scalarLogicalIndices(references[4]!)),
      ];
    case "embedding":
      requireCount(assignment.id, references, 1);
      return [operand("weight", references[0]!, ["token_id", "feature"])];
    case "per-layer-embedding":
      requireCount(assignment.id, references, 1);
      return [operand("weight", references[0]!, ["token_id", "layer*per_layer_width+feature"])];
    case "rms-norm":
      requireCount(assignment.id, references, 1);
      return [operand("normalization-scale", references[0]!, ["feature"])];
    case "position-embedding-2d":
      requireCount(assignment.id, references, 1);
      return [operand("position-table", references[0]!, ["axis", "position", "feature"])];
    case "conv2d-stride2":
      requireCount(assignment.id, references, 1);
      return [operand("convolution-kernel", references[0]!, ["output_channel", "input_channel", "kernel_time", "kernel_feature"])];
    case "layer-norm-channels":
      requireCount(assignment.id, references, 1);
      return [operand("normalization-scale", references[0]!, ["channel"])];
    case "causal-depthwise-convolution":
      requireCount(assignment.id, references, 1);
      return [operand("convolution-kernel", references[0]!, ["channel", "0", "kernel_index"])];
    case "per-dim-softplus-scale":
      requireCount(assignment.id, references, 1);
      return [operand("per-dimension-scale", references[0]!, ["feature%head_dim"])];
    default:
      throw new Error(`${assignment.id}: operação ${assignment.operation} usa tensores sem contrato explícito de papel aprendido.`);
  }
}

function addText(
  output: Gemma4LiteralLearnedOperandBinding[],
  scope: "text-prelude" | "text-layer" | "text-epilogue",
  operations: readonly Operation[],
): void {
  for (const operation of operations) {
    let operands: Gemma4LiteralLearnedOperand[] = [];
    switch (operation.op) {
      case "embedding":
        operands = [operand("weight", operation.weight, ["token_id", "feature"])];
        break;
      case "per_layer_embedding":
        operands = [operand("weight", operation.weight, ["token_id", "layer*per_layer_width+feature"])];
        break;
      case "rms_norm":
        operands = operation.weight ? [operand("normalization-scale", operation.weight, ["feature"])] : [];
        break;
      case "linear":
        operands = [operand("weight", operation.weight, ["output_feature", "input_feature"]),
          ...(operation.bias ? [operand("bias", operation.bias, ["output_feature"])] : [])];
        break;
      case "tensor_scale":
        operands = [operand("tensor-scale", operation.scalar, scalarLogicalIndices(operation.scalar))];
        break;
      case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "rotary_embedding":
      case "scaled_dot_product_attention": case "activation": case "elementwise":
        break;
    }
    if (operands.length === 0) continue;
    validateUniqueRoles(scope, operation.id, operands);
    output.push({ scope, definitionId: operation.id, operands });
  }
}

function operand(
  role: Gemma4LiteralLearnedOperandRole,
  tensor: TensorRef,
  logicalIndices: string[],
): Gemma4LiteralLearnedOperand {
  if (logicalIndices.length !== tensor.shape.length) {
    throw new Error(`${tensor.name}: papel ${role} declara ${logicalIndices.length} índices para rank ${tensor.shape.length}.`);
  }
  return {
    role,
    tensor: structuredClone(tensor),
    logicalIndices,
    decoderId: `decode_${tensor.name}`,
  };
}

function scalarLogicalIndices(tensor: TensorRef): string[] {
  if (tensor.shape.length === 0) return [];
  if (tensor.shape.length === 1 && tensor.shape[0] === 1) return ["0"];
  throw new Error(`${tensor.name}: operando escalar requer shape [] ou [1].`);
}

function requireCount(id: string, references: readonly TensorRef[], minimum: number, maximum = minimum): void {
  if (references.length < minimum || references.length > maximum) {
    throw new Error(`${id}: contrato aprendido requer ${minimum === maximum ? minimum : `${minimum}..${maximum}`} tensores; recebeu ${references.length}.`);
  }
}

function validateUniqueRoles(scope: string, definitionId: string, operands: readonly Gemma4LiteralLearnedOperand[]): void {
  const roles = new Set<Gemma4LiteralLearnedOperandRole>();
  for (const operand of operands) {
    if (roles.has(operand.role)) throw new Error(`${scope}:${definitionId}: papel aprendido duplicado ${operand.role}.`);
    roles.add(operand.role);
  }
}
