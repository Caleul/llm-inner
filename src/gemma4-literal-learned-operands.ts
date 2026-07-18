import { isDeepStrictEqual } from "node:util";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import {
  buildGemma4LiteralCalculationDomains,
  type Gemma4LiteralValueDomain,
} from "./gemma4-literal-domains.js";
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

export type Gemma4LiteralIntegerExpression =
  | { kind: "output-coordinate"; axis: string }
  | { kind: "reduction-index"; name: string; minInclusive: 0; endExclusive: number }
  | { kind: "input-scalar"; name: string; minInclusive: 0; endExclusive: number }
  | { kind: "constant"; value: number }
  | { kind: "add" | "multiply" | "modulo"; left: Gemma4LiteralIntegerExpression; right: Gemma4LiteralIntegerExpression };

export interface Gemma4LiteralIntegerExpressionLanguage {
  id: "gemma4-learned-index-expression-v1";
  resultType: "non-negative-safe-integer";
  evaluationOrder: "depth-first-left-to-right";
  bindings: {
    outputCoordinate: "named axis from the owning calculation output domain";
    reductionIndex: "named scalar from the owning calculation reduction domain";
    inputScalar: "explicit caller-bound scalar such as token_id or position_index";
  };
  arithmetic: {
    add: "exact safe-integer addition";
    multiply: "exact safe-integer multiplication";
    modulo: "non-negative remainder with a strictly positive divisor";
  };
  bounds: "every evaluated index must satisfy 0 <= index < the matching logical tensor dimension";
}

export interface Gemma4LiteralLearnedIndexEnvironment {
  outputCoordinates: Readonly<Record<string, number>>;
  reductionIndices?: Readonly<Record<string, number>>;
  inputScalars?: Readonly<Record<string, number>>;
}

export interface Gemma4LiteralLearnedOperand {
  role: Gemma4LiteralLearnedOperandRole;
  tensor: TensorRef;
  /**
   * Expressions are evaluated against the scalar view's named output
   * coordinates. They bind a logical tensor address before the storage
   * decoder performs row-major index-to-byte mapping.
   */
  logicalIndices: Gemma4LiteralIntegerExpression[];
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
  schemaVersion: 2;
  indexLanguage: Gemma4LiteralIntegerExpressionLanguage;
  assignments: Gemma4LiteralLearnedOperandBinding[];
}

type MultimodalAssignment = Gemma4CompositeProgram["assignments"][number] | Gemma4VisionAssignment | Gemma4AudioAssignment;

export function buildGemma4LiteralLearnedOperandBindings(program: Gemma4CompositeProgram): Gemma4LiteralLearnedOperandBindings {
  const assignments: Gemma4LiteralLearnedOperandBinding[] = [];
  const domains = buildGemma4LiteralCalculationDomains(program);
  addMultimodal(assignments, "composite", program.assignments, domains.assignments, program);
  addMultimodal(assignments, "vision", program.visionProgram.assignments, domains.assignments, program);
  addMultimodal(assignments, "audio", program.audioProgram.assignments, domains.assignments, program);
  addText(assignments, "text-prelude", program.textProgram.prelude, domains.assignments);
  for (const layer of program.textProgram.layers) addText(assignments, "text-layer", layer.operations, domains.assignments);
  addText(assignments, "text-epilogue", program.textProgram.epilogue, domains.assignments);
  return {
    schemaVersion: 2,
    indexLanguage: indexExpressionLanguage(),
    assignments,
  };
}

export function evaluateGemma4LiteralLearnedOperandIndices(
  operand: Gemma4LiteralLearnedOperand,
  environment: Gemma4LiteralLearnedIndexEnvironment,
): number[] {
  const indices = operand.logicalIndices.map((expression) => evaluateIndexExpression(expression, environment));
  if (indices.length !== operand.tensor.shape.length) {
    throw new Error(`${operand.tensor.name}: programa de índice produziu rank ${indices.length}; esperado ${operand.tensor.shape.length}.`);
  }
  indices.forEach((index, axis) => {
    const dimension = operand.tensor.shape[axis]!;
    if (index < 0 || index >= dimension) {
      throw new Error(`${operand.tensor.name}: índice aprendido ${index} fora do eixo ${axis} de tamanho ${dimension}.`);
    }
  });
  return indices;
}

export function gemma4LiteralOutputCoordinateEnvironment(
  domain: Gemma4LiteralValueDomain,
  coordinate: readonly number[],
): Gemma4LiteralLearnedIndexEnvironment {
  if (coordinate.length !== domain.axes.length) {
    throw new Error(`Coordenada aprendida possui rank ${coordinate.length}; domínio ${domain.shape.join("x")} requer ${domain.axes.length}.`);
  }
  const outputCoordinates: Record<string, number> = {};
  domain.axes.forEach((axis, index) => {
    const value = coordinate[index]!;
    if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${axis.name}: coordenada aprendida inválida ${value}.`);
    outputCoordinates[axis.name] = value;
  });
  return { outputCoordinates };
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
  domains: ReturnType<typeof buildGemma4LiteralCalculationDomains>["assignments"],
  program: Gemma4CompositeProgram,
): void {
  for (const assignment of assignments) {
    const references = assignment.tensors ?? [];
    if (references.length === 0) continue;
    const domain = requiredDomain(domains, scope, assignment.id);
    const operands = multimodalOperands(assignment, references, domain, program);
    validateUniqueRoles(scope, assignment.id, operands);
    output.push({ scope, definitionId: assignment.id, operands });
  }
}

function multimodalOperands(
  assignment: MultimodalAssignment,
  references: readonly TensorRef[],
  domain: Gemma4LiteralValueDomain,
  program: Gemma4CompositeProgram,
): Gemma4LiteralLearnedOperand[] {
  switch (assignment.operation) {
    case "linear":
      requireCount(assignment.id, references, 1, 2);
      return [operand("weight", references[0]!, [outputAxis(domain, "output_feature"), reduction("input_feature", references[0]!.shape[1]!)]),
        ...(references[1] ? [operand("bias", references[1], [outputAxis(domain, "output_feature")])] : [])];
    case "clipped-linear":
      requireCount(assignment.id, references, 5);
      return [
        operand("weight", references[0]!, [outputAxis(domain, "output_feature"), reduction("input_feature", references[0]!.shape[1]!)]),
        operand("input-min", references[1]!, scalarLogicalIndices(references[1]!)),
        operand("input-max", references[2]!, scalarLogicalIndices(references[2]!)),
        operand("output-min", references[3]!, scalarLogicalIndices(references[3]!)),
        operand("output-max", references[4]!, scalarLogicalIndices(references[4]!)),
      ];
    case "embedding":
      requireCount(assignment.id, references, 1);
      return [operand("weight", references[0]!, [inputScalar("token_id", references[0]!.shape[0]!), lastOutputAxis(domain)])];
    case "per-layer-embedding":
      requireCount(assignment.id, references, 1);
      return [operand("weight", references[0]!, [inputScalar("token_id", references[0]!.shape[0]!), add(
        multiply(outputAxis(domain, "layer"), constant(program.contract.text.perLayerInputSize)),
        lastOutputAxis(domain),
      )])];
    case "rms-norm":
      requireCount(assignment.id, references, 1);
      return [operand("normalization-scale", references[0]!, [lastOutputAxis(domain)])];
    case "position-embedding-2d":
      requireCount(assignment.id, references, 1);
      return [operand("position-table", references[0]!, [
        inputScalar("position_axis", references[0]!.shape[0]!),
        inputScalar("position_index", references[0]!.shape[1]!),
        lastOutputAxis(domain),
      ])];
    case "conv2d-stride2":
      requireCount(assignment.id, references, 1);
      return [operand("convolution-kernel", references[0]!, [
        outputAxis(domain, "channel"),
        reduction("input_channel", references[0]!.shape[1]!),
        reduction("kernel_time", references[0]!.shape[2]!),
        reduction("kernel_feature", references[0]!.shape[3]!),
      ])];
    case "layer-norm-channels":
      requireCount(assignment.id, references, 1);
      return [operand("normalization-scale", references[0]!, [outputAxis(domain, "channel")])];
    case "causal-depthwise-convolution":
      requireCount(assignment.id, references, 1);
      return [operand("convolution-kernel", references[0]!, [lastOutputAxis(domain), constant(0), reduction("kernel_index", references[0]!.shape[2]!)])];
    case "per-dim-softplus-scale":
      requireCount(assignment.id, references, 1);
      return [operand("per-dimension-scale", references[0]!, [modulo(lastOutputAxis(domain), constant(program.audioProgram.tower.headDim))])];
    default:
      throw new Error(`${assignment.id}: operação ${assignment.operation} usa tensores sem contrato explícito de papel aprendido.`);
  }
}

function addText(
  output: Gemma4LiteralLearnedOperandBinding[],
  scope: "text-prelude" | "text-layer" | "text-epilogue",
  operations: readonly Operation[],
  domains: ReturnType<typeof buildGemma4LiteralCalculationDomains>["assignments"],
): void {
  for (const operation of operations) {
    const domain = requiredDomain(domains, scope, operation.id);
    let operands: Gemma4LiteralLearnedOperand[] = [];
    switch (operation.op) {
      case "embedding":
        operands = [operand("weight", operation.weight, [inputScalar("token_id", operation.weight.shape[0]!), lastOutputAxis(domain)])];
        break;
      case "per_layer_embedding":
        operands = [operand("weight", operation.weight, [inputScalar("token_id", operation.weight.shape[0]!), add(
          multiply(outputAxis(domain, "layer"), constant(operation.layerWidth)), lastOutputAxis(domain),
        )])];
        break;
      case "rms_norm":
        operands = operation.weight ? [operand("normalization-scale", operation.weight, [lastOutputAxis(domain)])] : [];
        break;
      case "linear":
        operands = [operand("weight", operation.weight, [outputAxis(domain, "output_feature"), reduction("input_feature", operation.inFeatures)]),
          ...(operation.bias ? [operand("bias", operation.bias, [outputAxis(domain, "output_feature")])] : [])];
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
  logicalIndices: Gemma4LiteralIntegerExpression[],
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

function scalarLogicalIndices(tensor: TensorRef): Gemma4LiteralIntegerExpression[] {
  if (tensor.shape.length === 0) return [];
  if (tensor.shape.length === 1 && tensor.shape[0] === 1) return [constant(0)];
  throw new Error(`${tensor.name}: operando escalar requer shape [] ou [1].`);
}

function indexExpressionLanguage(): Gemma4LiteralIntegerExpressionLanguage {
  return {
    id: "gemma4-learned-index-expression-v1",
    resultType: "non-negative-safe-integer",
    evaluationOrder: "depth-first-left-to-right",
    bindings: {
      outputCoordinate: "named axis from the owning calculation output domain",
      reductionIndex: "named scalar from the owning calculation reduction domain",
      inputScalar: "explicit caller-bound scalar such as token_id or position_index",
    },
    arithmetic: {
      add: "exact safe-integer addition",
      multiply: "exact safe-integer multiplication",
      modulo: "non-negative remainder with a strictly positive divisor",
    },
    bounds: "every evaluated index must satisfy 0 <= index < the matching logical tensor dimension",
  };
}

function evaluateIndexExpression(
  expression: Gemma4LiteralIntegerExpression,
  environment: Gemma4LiteralLearnedIndexEnvironment,
): number {
  switch (expression.kind) {
    case "output-coordinate": return requiredInteger(environment.outputCoordinates, expression.axis, expression.kind);
    case "reduction-index": return boundedInteger(
      requiredInteger(environment.reductionIndices, expression.name, expression.kind), expression, expression.kind,
    );
    case "input-scalar": return boundedInteger(
      requiredInteger(environment.inputScalars, expression.name, expression.kind), expression, expression.kind,
    );
    case "constant": return safeInteger(expression.value, "constant");
    case "add": return safeInteger(evaluateIndexExpression(expression.left, environment) + evaluateIndexExpression(expression.right, environment), "add");
    case "multiply": return safeInteger(evaluateIndexExpression(expression.left, environment) * evaluateIndexExpression(expression.right, environment), "multiply");
    case "modulo": {
      const left = evaluateIndexExpression(expression.left, environment);
      const right = evaluateIndexExpression(expression.right, environment);
      if (right <= 0) throw new Error(`modulo: divisor de índice aprendido deve ser positivo; recebeu ${right}.`);
      return safeInteger(left % right, "modulo");
    }
  }
}

function requiredInteger(values: Readonly<Record<string, number>> | undefined, name: string, kind: string): number {
  if (!values || !(name in values)) throw new Error(`${kind}: binding de índice aprendido ausente: ${name}.`);
  return safeInteger(values[name]!, `${kind}:${name}`);
}

function safeInteger(value: number, context: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${context}: índice aprendido deve ser inteiro seguro não negativo; recebeu ${value}.`);
  return value;
}

function requiredDomain(
  domains: ReturnType<typeof buildGemma4LiteralCalculationDomains>["assignments"],
  scope: Gemma4LiteralLearnedOperandScope,
  definitionId: string,
): Gemma4LiteralValueDomain {
  const domain = domains.find((candidate) => candidate.scope === scope && candidate.definitionId === definitionId)?.domain;
  if (!domain) throw new Error(`${scope}:${definitionId}: domínio ausente para índices aprendidos.`);
  return domain;
}

function outputAxis(domain: Gemma4LiteralValueDomain, axis: string): Gemma4LiteralIntegerExpression {
  if (!domain.axes.some((candidate) => candidate.name === axis)) throw new Error(`${axis}: eixo de saída ausente em ${domain.shape.join("x")}.`);
  return { kind: "output-coordinate", axis };
}

function lastOutputAxis(domain: Gemma4LiteralValueDomain): Gemma4LiteralIntegerExpression {
  const axis = domain.axes.at(-1)?.name;
  if (!axis) throw new Error("Operando aprendido requer domínio de saída não escalar.");
  return outputAxis(domain, axis);
}

function boundedInteger(value: number, bounds: { minInclusive: 0; endExclusive: number }, context: string): number {
  if (!Number.isSafeInteger(bounds.endExclusive) || bounds.endExclusive <= bounds.minInclusive ||
    value < bounds.minInclusive || value >= bounds.endExclusive) {
    throw new Error(`${context}: índice ${value} fora de ${bounds.minInclusive}..${bounds.endExclusive - 1}.`);
  }
  return value;
}

function reduction(name: string, endExclusive: number): Gemma4LiteralIntegerExpression {
  return { kind: "reduction-index", name, minInclusive: 0, endExclusive };
}
function inputScalar(name: string, endExclusive: number): Gemma4LiteralIntegerExpression {
  return { kind: "input-scalar", name, minInclusive: 0, endExclusive };
}
function constant(value: number): Gemma4LiteralIntegerExpression { return { kind: "constant", value }; }
function add(left: Gemma4LiteralIntegerExpression, right: Gemma4LiteralIntegerExpression): Gemma4LiteralIntegerExpression { return { kind: "add", left, right }; }
function multiply(left: Gemma4LiteralIntegerExpression, right: Gemma4LiteralIntegerExpression): Gemma4LiteralIntegerExpression { return { kind: "multiply", left, right }; }
function modulo(left: Gemma4LiteralIntegerExpression, right: Gemma4LiteralIntegerExpression): Gemma4LiteralIntegerExpression { return { kind: "modulo", left, right }; }

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
