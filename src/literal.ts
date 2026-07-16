import { executeReferenceF32, generateReferenceF32 } from "./executor.js";
import type {
  DenseF32Tensor,
  JsonObject,
  ModelCatalog,
  ModelIR,
  Operation,
  ReferenceF32ExecutionRequest,
  ReferenceF32ExecutionResult,
  ReferenceF32GenerationRequest,
  ReferenceF32GenerationResult,
  TensorInfo,
  TensorRef,
} from "./types.js";

/** Storage-level reader required to remove source-checkpoint dependencies. */
export interface LiteralTensorReader {
  readTensorBytes(tensor: TensorInfo): Promise<Buffer>;
}

export interface LiteralInput {
  name: "input_ids" | "position_ids" | "attention_mask" | "past_key_values";
  description: string;
  required: boolean;
  dtype: string;
  shape: string;
}

export interface LiteralConstant {
  name: string;
  storageDtype: "F32";
  storageShape: number[];
  logicalShape: number[];
  layout: "row-major";
  byteOrder: "little-endian";
  encoding: "base64";
  payloadBase64: string;
}

export interface LiteralKvCacheTransition {
  id: string;
  layer: number;
  operation: "append-post-rope" | "reuse-producer";
  keyInput: string;
  valueInput: string;
  cacheOutput: string;
  producerLayer?: number;
}

/**
 * A source-independent, literal calculation program for the dense F32
 * Safetensors contract. The operation graph intentionally retains the same
 * stable operation ids and named dataflow as ModelIR, but every referenced
 * tensor is embedded as exact little-endian bytes rather than a shard path.
 */
export interface LiteralCalculationProgram {
  schemaVersion: 1;
  kind: "literal-calculation-program";
  sourceFormat: "safetensors";
  architecture: ModelIR["architecture"];
  config: JsonObject;
  numericPolicy: {
    inputDtype: "I32";
    computeDtype: "F32";
    accumulationDtype: "F32";
    outputDtype: "F32";
    scalarSemantics: "IEEE-754 binary32; host libm results rounded to F32";
  };
  inputs: LiteralInput[];
  constants: LiteralConstant[];
  assignments: {
    prelude: Operation[];
    layers: ModelIR["layers"];
    epilogue: Operation[];
  };
  stateTransitions: LiteralKvCacheTransition[];
  outputs: { logits: string };
}

const F32_BYTES = 4;

/**
 * Creates a self-contained literal program from an already validated
 * architecture IR. This deliberately accepts only ordinary dense F32
 * Safetensors: F16/BF16 conversion and quantized dequantization need their
 * own literal decoder assignments and are rejected instead of being silently
 * materialized into a different storage contract.
 */
export async function buildDenseF32LiteralProgram(
  ir: ModelIR,
  catalog: ModelCatalog,
  reader: LiteralTensorReader,
): Promise<LiteralCalculationProgram> {
  if (catalog.format !== "safetensors") {
    throw new Error(`Exportação literal F32 requer Safetensors denso; recebeu formato ${catalog.format}.`);
  }
  const references = referencedTensors(ir);
  const constants: LiteralConstant[] = [];
  for (const reference of references.values()) {
    const tensor = catalog.tensors.get(reference.name);
    assertDenseF32Reference(reference, tensor);
    const payload = await reader.readTensorBytes(tensor!);
    const expectedBytes = product(tensor!.storageShape) * F32_BYTES;
    if (payload.length !== expectedBytes) {
      throw new Error(`${reference.name}: payload literal possui ${payload.length} bytes, esperado ${expectedBytes} para F32.`);
    }
    constants.push({
      name: reference.name,
      storageDtype: "F32",
      storageShape: [...tensor!.storageShape],
      logicalShape: [...tensor!.logicalShape],
      layout: "row-major",
      byteOrder: "little-endian",
      encoding: "base64",
      payloadBase64: payload.toString("base64"),
    });
  }

  const assignments = cloneF32Assignments(ir);
  const program: LiteralCalculationProgram = {
    schemaVersion: 1,
    kind: "literal-calculation-program",
    sourceFormat: "safetensors",
    architecture: structuredClone(ir.architecture),
    config: structuredClone(ir.config),
    numericPolicy: {
      inputDtype: "I32",
      computeDtype: "F32",
      accumulationDtype: "F32",
      outputDtype: "F32",
      scalarSemantics: "IEEE-754 binary32; host libm results rounded to F32",
    },
    inputs: literalInputs(),
    constants,
    assignments,
    stateTransitions: cacheTransitions(assignments),
    outputs: { logits: assignments.epilogue.some((operation) => operation.output === "softcapped_logits") ? "softcapped_logits" : "logits" },
  };
  validateLiteralCalculationProgram(program);
  return program;
}

/** Replays a validated literal program without opening a source checkpoint. */
export function executeLiteralF32(
  program: LiteralCalculationProgram,
  request: Omit<ReferenceF32ExecutionRequest, "tensors">,
): ReferenceF32ExecutionResult {
  validateLiteralCalculationProgram(program);
  return executeReferenceF32(toEmbeddedModelIR(program), { ...request, tensors: decodeF32Constants(program.constants) });
}

/** Greedily generates from the same embedded constants and explicit KV contract. */
export function generateLiteralF32(
  program: LiteralCalculationProgram,
  request: Omit<ReferenceF32GenerationRequest, "tensors">,
): ReferenceF32GenerationResult {
  validateLiteralCalculationProgram(program);
  return generateReferenceF32(toEmbeddedModelIR(program), { ...request, tensors: decodeF32Constants(program.constants) });
}

/**
 * Defensive validation for programs read from JSON. It proves that every
 * operation reads a declared input, an earlier assignment, or an embedded
 * constant; neither a shard path nor an implicit decoder can enter replay.
 */
export function validateLiteralCalculationProgram(program: LiteralCalculationProgram): void {
  if (program.schemaVersion !== 1 || program.kind !== "literal-calculation-program" || program.sourceFormat !== "safetensors") {
    throw new Error("Programa literal inválido: schemaVersion, kind ou sourceFormat não reconhecido.");
  }
  if (
    program.numericPolicy.inputDtype !== "I32" ||
    program.numericPolicy.computeDtype !== "F32" ||
    program.numericPolicy.accumulationDtype !== "F32" ||
    program.numericPolicy.outputDtype !== "F32"
  ) {
    throw new Error("Programa literal inválido: esta versão requer política explícita F32/I32.");
  }
  const constants = new Map<string, LiteralConstant>();
  for (const constant of program.constants) {
    if (constants.has(constant.name)) throw new Error(`Programa literal contém constante duplicada: ${constant.name}.`);
    if (
      constant.storageDtype !== "F32" || constant.layout !== "row-major" || constant.byteOrder !== "little-endian" || constant.encoding !== "base64" ||
      !sameShape(constant.storageShape, constant.logicalShape) || !validShape(constant.storageShape)
    ) {
      throw new Error(`${constant.name}: contrato de constante literal F32 inválido.`);
    }
    const payload = Buffer.from(constant.payloadBase64, "base64");
    if (payload.length !== product(constant.storageShape) * F32_BYTES) {
      throw new Error(`${constant.name}: payload base64 não corresponde ao shape F32 declarado.`);
    }
    constants.set(constant.name, constant);
  }
  if (constants.size === 0) throw new Error("Programa literal não contém constantes incorporadas.");

  const declaredInputs = new Set(program.inputs.map((input) => input.name));
  for (const required of ["input_ids", "position_ids", "attention_mask", "past_key_values"] as const) {
    if (!declaredInputs.has(required)) throw new Error(`Programa literal não declara a entrada ${required}.`);
  }
  const available = new Set<string>(declaredInputs);
  for (const operation of allOperations(program.assignments)) {
    validateOperationInputs(operation, available, constants);
    if (available.has(operation.output)) throw new Error(`${operation.id}: saída ${operation.output} já foi declarada.`);
    available.add(operation.output);
  }
  if (!available.has(program.outputs.logits)) throw new Error(`Programa literal não produz logits declarados: ${program.outputs.logits}.`);
  validateCacheTransitions(program.stateTransitions, program.assignments);
}

function literalInputs(): LiteralInput[] {
  return [
    { name: "input_ids", description: "Token IDs supplied by the caller.", required: true, dtype: "I32", shape: "[batch, sequence]" },
    { name: "position_ids", description: "Absolute RoPE positions; defaults to 0..sequence-1 only when omitted by the declared executor contract.", required: false, dtype: "I32", shape: "[batch, sequence]" },
    { name: "attention_mask", description: "Optional canonical additive attention bias [batch, 1|heads, query, key].", required: false, dtype: "F32", shape: "[batch, 1|heads, query, key]" },
    { name: "past_key_values", description: "Optional canonical post-RoPE KV cache keyed by producer layer.", required: false, dtype: "F32", shape: "layer -> { key/value: [batch, kv_heads, cached_sequence, head_dim] }" },
  ];
}

function cloneF32Assignments(ir: ModelIR): LiteralCalculationProgram["assignments"] {
  const clone = structuredClone({ prelude: ir.prelude, layers: ir.layers, epilogue: ir.epilogue });
  for (const operation of allOperations(clone)) {
    operation.dtypePolicy = { inputDtype: "F32", computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
    // Display-only previews cannot change the executable literal program.
    if (operation.op === "linear") delete operation.preview;
  }
  return clone;
}

function cacheTransitions(assignments: LiteralCalculationProgram["assignments"]): LiteralKvCacheTransition[] {
  return allOperations(assignments)
    .filter((operation): operation is Extract<Operation, { op: "scaled_dot_product_attention" }> => operation.op === "scaled_dot_product_attention")
    .map((operation) => {
      if (operation.layer === undefined) throw new Error(`${operation.id}: atenção literal sem camada não possui dono de cache KV.`);
      if (operation.kvSharing) {
        if (operation.kvSharing.producerLayer === undefined) throw new Error(`${operation.id}: compartilhamento KV não declara producerLayer.`);
        return {
          id: `${operation.id}_kv_cache`, layer: operation.layer, operation: "reuse-producer",
          keyInput: operation.key, valueInput: operation.value,
          cacheOutput: `past_key_values.${operation.kvSharing.producerLayer}`,
          producerLayer: operation.kvSharing.producerLayer,
        };
      }
      return {
        id: `${operation.id}_kv_cache`, layer: operation.layer, operation: "append-post-rope",
        keyInput: operation.key, valueInput: operation.value, cacheOutput: `past_key_values.${operation.layer}`,
      };
    });
}

function validateCacheTransitions(transitions: readonly LiteralKvCacheTransition[], assignments: LiteralCalculationProgram["assignments"]): void {
  const attention = allOperations(assignments).filter((operation) => operation.op === "scaled_dot_product_attention");
  if (transitions.length !== attention.length) throw new Error("Programa literal não declara uma transição KV para cada atenção.");
  const byId = new Map(transitions.map((transition) => [transition.id, transition]));
  for (const operation of attention) {
    if (operation.layer === undefined) throw new Error(`${operation.id}: atenção sem camada.`);
    const transition = byId.get(`${operation.id}_kv_cache`);
    if (!transition || transition.layer !== operation.layer || transition.keyInput !== operation.key || transition.valueInput !== operation.value) {
      throw new Error(`${operation.id}: transição KV literal não corresponde à atribuição de atenção.`);
    }
    if (operation.kvSharing) {
      if (transition.operation !== "reuse-producer" || transition.producerLayer !== operation.kvSharing.producerLayer) {
        throw new Error(`${operation.id}: compartilhamento KV literal não corresponde ao produtor declarado.`);
      }
    } else if (transition.operation !== "append-post-rope" || transition.cacheOutput !== `past_key_values.${operation.layer}`) {
      throw new Error(`${operation.id}: transição KV append literal inválida.`);
    }
  }
}

function validateOperationInputs(operation: Operation, available: ReadonlySet<string>, constants: ReadonlyMap<string, LiteralConstant>): void {
  const requireValue = (name: string): void => {
    if (name.startsWith("attention_mask:")) {
      if (!available.has("attention_mask")) throw new Error(`${operation.id}: máscara de atenção não declarada.`);
      return;
    }
    if (!available.has(name)) throw new Error(`${operation.id}: entrada ${name} não foi declarada antes do uso.`);
  };
  const requireTensor = (reference: TensorRef): void => {
    const constant = constants.get(reference.name);
    if (!constant || !sameShape(reference.shape, constant.logicalShape) || reference.storageDtype !== constant.storageDtype || reference.quantization) {
      throw new Error(`${operation.id}: tensor ${reference.name} não possui constante literal F32 compatível.`);
    }
  };
  switch (operation.op) {
    case "embedding": requireValue(operation.tokenInput); requireTensor(operation.weight); break;
    case "rms_norm": requireValue(operation.input); requireTensor(operation.weight); break;
    case "linear": requireValue(operation.input); requireTensor(operation.weight); if (operation.bias) requireTensor(operation.bias); break;
    case "reshape_heads": case "activation": requireValue(operation.input); break;
    case "rotary_embedding": requireValue(operation.input); requireValue(operation.positionInput); break;
    case "scaled_dot_product_attention": requireValue(operation.query); requireValue(operation.key); requireValue(operation.value); requireValue(operation.maskInput); break;
    case "elementwise": operation.inputs.forEach(requireValue); break;
  }
}

function toEmbeddedModelIR(program: LiteralCalculationProgram): ModelIR {
  return {
    schemaVersion: 2,
    source: { path: "embedded://literal-calculation-program", format: "safetensors" },
    architecture: structuredClone(program.architecture),
    config: structuredClone(program.config),
    preview: { outputRows: 1, inputTerms: 1, includeWeights: false },
    inputs: program.inputs.map(({ name, description }) => ({ name, description })),
    prelude: structuredClone(program.assignments.prelude),
    layers: structuredClone(program.assignments.layers),
    epilogue: structuredClone(program.assignments.epilogue),
    fidelity: { exactByConstruction: false, assumptions: [], unsupported: [], warnings: [] },
  };
}

function decodeF32Constants(constants: readonly LiteralConstant[]): ReadonlyMap<string, DenseF32Tensor> {
  const decoded = new Map<string, DenseF32Tensor>();
  for (const constant of constants) {
    const bytes = Buffer.from(constant.payloadBase64, "base64");
    const source = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / F32_BYTES);
    decoded.set(constant.name, { shape: [...constant.logicalShape], values: Float32Array.from(source) });
  }
  return decoded;
}

function referencedTensors(ir: ModelIR): Map<string, TensorRef> {
  const references = new Map<string, TensorRef>();
  for (const operation of allOperations({ prelude: ir.prelude, layers: ir.layers, epilogue: ir.epilogue })) {
    if (operation.op === "embedding" || operation.op === "rms_norm") registerReference(references, operation.weight);
    if (operation.op === "linear") {
      registerReference(references, operation.weight);
      if (operation.bias) registerReference(references, operation.bias);
    }
  }
  return references;
}

function registerReference(references: Map<string, TensorRef>, reference: TensorRef): void {
  const previous = references.get(reference.name);
  if (previous && (previous.storageDtype !== reference.storageDtype || !sameShape(previous.shape, reference.shape) || previous.quantization || reference.quantization)) {
    throw new Error(`IR contém referências incompatíveis para ${reference.name}.`);
  }
  references.set(reference.name, reference);
}

function assertDenseF32Reference(reference: TensorRef, tensor: TensorInfo | undefined): void {
  if (!tensor || reference.storageDtype !== "F32" || tensor.storageDtype !== "F32" || reference.quantization || tensor.quantization) {
    throw new Error(`${reference.name}: exportação literal atual aceita somente tensor Safetensors F32 denso não quantizado.`);
  }
  if (!sameShape(reference.shape, tensor.logicalShape) || !sameShape(tensor.storageShape, tensor.logicalShape) || !validShape(tensor.storageShape)) {
    throw new Error(`${reference.name}: shape lógico/storage incompatível para exportação literal F32.`);
  }
}

function allOperations(assignments: LiteralCalculationProgram["assignments"]): Operation[] {
  return [...assignments.prelude, ...assignments.layers.flatMap((layer) => layer.operations), ...assignments.epilogue];
}

function product(shape: readonly number[]): number {
  return shape.reduce((result, dimension) => result * dimension, 1);
}

function validShape(shape: readonly number[]): boolean {
  return shape.length > 0 && shape.every((dimension) => Number.isInteger(dimension) && dimension > 0) && Number.isSafeInteger(product(shape));
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}
