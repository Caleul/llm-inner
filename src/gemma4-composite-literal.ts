import {
  buildLiteralStorageBundle,
  decodeLiteralStorageBundleF32,
  validateLiteralStorageBundle,
  validateLiteralStorageReference,
  type LiteralConstant,
  type LiteralStorageBundle,
  type LiteralTensorReader,
} from "./literal.js";
import {
  executeGemma4CompositeF32,
  generateGemma4CompositeF32,
  type Gemma4CompositeExecutionRequest,
  type Gemma4CompositeExecutionResult,
  type Gemma4CompositeGenerationRequest,
  type Gemma4CompositeGenerationResult,
  type Gemma4CompositeProgram,
} from "./gemma4-composite.js";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { ModelCatalog, Operation, TensorRef } from "./types.js";

export interface Gemma4CompositeLiteralInput {
  name: string;
  description: string;
  required: boolean;
  dtype: string;
  shape: string;
}

/**
 * A source-independent literal program for the complete registered Gemma 4
 * composite boundary.  The three assignment scopes are kept separate because
 * image and video reuse the vision feature program, while their outer scatter
 * steps remain distinct, named dependencies in the enclosing program.
 */
export interface Gemma4CompositeLiteralCalculationProgram extends LiteralStorageBundle {
  schemaVersion: 1;
  kind: "gemma4-composite-literal-calculation-program";
  sourceFormat: "safetensors";
  numericPolicy: {
    inputDtype: "I32/F32/BOOL";
    computeDtype: "F32";
    accumulationDtype: "F32";
    outputDtype: "F32";
    scalarSemantics: "IEEE-754 binary32; host libm results rounded to F32";
  };
  inputs: Gemma4CompositeLiteralInput[];
  /** No checkpoint path is retained: all tensor bytes are in `constants`. */
  program: Gemma4CompositeProgram;
  assignments: {
    composite: Gemma4CompositeProgram["assignments"];
    vision: Gemma4VisionAssignment[];
    audio: Gemma4AudioAssignment[];
    textLayers: Gemma4CompositeProgram["textProgram"]["layers"];
    textEpilogue: Operation[];
  };
  outputs: { embeddings: "hidden_states_0"; perLayerInputs: "ple_inputs"; logits: "softcapped_logits" | "logits" };
}

/**
 * Literalizes the entire registered composite, not only Gemma4Text.  Dense
 * Gemma 4 is deliberately required here: a quantized package needs an
 * architecture-specific quantization contract before it can be replayed.
 */
export async function buildGemma4CompositeLiteralCalculationProgram(
  program: Gemma4CompositeProgram,
  catalog: ModelCatalog,
  reader: LiteralTensorReader,
): Promise<Gemma4CompositeLiteralCalculationProgram> {
  if (catalog.format !== "safetensors" || program.sourceFormat !== "safetensors") {
    throw new Error("Programa literal Gemma 4 composite requer Safetensors denso registrado.");
  }
  const references = compositeReferences(program);
  const catalogNames = new Set(catalog.tensors.keys());
  if (references.size !== catalogNames.size || [...catalogNames].some((name) => !references.has(name))) {
    const omitted = [...catalogNames].filter((name) => !references.has(name));
    throw new Error(`Programa literal Gemma 4 composite não possui atribuição semântica para todos os tensores do pacote${omitted.length ? `: ${omitted.slice(0, 4).join(", ")}` : "."}`);
  }
  const storage = await buildLiteralStorageBundle(catalog, reader, references.values());
  const embeddedProgram = embeddedCompositeProgram(program);
  const literal: Gemma4CompositeLiteralCalculationProgram = {
    schemaVersion: 1,
    kind: "gemma4-composite-literal-calculation-program",
    sourceFormat: "safetensors",
    numericPolicy: {
      inputDtype: "I32/F32/BOOL",
      computeDtype: "F32",
      accumulationDtype: "F32",
      outputDtype: "F32",
      scalarSemantics: "IEEE-754 binary32; host libm results rounded to F32",
    },
    inputs: literalInputs(),
    ...storage,
    program: embeddedProgram,
    assignments: {
      composite: structuredClone(embeddedProgram.assignments),
      vision: structuredClone(embeddedProgram.visionProgram.assignments),
      audio: structuredClone(embeddedProgram.audioProgram.assignments),
      textLayers: structuredClone(embeddedProgram.textProgram.layers),
      textEpilogue: structuredClone(embeddedProgram.textProgram.epilogue),
    },
    outputs: structuredClone(embeddedProgram.outputs),
  };
  validateGemma4CompositeLiteralCalculationProgram(literal);
  return literal;
}

/** Replays prefill solely from the literal's embedded bytes and assignments. */
export function executeGemma4CompositeLiteralF32(
  literal: Gemma4CompositeLiteralCalculationProgram,
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
): Gemma4CompositeExecutionResult {
  validateGemma4CompositeLiteralCalculationProgram(literal);
  return executeGemma4CompositeF32(literal.program, { ...request, tensors: decodeLiteralStorageBundleF32(literal) });
}

/** Replays greedy cached decoding solely from the literal's embedded bytes. */
export function generateGemma4CompositeLiteralF32(
  literal: Gemma4CompositeLiteralCalculationProgram,
  request: Omit<Gemma4CompositeGenerationRequest, "tensors">,
): Gemma4CompositeGenerationResult {
  validateGemma4CompositeLiteralCalculationProgram(literal);
  return generateGemma4CompositeF32(literal.program, { ...request, tensors: decodeLiteralStorageBundleF32(literal) });
}

/**
 * Validates all three semantic scopes and their storage references before
 * replay.  This rejects a partial tower, an omitted vision-mask/cache step,
 * or a program which tries to retain a source-model path.
 */
export function validateGemma4CompositeLiteralCalculationProgram(literal: Gemma4CompositeLiteralCalculationProgram): void {
  if (literal.schemaVersion !== 1 || literal.kind !== "gemma4-composite-literal-calculation-program" || literal.sourceFormat !== "safetensors" ||
    literal.numericPolicy.inputDtype !== "I32/F32/BOOL" || literal.numericPolicy.computeDtype !== "F32" ||
    literal.numericPolicy.accumulationDtype !== "F32" || literal.numericPolicy.outputDtype !== "F32" ||
    literal.numericPolicy.scalarSemantics !== "IEEE-754 binary32; host libm results rounded to F32") {
    throw new Error("Programa literal Gemma 4 composite possui cabeçalho ou política numérica inválida.");
  }
  validateLiteralStorageBundle(literal);
  const constants = new Map<string, LiteralConstant>(literal.constants.map((constant) => [constant.name, constant]));
  const requiredInputs = literalInputs().map((input) => input.name);
  if (literal.inputs.length !== requiredInputs.length || requiredInputs.some((name) => !literal.inputs.some((input) => input.name === name))) {
    throw new Error("Programa literal Gemma 4 composite não declara todos os controles de entrada multimodal.");
  }
  if (literal.program.textProgram.source.path !== "embedded://gemma4-composite-literal" || literal.program.textProgram.source.format !== "safetensors") {
    throw new Error("Programa literal Gemma 4 composite reteve uma referência de source checkpoint.");
  }
  const compositeRequired = ["composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask", "composite_text_core"];
  if (compositeRequired.some((id) => !literal.assignments.composite.some((assignment) => assignment.id === id))) {
    throw new Error("Programa literal Gemma 4 composite omite uma transição de máscara ou cache obrigatória.");
  }
  if (literal.outputs.logits !== literal.program.outputs.logits || literal.assignments.composite.length !== literal.program.assignments.length ||
    literal.assignments.vision.length !== literal.program.visionProgram.assignments.length || literal.assignments.audio.length !== literal.program.audioProgram.assignments.length) {
    throw new Error("Programa literal Gemma 4 composite diverge das atribuições registradas.");
  }
  validateAssignmentScope(literal.assignments.composite, new Set(requiredInputs), constants, "composite");
  validateAssignmentScope(literal.assignments.vision, new Set(["pixel_values", "pixel_position_ids", "text_embeddings", "input_ids"]), constants, "vision");
  validateAssignmentScope(literal.assignments.audio, new Set(["input_features", "input_features_mask", "text_embeddings", "input_ids"]), constants, "audio");
  validateTextScope(literal.assignments.textLayers, literal.assignments.textEpilogue, constants);

  const registered = compositeReferences(literal.program);
  const literalNames = new Set(literal.constants.map((constant) => constant.name));
  if (registered.size !== literalNames.size || [...registered.keys()].some((name) => !literalNames.has(name))) {
    throw new Error("Programa literal Gemma 4 composite contém constantes omitidas ou não alcançáveis pelas atribuições.");
  }
}

function literalInputs(): Gemma4CompositeLiteralInput[] {
  return [
    { name: "input_ids", description: "Token IDs, including declared modal placeholders.", required: true, dtype: "I32", shape: "[batch, sequence]" },
    { name: "position_ids", description: "Absolute text RoPE positions.", required: false, dtype: "I32", shape: "[batch, sequence]" },
    { name: "attention_mask", description: "Caller additive text attention bias; mutually exclusive with mm_token_type_ids.", required: false, dtype: "F32", shape: "[batch, 1|heads, query, key]" },
    { name: "past_key_values", description: "Post-RoPE text KV cache for incremental decode.", required: false, dtype: "F32", shape: "layer -> {key,value}" },
    { name: "pixel_values", description: "Patchified image pixels.", required: false, dtype: "F32", shape: "[batch, patches, 3*patch_size^2]" },
    { name: "image_position_ids", description: "Image patch [x,y] positions; required with pixel_values.", required: false, dtype: "I32", shape: "[batch][patch][x,y]" },
    { name: "pixel_values_videos", description: "Patchified video frames.", required: false, dtype: "F32", shape: "[videos, frames, patches, 3*patch_size^2]" },
    { name: "video_position_ids", description: "Video frame patch [x,y] positions; required with pixel_values_videos.", required: false, dtype: "I32", shape: "[videos][frames][patch][x,y]" },
    { name: "input_features", description: "Audio features before stride-2 subsampling.", required: false, dtype: "F32", shape: "[batch, frames, features]" },
    { name: "input_features_mask", description: "Boolean validity mask for input_features.", required: false, dtype: "BOOL", shape: "[batch, frames]" },
    { name: "mm_token_type_ids", description: "Image/video block IDs used only during uncached prefill.", required: false, dtype: "I32", shape: "[batch, sequence]" },
  ];
}

function embeddedCompositeProgram(program: Gemma4CompositeProgram): Gemma4CompositeProgram {
  const embedded = structuredClone(program);
  embedded.textProgram.source = { path: "embedded://gemma4-composite-literal", format: "safetensors" };
  return embedded;
}

function compositeReferences(program: Gemma4CompositeProgram): Map<string, TensorRef> {
  const references = new Map<string, TensorRef>();
  const register = (reference: TensorRef): void => {
    const previous = references.get(reference.name);
    if (previous && (previous.storageDtype !== reference.storageDtype || !sameShape(previous.shape, reference.shape) || !sameQuantization(previous.quantization, reference.quantization))) {
      throw new Error(`Programa Gemma 4 composite possui referências incompatíveis para ${reference.name}.`);
    }
    references.set(reference.name, reference);
  };
  for (const assignment of [...program.assignments, ...program.visionProgram.assignments, ...program.audioProgram.assignments]) assignment.tensors?.forEach(register);
  for (const operation of [...program.textProgram.prelude, ...program.textProgram.layers.flatMap((layer) => layer.operations), ...program.textProgram.epilogue]) {
    if (operation.op === "embedding" || operation.op === "per_layer_embedding") register(operation.weight);
    if (operation.op === "rms_norm" && operation.weight) register(operation.weight);
    if (operation.op === "linear") { register(operation.weight); if (operation.bias) register(operation.bias); }
    if (operation.op === "tensor_scale") register(operation.scalar);
  }
  return references;
}

function validateAssignmentScope(
  assignments: ReadonlyArray<{ id: string; inputs: string[]; output: string; tensors?: TensorRef[] }>,
  inputs: Set<string>, constants: ReadonlyMap<string, LiteralConstant>, scope: string,
): void {
  const available = new Set(inputs);
  for (const assignment of assignments) {
    for (const input of assignment.inputs) if (!available.has(input)) throw new Error(`${scope}:${assignment.id} lê predecessor não declarado ${input}.`);
    assignment.tensors?.forEach((reference) => validateLiteralStorageReference(reference, constants));
    if (available.has(assignment.output)) throw new Error(`${scope}:${assignment.id} redeclara saída ${assignment.output}.`);
    available.add(assignment.output);
  }
}

function validateTextScope(layers: Gemma4CompositeProgram["textProgram"]["layers"], epilogue: Operation[], constants: ReadonlyMap<string, LiteralConstant>): void {
  const available = new Set(["input_ids", "position_ids", "attention_mask", "past_key_values", "hidden_states_0", "ple_inputs"]);
  for (const operation of [...layers.flatMap((layer) => layer.operations), ...epilogue]) {
    textOperationInputs(operation).forEach((input) => {
      if (input.startsWith("attention_mask:")) { if (!available.has("attention_mask")) throw new Error(`${operation.id}: máscara literal não declarada.`); }
      else if (!available.has(input)) throw new Error(`${operation.id}: predecessor literal não declarado ${input}.`);
    });
    textOperationTensors(operation).forEach((reference) => validateLiteralStorageReference(reference, constants));
    if (available.has(operation.output)) throw new Error(`${operation.id}: saída textual literal duplicada.`);
    available.add(operation.output);
  }
}

function textOperationInputs(operation: Operation): string[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.tokenInput];
    case "rms_norm": case "linear": case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": case "tensor_scale": return [operation.input];
    case "rotary_embedding": return [operation.input, operation.positionInput];
    case "scaled_dot_product_attention": return [operation.query, operation.key, operation.value, operation.maskInput];
    case "elementwise": return operation.inputs;
  }
}

function textOperationTensors(operation: Operation): TensorRef[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.weight];
    case "rms_norm": return operation.weight ? [operation.weight] : [];
    case "linear": return operation.bias ? [operation.weight, operation.bias] : [operation.weight];
    case "tensor_scale": return [operation.scalar];
    default: return [];
  }
}

function sameShape(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function sameQuantization(left: TensorRef["quantization"], right: TensorRef["quantization"]): boolean {
  if (!left || !right) return left === right;
  return left.family === right.family && left.mode === right.mode && left.bits === right.bits && left.groupSize === right.groupSize && left.tensorType === right.tensorType && left.scaleTensor === right.scaleTensor && left.biasTensor === right.biasTensor && left.globalScaleTensor === right.globalScaleTensor;
}
