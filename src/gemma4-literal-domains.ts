import { isDeepStrictEqual } from "node:util";
import type { Gemma4AudioAssignment } from "./gemma4-audio.js";
import type { Gemma4CompositeAssignment, Gemma4CompositeProgram } from "./gemma4-composite.js";
import type { Gemma4VisionAssignment } from "./gemma4-vision.js";
import type { DtypePolicy, Operation } from "./types.js";

export type Gemma4LiteralCalculationScope = "composite" | "vision" | "audio" | "text-prelude" | "text-layer" | "text-epilogue";

export interface Gemma4LiteralAxisDomain {
  axis: number;
  name: string;
  size: string;
  indexDomain: string;
}

/**
 * A serialized, source-independent coordinate contract for one named value.
 * Dynamic dimensions are algebraic expressions over declared caller-input
 * dimensions; fixed architecture dimensions are emitted as decimal literals.
 */
export interface Gemma4LiteralValueDomain {
  structure: "tensor" | "tuple";
  dtype: string;
  shape: string[];
  layout: "row-major" | "tuple-of-row-major-tensors";
  axes: Gemma4LiteralAxisDomain[];
  dimensionDefinitions?: Record<string, string>;
  dtypePolicy?: DtypePolicy;
}

export interface Gemma4LiteralAssignmentDomain {
  definitionId: string;
  operation: string;
  scope: Gemma4LiteralCalculationScope;
  output: string;
  domain: Gemma4LiteralValueDomain;
}

export interface Gemma4LiteralCalculationDomains {
  kind: "gemma4-literal-calculation-domains";
  schemaVersion: 1;
  dimensionDefinitions: Record<string, string>;
  assignments: Gemma4LiteralAssignmentDomain[];
}

type Assignment = Gemma4CompositeAssignment | Gemma4VisionAssignment | Gemma4AudioAssignment;

const COMMON_DIMENSIONS = {
  B: "input_ids.shape[0]",
  S: "input_ids.shape[1]",
  K: "past_key_values key length plus S; S when no cache is supplied",
  VB: "vision-program invocation batch (images, or videos multiplied by frames)",
  VP: "pixel_values.shape[-2] patches per vision invocation",
  VPOOL: "VP divided by vision_pooling_kernel_size squared; the adapter requires an integer",
  VVALID: "number of true pooled-position mask rows across VB in stable batch-major order",
  AB: "input_features.shape[0]",
  AT0: "input_features.shape[1]",
  AF0: "input_features.shape[2]",
  AT1: "ceil(AT0/2)",
  AF1: "ceil(AF0/2)",
  AT2: "ceil(AT1/2)",
  AF2: "ceil(AF1/2)",
  ABLOCKS: "ceil(AT2/audio_attention_chunk_size)",
  AVALID: "number of true audio_output_mask rows across AB in stable batch-major order",
} as const;

export function buildGemma4LiteralCalculationDomains(program: Gemma4CompositeProgram): Gemma4LiteralCalculationDomains {
  const assignments = [
    ...buildCompositeDomains(program),
    ...buildVisionDomains(program),
    ...buildAudioDomains(program),
    ...buildTextDomains(program),
  ];
  const ids = new Set<string>();
  for (const entry of assignments) {
    const key = `${entry.scope}:${entry.definitionId}`;
    if (ids.has(key)) throw new Error(`Domínio literal Gemma 4 duplicado: ${key}.`);
    ids.add(key);
    validateValueDomain(entry.domain, key);
  }
  return {
    kind: "gemma4-literal-calculation-domains",
    schemaVersion: 1,
    dimensionDefinitions: { ...COMMON_DIMENSIONS },
    assignments,
  };
}

export function validateGemma4LiteralCalculationDomains(
  domains: Gemma4LiteralCalculationDomains,
  program: Gemma4CompositeProgram,
): void {
  if (domains.kind !== "gemma4-literal-calculation-domains" || domains.schemaVersion !== 1) {
    throw new Error("Programa literal Gemma 4 possui cabeçalho de domínios de cálculo inválido.");
  }
  const expected = buildGemma4LiteralCalculationDomains(program);
  if (!isDeepStrictEqual(domains, expected)) {
    throw new Error("Programa literal Gemma 4 possui domínios de shape/dtype/layout ausentes ou divergentes.");
  }
}

export function instantiateGemma4LiteralValueDomain(
  domain: Gemma4LiteralValueDomain,
  invocationId: string | undefined,
): Gemma4LiteralValueDomain {
  if (!invocationId) return structuredClone(domain);
  const replacements = invocationId === "composite_image_features"
    ? { VB: "IMAGE_BATCH", VP: "IMAGE_PATCHES", VPOOL: "IMAGE_POOL_CELLS", VVALID: "IMAGE_SOFT_TOKENS" }
    : invocationId === "composite_video_features"
      ? { VB: "VIDEO_BATCH*VIDEO_FRAMES", VP: "VIDEO_PATCHES", VPOOL: "VIDEO_POOL_CELLS", VVALID: "VIDEO_SOFT_TOKENS" }
      : undefined;
  if (!replacements) return structuredClone(domain);
  const instantiatedDefinitions = invocationId === "composite_image_features"
    ? {
        IMAGE_BATCH: "pixel_values.shape[0]", IMAGE_PATCHES: "pixel_values.shape[1]",
        IMAGE_POOL_CELLS: "IMAGE_PATCHES divided by vision_pooling_kernel_size squared",
        IMAGE_SOFT_TOKENS: "true rows in the image pooled-position mask in batch-major order",
      }
    : {
        VIDEO_BATCH: "pixel_values_videos.shape[0]", VIDEO_FRAMES: "pixel_values_videos.shape[1]",
        VIDEO_PATCHES: "pixel_values_videos.shape[2]",
        VIDEO_POOL_CELLS: "VIDEO_PATCHES divided by vision_pooling_kernel_size squared",
        VIDEO_SOFT_TOKENS: "true rows in the video pooled-position mask in video-frame-major order",
      };
  const replace = (value: string): string => Object.entries(replacements).sort(([left], [right]) => right.length - left.length)
    .reduce((current, [source, target]) => current.replaceAll(source, target), value);
  return {
    ...structuredClone(domain),
    shape: domain.shape.map(replace),
    axes: domain.axes.map((axis) => ({ ...axis, size: replace(axis.size), indexDomain: replace(axis.indexDomain) })),
    dimensionDefinitions: instantiatedDefinitions,
  };
}

function buildCompositeDomains(program: Gemma4CompositeProgram): Gemma4LiteralAssignmentDomain[] {
  const hidden = program.contract.text.hiddenSize;
  const layers = program.contract.text.layers;
  const ple = program.contract.text.perLayerInputSize;
  const vocab = program.contract.text.vocabSize;
  const prelude = new Map(program.textProgram.prelude.map((operation) => [operation.id, operation]));
  const preludeByOutput = new Map(program.textProgram.prelude.map((operation) => [operation.output, operation]));
  const values = new Map<string, Gemma4LiteralValueDomain>([
    ["input_ids", tensor("I32", ["B", "S"], ["batch", "sequence"])],
    ["mm_token_type_ids", tensor("I32", ["B", "S"], ["batch", "sequence"])],
    ["pixel_values", tensor("F32", ["IMAGE_BATCH", "IMAGE_PATCHES", String(3 * program.contract.modalities.visionTower.patchSize ** 2)], ["image", "patch", "patch_feature"])],
    ["image_position_ids", tensor("I32", ["IMAGE_BATCH", "IMAGE_PATCHES", "2"], ["image", "patch", "xy"])],
    ["pixel_values_videos", tensor("F32", ["VIDEO_BATCH", "VIDEO_FRAMES", "VIDEO_PATCHES", String(3 * program.contract.modalities.visionTower.patchSize ** 2)], ["video", "frame", "patch", "patch_feature"])],
    ["video_position_ids", tensor("I32", ["VIDEO_BATCH", "VIDEO_FRAMES", "VIDEO_PATCHES", "2"], ["video", "frame", "patch", "xy"])],
    ["input_features", tensor("F32", ["AB", "AT0", "AF0"], ["batch", "frame", "feature"])],
    ["input_features_mask", tensor("BOOL", ["AB", "AT0"], ["batch", "frame"])],
  ]);
  const result: Gemma4LiteralAssignmentDomain[] = [];
  for (const assignment of program.assignments) {
    let output: Gemma4LiteralValueDomain;
    switch (assignment.operation) {
      case "placeholder-masks": output = withPolicy(tuple("BOOL", ["3", "B", "S"], ["modality_mask", "batch", "sequence"]), exactPolicy("I32", "comparison", "BOOL")); break;
      case "vision-block-sequence-ids": output = withPolicy(tensor("I32", ["B", "S"], ["batch", "sequence"]), exactPolicy("I32", "exact-integer", "I32")); break;
      case "causal-attention-mask": case "vision-sliding-attention-mask": output = withPolicy(tensor("F32", ["B", "1", "S", "K"], ["batch", "mask_head", "query", "key"]), exactPolicy("I32", "comparison-and-F32-selection", "F32")); break;
      case "replace-multimodal-ids-with-pad": output = withPolicy(tensor("I32", ["B", "S"], ["batch", "sequence"]), exactPolicy("I32", "exact-integer-selection", "I32")); break;
      case "embedding": output = withPolicy(tensor("BF16", ["B", "S", String(hidden)], ["batch", "sequence", "hidden"]), prelude.get("token_embedding")?.dtypePolicy); break;
      case "per-layer-embedding": output = withPolicy(tensor("BF16", ["B", "S", String(layers), String(ple)], ["batch", "sequence", "layer", "ple_feature"]), prelude.get("ple_token_identity")?.dtypePolicy); break;
      case "vision-feature-program": output = withPolicy(tensor("BF16", [assignment.inputs[0] === "pixel_values" ? "IMAGE_SOFT_TOKENS" : "VIDEO_SOFT_TOKENS", String(hidden)], ["soft_token", "hidden"]), exactPolicy("declared-input", "declared-vision-subprogram", "BF16")); break;
      case "video-frame-flatten": output = assignment.inputs[0] === "pixel_values_videos"
        ? withPolicy(tensor("F32", ["VIDEO_BATCH*VIDEO_FRAMES", "VIDEO_PATCHES", String(3 * program.contract.modalities.visionTower.patchSize ** 2)], ["video_frame", "patch", "patch_feature"]), exactPolicy("F32", "row-major-alias", "F32"))
        : withPolicy(tensor("I32", ["VIDEO_BATCH*VIDEO_FRAMES", "VIDEO_PATCHES", "2"], ["video_frame", "patch", "xy"]), exactPolicy("I32", "row-major-alias", "I32")); break;
      case "audio-feature-program": output = withPolicy(tensor("BF16", ["AVALID", String(hidden)], ["soft_token", "hidden"]), exactPolicy("declared-input", "declared-audio-subprogram", "BF16")); break;
      case "masked-scatter": output = withPolicy(cloneDomain(requiredDomain(values, assignment.inputs[0]!, assignment.id)), exactPolicy("BF16", "exact-selection", "BF16")); break;
      case "linear": output = withPolicy(replaceLast(requiredDomain(values, assignment.inputs[0]!, assignment.id), String(requireMatrix(assignment)[0]), "BF16"), preludeByOutput.get(assignment.output)?.dtypePolicy); break;
      case "scale-f32": case "rms-norm": case "add": output = withPolicy(cloneDomain(requiredDomain(values, assignment.inputs[0]!, assignment.id)), preludeByOutput.get(assignment.output)?.dtypePolicy); break;
      case "reshape-per-layer": output = withPolicy(tensor("BF16", ["B", "S", String(layers), String(ple)], ["batch", "sequence", "layer", "ple_feature"]), preludeByOutput.get(assignment.output)?.dtypePolicy); break;
      case "text-core": output = withPolicy(tensor(policyOutput(program.textProgram.epilogue.at(-1)!, "BF16"), ["B", "S", String(vocab)], ["batch", "sequence", "vocabulary"]), exactPolicy("declared-input", "declared-text-subprogram", policyOutput(program.textProgram.epilogue.at(-1)!, "BF16"))); break;
    }
    values.set(assignment.output, output);
    result.push(entry("composite", assignment, output));
  }
  return result;
}

function buildVisionDomains(program: Gemma4CompositeProgram): Gemma4LiteralAssignmentDomain[] {
  const tower = program.visionProgram.tower;
  const hidden = tower.hiddenSize, heads = tower.attentionHeads, headDim = tower.headDim;
  const values = new Map<string, Gemma4LiteralValueDomain>([
    ["pixel_values", tensor("F32", ["VB", "VP", String(3 * tower.patchSize ** 2)], ["batch", "patch", "patch_feature"])],
    ["pixel_position_ids", tensor("I32", ["VB", "VP", "2"], ["batch", "patch", "xy"])],
    ["text_embeddings", tensor("BF16", ["B", "S", String(program.contract.text.hiddenSize)], ["batch", "sequence", "hidden"])],
    ["input_ids", tensor("I32", ["B", "S"], ["batch", "sequence"])],
  ]);
  const result: Gemma4LiteralAssignmentDomain[] = [];
  for (const assignment of program.visionProgram.assignments) {
    const first = requiredDomain(values, assignment.inputs[0]!, assignment.id);
    let output: Gemma4LiteralValueDomain;
    switch (assignment.operation) {
      case "pixel-affine": case "add": case "rms-norm": case "gelu-tanh": case "multiply": case "masked-softmax": case "scale-f32": output = withAssignmentPolicy(cloneDomain(first), assignment); break;
      case "linear": case "clipped-linear": output = withAssignmentPolicy(replaceLast(first, String(requireMatrix(assignment)[0]), policyOutput(assignment, program.visionProgram.runtimeDtype)), assignment); break;
      case "position-embedding-2d": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.visionProgram.runtimeDtype), ["VB", "VP", String(hidden)], ["batch", "patch", "hidden"]), assignment); break;
      case "reshape-heads": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.visionProgram.runtimeDtype), ["VB", String(heads), "VP", String(headDim)], ["batch", "head", "patch", "head_feature"]), assignment); break;
      case "multidimensional-rope": output = withAssignmentPolicy(cloneDomain(first), assignment); break;
      case "attention-score-matmul": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.visionProgram.runtimeDtype), ["VB", String(heads), "VP", "VP"], ["batch", "head", "query_patch", "key_patch"]), assignment); break;
      case "attention-value-matmul": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.visionProgram.runtimeDtype), ["VB", "VP", String(hidden)], ["batch", "patch", "hidden"]), assignment); break;
      case "pool-by-position": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.visionProgram.runtimeDtype), ["VB", "VPOOL", String(hidden)], ["batch", "pool_cell", "hidden"]), assignment); break;
      case "pool-valid-mask": output = withPolicy(tensor("BOOL", ["VB", "VPOOL"], ["batch", "pool_cell"]), exactPolicy("declared-input", "exact-comparison", "BOOL")); break;
      case "strip-padding": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.visionProgram.runtimeDtype), ["VVALID", String(program.visionProgram.textHiddenSize)], ["soft_token", "text_hidden"]), assignment); break;
      case "masked-scatter-image-features": output = withPolicy(cloneDomain(requiredDomain(values, assignment.inputs[0]!, assignment.id)), exactPolicy("declared-input", "exact-selection", first.dtype)); break;
    }
    values.set(assignment.output, output);
    result.push(entry("vision", assignment, output));
  }
  return result;
}

function buildAudioDomains(program: Gemma4CompositeProgram): Gemma4LiteralAssignmentDomain[] {
  const tower = program.audioProgram.tower;
  const hidden = tower.hiddenSize, heads = tower.attentionHeads, headDim = tower.headDim;
  const context = tower.attentionChunkSize + tower.attentionContextLeft - 1 + tower.attentionContextRight;
  const relative = Math.floor(context / 2) + 1;
  const values = new Map<string, Gemma4LiteralValueDomain>([
    ["input_features", tensor("F32", ["AB", "AT0", "AF0"], ["batch", "frame", "feature"])],
    ["input_features_mask", tensor("BOOL", ["AB", "AT0"], ["batch", "frame"])],
    ["text_embeddings", tensor("BF16", ["B", "S", String(program.contract.text.hiddenSize)], ["batch", "sequence", "hidden"])],
    ["input_ids", tensor("I32", ["B", "S"], ["batch", "sequence"])],
  ]);
  const result: Gemma4LiteralAssignmentDomain[] = [];
  let convStage = 0, reshapeStage = 0, subsampleMaskStage = 0;
  for (const assignment of program.audioProgram.assignments) {
    const first = requiredDomain(values, assignment.inputs[0]!, assignment.id);
    let output: Gemma4LiteralValueDomain;
    switch (assignment.operation) {
      case "mask-input-features": case "layer-norm-channels": case "relu": case "rms-norm": case "clip": case "silu": case "scale-f32": case "add": case "per-dim-softplus-scale": case "attention-logit-add": case "attention-softcap": case "chunked-attention-mask": case "chunked-relative-attention-softmax": case "cast-bf16": output = withAssignmentPolicy(cloneDomain(first), assignment); break;
      case "reshape-conv-features": {
        const reshaped = reshapeStage++ === 0
          ? tensor("F32", ["AB", "1", "AT0", "AF0"], ["batch", "channel", "frame", "feature"])
          : tensor(policyOutput(assignment, program.audioProgram.runtimeDtype), ["AB", "AT2", `${requireInputWidth(first)}*AF2`], ["batch", "frame", "flattened_channel_feature"]);
        output = withPolicy(reshaped, exactPolicy(first.dtype, "row-major-alias", reshaped.dtype));
        break;
      }
      case "conv2d-stride2": {
        const matrix = assignment.tensors?.[0];
        const stage = convStage++;
        output = withAssignmentPolicy(tensor(policyOutput(assignment, program.audioProgram.runtimeDtype), ["AB", String(matrix?.shape[0]), stage === 0 ? "AT1" : "AT2", stage === 0 ? "AF1" : "AF2"], ["batch", "channel", "frame", "feature"]), assignment);
        break;
      }
      case "subsample-mask": output = withPolicy(tensor("BOOL", ["AB", subsampleMaskStage++ === 0 ? "AT1" : "AT2"], ["batch", "frame"]), exactPolicy("BOOL", "exact-boolean-subsample", "BOOL")); break;
      case "linear": case "clipped-linear": output = withAssignmentPolicy(replaceLast(first, String(requireMatrix(assignment)[0]), policyOutput(assignment, program.audioProgram.runtimeDtype)), assignment); break;
      case "relative-position-encoding": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.audioProgram.runtimeDtype), ["1", String(relative), String(hidden)], ["singleton_batch", "relative_position", "hidden"]), assignment); break;
      case "chunked-attention-content-matmul": case "relative-attention-shift": output = withAssignmentPolicy(tensor(policyOutput(assignment, "F32"), ["AB", String(heads), "ABLOCKS", String(tower.attentionChunkSize), String(context)], ["batch", "head", "block", "query_in_block", "key_slot"]), assignment); break;
      case "relative-attention-position-matmul": output = withAssignmentPolicy(tensor(policyOutput(assignment, "F32"), ["AB", String(heads), "ABLOCKS", String(tower.attentionChunkSize), String(relative)], ["batch", "head", "block", "query_in_block", "relative_position"]), assignment); break;
      case "chunked-relative-attention-values": output = withAssignmentPolicy(tensor(policyOutput(assignment, "F32"), ["AB", "AT2", String(hidden)], ["batch", "frame", "hidden"]), assignment); break;
      case "split-gated-linear-unit": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.audioProgram.runtimeDtype), ["AB", "AT2", String(hidden)], ["batch", "frame", "hidden"]), assignment); break;
      case "causal-depthwise-convolution": output = withAssignmentPolicy(cloneDomain(first), assignment); break;
      case "strip-padding": output = withAssignmentPolicy(tensor(policyOutput(assignment, program.audioProgram.runtimeDtype), ["AVALID", String(program.audioProgram.textHiddenSize)], ["soft_token", "text_hidden"]), assignment); break;
      case "masked-scatter-audio-features": output = withPolicy(cloneDomain(requiredDomain(values, assignment.inputs[0]!, assignment.id)), exactPolicy("declared-input", "exact-selection", first.dtype)); break;
    }
    values.set(assignment.output, output);
    result.push(entry("audio", assignment, output));
  }
  return result;
}

function buildTextDomains(program: Gemma4CompositeProgram): Gemma4LiteralAssignmentDomain[] {
  const architecture = program.textProgram.architecture;
  const values = new Map<string, Gemma4LiteralValueDomain>([
    ["input_ids", tensor("I32", ["B", "S"], ["batch", "sequence"])],
    ["hidden_states_0", tensor("BF16", ["B", "S", String(architecture.hiddenSize)], ["batch", "sequence", "hidden"])],
    ["ple_inputs", tensor("BF16", ["B", "S", String(architecture.numLayers), String(program.contract.text.perLayerInputSize)], ["batch", "sequence", "layer", "ple_feature"])],
    ["position_ids", tensor("I32", ["B", "S"], ["batch", "sequence"])],
    ["attention_mask:full", tensor("F32", ["B", "1", "S", "K"], ["batch", "mask_head", "query", "key"])],
    ["attention_mask:sliding", tensor("F32", ["B", "1", "S", "K"], ["batch", "mask_head", "query", "key"])],
  ]);
  const operations = [
    ...program.textProgram.prelude.map((operation) => ({ operation, scope: "text-prelude" as const })),
    ...program.textProgram.layers.flatMap((layer) => layer.operations.map((operation) => ({ operation, scope: "text-layer" as const }))),
    ...program.textProgram.epilogue.map((operation) => ({ operation, scope: "text-epilogue" as const })),
  ];
  const result: Gemma4LiteralAssignmentDomain[] = [];
  for (const { operation, scope } of operations) {
    const firstInput = textInputs(operation)[0];
    const first = requiredDomain(values, firstInput, operation.id);
    let output: Gemma4LiteralValueDomain;
    switch (operation.op) {
      case "rms_norm": case "rotary_embedding": case "activation": case "elementwise": case "tensor_scale": output = withPolicy(cloneDomain(first), operation.dtypePolicy); break;
      case "linear": output = withPolicy(replaceLast(first, String(operation.outFeatures), policyOutput(operation, "F32")), operation.dtypePolicy); break;
      case "reshape_heads": output = withPolicy(tensor(policyOutput(operation, "F32"), ["B", String(operation.numHeads), "S", String(operation.headDim)], ["batch", "head", "sequence", "head_feature"]), operation.dtypePolicy); break;
      case "scaled_dot_product_attention": output = withPolicy(tensor(policyOutput(operation, "F32"), ["B", "S", String(operation.numAttentionHeads * operation.headDim)], ["batch", "sequence", "attention_hidden"]), operation.dtypePolicy); break;
      case "reshape_per_layer": output = withPolicy(tensor(policyOutput(operation, "F32"), ["B", "S", String(operation.numLayers), String(operation.layerWidth)], ["batch", "sequence", "layer", "ple_feature"]), operation.dtypePolicy); break;
      case "select_per_layer": output = withPolicy(tensor(policyOutput(operation, "F32"), ["B", "S", String(operation.layerWidth)], ["batch", "sequence", "ple_feature"]), operation.dtypePolicy); break;
      case "embedding": output = withPolicy(tensor(policyOutput(operation, "F32"), ["B", "S", String(operation.weight.shape[1])], ["batch", "sequence", "hidden"]), operation.dtypePolicy); break;
      case "per_layer_embedding": output = withPolicy(tensor(policyOutput(operation, "F32"), ["B", "S", String(operation.numLayers), String(operation.layerWidth)], ["batch", "sequence", "layer", "ple_feature"]), operation.dtypePolicy); break;
    }
    values.set(operation.output, output);
    result.push({ definitionId: operation.id, operation: operation.op, scope, output: operation.output, domain: output });
  }
  return result;
}

function textInputs(operation: Operation): string[] {
  switch (operation.op) {
    case "embedding": case "per_layer_embedding": return [operation.tokenInput];
    case "rms_norm": case "linear": case "reshape_heads": case "reshape_per_layer": case "select_per_layer": case "activation": case "tensor_scale": return [operation.input];
    case "rotary_embedding": return [operation.input, operation.positionInput];
    case "scaled_dot_product_attention": return [operation.query, operation.key, operation.value, operation.maskInput];
    case "elementwise": return operation.inputs;
  }
}

function entry(scope: Gemma4LiteralCalculationScope, assignment: Assignment, domain: Gemma4LiteralValueDomain): Gemma4LiteralAssignmentDomain {
  return { definitionId: assignment.id, operation: assignment.operation, scope, output: assignment.output, domain };
}

function tensor(dtype: string, shape: string[], names: string[]): Gemma4LiteralValueDomain {
  return valueDomain("tensor", dtype, shape, names, "row-major");
}

function tuple(dtype: string, shape: string[], names: string[]): Gemma4LiteralValueDomain {
  return valueDomain("tuple", dtype, shape, names, "tuple-of-row-major-tensors");
}

function valueDomain(structure: "tensor" | "tuple", dtype: string, shape: string[], names: string[], layout: Gemma4LiteralValueDomain["layout"]): Gemma4LiteralValueDomain {
  if (shape.length !== names.length) throw new Error("Domínio literal Gemma 4 recebeu nomes de eixos incompatíveis.");
  return {
    structure,
    dtype,
    shape,
    layout,
    axes: shape.map((size, axis) => ({ axis, name: names[axis]!, size, indexDomain: `0 <= ${names[axis]} < ${size}` })),
  };
}

function cloneDomain(domain: Gemma4LiteralValueDomain): Gemma4LiteralValueDomain { return structuredClone(domain); }
function withPolicy(domain: Gemma4LiteralValueDomain, policy: DtypePolicy | undefined): Gemma4LiteralValueDomain { return policy ? { ...domain, dtype: policy.outputDtype ?? domain.dtype, dtypePolicy: structuredClone(policy) } : domain; }
function withAssignmentPolicy(domain: Gemma4LiteralValueDomain, assignment: Assignment): Gemma4LiteralValueDomain { return withPolicy(domain, "dtypePolicy" in assignment ? assignment.dtypePolicy : undefined); }

function replaceLast(domain: Gemma4LiteralValueDomain, size: string, dtype: string): Gemma4LiteralValueDomain {
  if (domain.structure !== "tensor" || domain.shape.length === 0) throw new Error("Linear Gemma 4 requer domínio tensorial não escalar.");
  const shape = [...domain.shape]; shape[shape.length - 1] = size;
  const names = domain.axes.map((axis) => axis.name); names[names.length - 1] = "output_feature";
  return tensor(dtype, shape, names);
}

function requiredDomain(values: ReadonlyMap<string, Gemma4LiteralValueDomain>, name: string | undefined, id: string): Gemma4LiteralValueDomain {
  const domain = name ? values.get(name) : undefined;
  if (!domain) throw new Error(`${id}: domínio literal do input ${name ?? "ausente"} não foi declarado antes do consumidor.`);
  return domain;
}

function requireMatrix(assignment: Assignment): number[] {
  const matrix = assignment.tensors?.find((tensor) => tensor.shape.length === 2);
  if (!matrix) throw new Error(`${assignment.id}: domínio linear requer matriz [out,in].`);
  return matrix.shape;
}

function requireInputWidth(domain: Gemma4LiteralValueDomain): string {
  if (domain.shape.length !== 4) throw new Error("Flatten de áudio requer tensor 4-D.");
  return domain.shape[1]!;
}

function policyOutput(value: object, fallback: string): string {
  return "dtypePolicy" in value ? (value as { dtypePolicy?: DtypePolicy }).dtypePolicy?.outputDtype ?? fallback : fallback;
}

function exactPolicy(inputDtype: string, computeDtype: string, outputDtype: string): DtypePolicy {
  return { inputDtype, computeDtype, accumulationDtype: "none", outputDtype };
}

function validateValueDomain(domain: Gemma4LiteralValueDomain, id: string): void {
  const policy = domain.dtypePolicy;
  if ((domain.structure !== "tensor" && domain.structure !== "tuple") || !domain.dtype || domain.shape.length === 0 || domain.axes.length !== domain.shape.length ||
    domain.shape.some((size) => !size) || domain.axes.some((axis, index) => axis.axis !== index || !axis.name || axis.size !== domain.shape[index] || !axis.indexDomain) ||
    !policy || !policy.inputDtype || !policy.computeDtype || !policy.accumulationDtype || policy.outputDtype !== domain.dtype) {
    throw new Error(`${id}: domínio literal de valor inválido.`);
  }
}
