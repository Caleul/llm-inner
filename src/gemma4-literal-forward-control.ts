import { isDeepStrictEqual } from "node:util";
import type {
  Gemma4CompositeExecutionRequest,
  Gemma4CompositeForwardSelection,
  Gemma4CompositeProgram,
} from "./gemma4-composite.js";

export type Gemma4LiteralModality = "image" | "video" | "audio";

export interface Gemma4LiteralForwardModalityBranch {
  modality: Gemma4LiteralModality;
  order: number;
  inputGroup: string[];
  presenceRule: "all-present-or-all-absent";
  activeOperationIds: string[];
  activeInvocationId: string;
  inactiveIdentity: {
    output: string;
    input: string;
    operation: "identity-alias-no-cast";
  };
}

export interface Gemma4LiteralForwardControlProgram {
  kind: "gemma4-literal-forward-control-program";
  schemaVersion: 1;
  requiredInputs: ["input_ids"];
  executionOrder: {
    beforeModalities: string[];
    modalities: ["image", "video", "audio"];
    afterModalities: string[];
    textCoreOperationId: "composite_text_core";
    terminalOperationId: string;
  };
  modalityBranches: Gemma4LiteralForwardModalityBranch[];
  attentionMasks: {
    selectorInput: "mm_token_type_ids";
    present: {
      mode: "serialized-vision-block-masks";
      operationIds: ["composite_block_sequence_ids", "composite_full_attention_mask", "composite_sliding_attention_mask"];
      forbiddenInputs: ["attention_mask", "past_key_values"];
    };
    absent: {
      callerMaskInput: "attention_mask";
      callerMaskMode: "caller-additive-mask";
      defaultMode: "text-program-causal-or-incremental-cache";
    };
  };
  textState: {
    positionIds: {
      input: "position_ids";
      present: "caller-i32-values";
      absentScalarAssignment: "position_ids[batch,sequence]=I32(sequence)";
    };
    pastKeyValues: {
      input: "past_key_values";
      present: "caller-post-rope-producer-cache";
      absent: "empty-cache";
    };
  };
}

export interface Gemma4LiteralForwardControlResult extends Gemma4CompositeForwardSelection {
  activeTopLevelOperationIds: string[];
  inactiveIdentityAssignments: Array<Gemma4LiteralForwardModalityBranch["inactiveIdentity"]>;
  attentionMaskMode: "serialized-vision-block-masks" | "caller-additive-mask" | "text-program-causal-or-incremental-cache";
}

/**
 * Serializes the optional-input branch semantics which were previously only
 * expressed by `if` statements in the composite host executor. The resulting
 * program is deliberately finite and names the exact top-level assignments
 * and identity aliases selected when a modality is absent.
 */
export function buildGemma4LiteralForwardControlProgram(
  program: Gemma4CompositeProgram,
): Gemma4LiteralForwardControlProgram {
  const operationIds = program.assignments.map((assignment) => assignment.id);
  const required = <T extends string>(id: T): T => {
    if (!operationIds.includes(id)) throw new Error(`Controle forward Gemma 4 requer atribuição ${id}.`);
    return id;
  };
  const terminalOperationId = program.textProgram.epilogue.at(-1)?.id;
  if (terminalOperationId !== "lm_head" && terminalOperationId !== "final_logit_softcap") {
    throw new Error("Controle forward Gemma 4 requer lm_head ou final_logit_softcap terminal.");
  }
  return {
    kind: "gemma4-literal-forward-control-program",
    schemaVersion: 1,
    requiredInputs: ["input_ids"],
    executionOrder: {
      beforeModalities: [
        required("composite_placeholder_masks"),
        required("composite_pad_substitution"),
        required("composite_text_embedding"),
        required("composite_ple_identity"),
      ],
      modalities: ["image", "video", "audio"],
      afterModalities: [
        required("composite_ple_context_projection"),
        required("composite_ple_context_scale"),
        required("composite_ple_context_reshape"),
        required("composite_ple_context_norm"),
        required("composite_ple_combine"),
        required("composite_ple_combine_scale"),
      ],
      textCoreOperationId: required("composite_text_core"),
      terminalOperationId,
    },
    modalityBranches: [
      {
        modality: "image",
        order: 0,
        inputGroup: ["pixel_values", "image_position_ids"],
        presenceRule: "all-present-or-all-absent",
        activeOperationIds: [required("composite_image_features"), required("composite_image_scatter")],
        activeInvocationId: "composite_image_features",
        inactiveIdentity: {
          output: "composite_embeddings_after_image",
          input: "composite_text_embeddings",
          operation: "identity-alias-no-cast",
        },
      },
      {
        modality: "video",
        order: 1,
        inputGroup: ["pixel_values_videos", "video_position_ids"],
        presenceRule: "all-present-or-all-absent",
        activeOperationIds: [
          required("composite_video_pixel_flatten"),
          required("composite_video_position_flatten"),
          required("composite_video_features"),
          required("composite_video_scatter"),
        ],
        activeInvocationId: "composite_video_features",
        inactiveIdentity: {
          output: "composite_embeddings_after_video",
          input: "composite_embeddings_after_image",
          operation: "identity-alias-no-cast",
        },
      },
      {
        modality: "audio",
        order: 2,
        inputGroup: ["input_features", "input_features_mask"],
        presenceRule: "all-present-or-all-absent",
        activeOperationIds: [required("composite_audio_features"), required("composite_audio_scatter")],
        activeInvocationId: "composite_audio_features",
        inactiveIdentity: {
          output: "hidden_states_0",
          input: "composite_embeddings_after_video",
          operation: "identity-alias-no-cast",
        },
      },
    ],
    attentionMasks: {
      selectorInput: "mm_token_type_ids",
      present: {
        mode: "serialized-vision-block-masks",
        operationIds: [
          required("composite_block_sequence_ids"),
          required("composite_full_attention_mask"),
          required("composite_sliding_attention_mask"),
        ],
        forbiddenInputs: ["attention_mask", "past_key_values"],
      },
      absent: {
        callerMaskInput: "attention_mask",
        callerMaskMode: "caller-additive-mask",
        defaultMode: "text-program-causal-or-incremental-cache",
      },
    },
    textState: {
      positionIds: {
        input: "position_ids",
        present: "caller-i32-values",
        absentScalarAssignment: "position_ids[batch,sequence]=I32(sequence)",
      },
      pastKeyValues: {
        input: "past_key_values",
        present: "caller-post-rope-producer-cache",
        absent: "empty-cache",
      },
    },
  };
}

export function validateGemma4LiteralForwardControlProgram(
  control: Gemma4LiteralForwardControlProgram,
  program: Gemma4CompositeProgram,
): void {
  if (!isDeepStrictEqual(control, buildGemma4LiteralForwardControlProgram(program))) {
    throw new Error("Programa literal Gemma 4 possui controle forward, roteamento modal ou aliases de ausência divergentes.");
  }
}

/** Executes only artifact data: input presence selects serialized branches. */
export function executeGemma4LiteralForwardControlProgram(
  control: Gemma4LiteralForwardControlProgram,
  program: Gemma4CompositeProgram,
  presentInputs: ReadonlySet<string>,
): Gemma4LiteralForwardControlResult {
  validateGemma4LiteralForwardControlProgram(control, program);
  for (const required of control.requiredInputs) {
    if (!presentInputs.has(required)) throw new Error(`Controle forward Gemma 4 requer input declarado ${required}.`);
  }
  const activeTopLevelOperationIds = [...control.executionOrder.beforeModalities];
  const inactiveIdentityAssignments: Gemma4LiteralForwardControlResult["inactiveIdentityAssignments"] = [];
  const modalities = { image: false, video: false, audio: false };
  for (const branch of control.modalityBranches) {
    const count = branch.inputGroup.filter((input) => presentInputs.has(input)).length;
    if (count !== 0 && count !== branch.inputGroup.length) {
      throw new Error(`Controle forward Gemma 4 requer ${branch.inputGroup.join(" e ")} juntos, ou todos ausentes.`);
    }
    const active = count === branch.inputGroup.length;
    modalities[branch.modality] = active;
    if (active) activeTopLevelOperationIds.push(...branch.activeOperationIds);
    else inactiveIdentityAssignments.push(structuredClone(branch.inactiveIdentity));
  }
  const visionMasks = presentInputs.has(control.attentionMasks.selectorInput);
  activeTopLevelOperationIds.push(...control.executionOrder.afterModalities);
  if (visionMasks) {
    const forbidden = control.attentionMasks.present.forbiddenInputs.find((input) => presentInputs.has(input));
    if (forbidden) {
      throw new Error(`Controle forward Gemma 4 não combina ${control.attentionMasks.selectorInput} com ${forbidden}.`);
    }
    activeTopLevelOperationIds.push(...control.attentionMasks.present.operationIds);
  }
  activeTopLevelOperationIds.push(control.executionOrder.textCoreOperationId);
  return {
    modalities,
    visionMasks,
    activeTopLevelOperationIds,
    inactiveIdentityAssignments,
    attentionMaskMode: visionMasks
      ? control.attentionMasks.present.mode
      : presentInputs.has(control.attentionMasks.absent.callerMaskInput)
        ? control.attentionMasks.absent.callerMaskMode
        : control.attentionMasks.absent.defaultMode,
    positionIdsMode: presentInputs.has(control.textState.positionIds.input)
      ? control.textState.positionIds.present
      : "sequence-index-default",
    pastKeyValuesMode: presentInputs.has(control.textState.pastKeyValues.input)
      ? control.textState.pastKeyValues.present
      : control.textState.pastKeyValues.absent,
  };
}

/** Maps one host request to the artifact's canonical snake-case input names. */
export function gemma4CompositeRequestInputPresence(
  request: Omit<Gemma4CompositeExecutionRequest, "tensors">,
): ReadonlySet<string> {
  const inputs = new Set<string>(["input_ids"]);
  const optional: Array<[string, unknown]> = [
    ["position_ids", request.positionIds],
    ["attention_mask", request.attentionMask],
    ["past_key_values", request.pastKeyValues],
    ["pixel_values", request.pixelValues],
    ["image_position_ids", request.imagePositionIds],
    ["pixel_values_videos", request.pixelValuesVideos],
    ["video_position_ids", request.videoPositionIds],
    ["input_features", request.inputFeatures],
    ["input_features_mask", request.inputFeaturesMask],
    ["mm_token_type_ids", request.mmTokenTypeIds],
  ];
  for (const [name, value] of optional) if (value !== undefined) inputs.add(name);
  return inputs;
}
