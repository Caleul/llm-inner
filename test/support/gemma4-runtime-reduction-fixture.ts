import type { Gemma4VisionProgram } from "../../src/gemma4-vision.js";
import type { Gemma4AudioProgram } from "../../src/gemma4-audio.js";

export function fixtureProgram(): Gemma4VisionProgram {
  return {
    kind: "gemma4-vision-features",
    sourceFormat: "safetensors",
    tower: { attentionHeads: 1, headDim: 2 } as Gemma4VisionProgram["tower"],
    textHiddenSize: 2,
    rmsNormEpsilon: 1e-6,
    runtimeDtype: "BF16",
    assignments: [
      { id: "vision_layer_0_q_rope", operation: "multidimensional-rope", inputs: ["q"], output: "vision_layer_0_q_rotated" },
      { id: "vision_layer_0_k_rope", operation: "multidimensional-rope", inputs: ["k"], output: "vision_layer_0_k_rotated" },
      {
        id: "vision_layer_0_attention_scores",
        operation: "attention-score-matmul",
        inputs: ["vision_layer_0_q_rotated", "vision_layer_0_k_rotated"],
        output: "vision_layer_0_attention_scores",
        dtypePolicy: { inputDtype: "BF16", computeDtype: "pytorch-native-batched-matmul", accumulationDtype: "runtime-defined", outputDtype: "BF16" },
      },
    ],
    output: "image_features",
  };
}

export function fixtureAudioProgram(): Gemma4AudioProgram {
  const dtypePolicy = { inputDtype: "F32", computeDtype: "pytorch-cpu-f32-matmul", accumulationDtype: "runtime-defined", outputDtype: "F32" } as const;
  return {
    kind: "gemma4-audio-features",
    sourceFormat: "safetensors",
    tower: { attentionHeads: 1, headDim: 2, attentionChunkSize: 2, attentionContextLeft: 1, attentionContextRight: 0 } as Gemma4AudioProgram["tower"],
    textHiddenSize: 2,
    runtimeDtype: "BF16",
    attentionMaskContract: "transformers-eager-additive-mask-logical-not-v1",
    assignments: [
      { id: "audio_content", operation: "chunked-attention-content-matmul", inputs: ["q", "k"], output: "content", dtypePolicy },
      { id: "audio_position", operation: "relative-attention-position-matmul", inputs: ["q", "relative"], output: "position", dtypePolicy },
      { id: "audio_value", operation: "chunked-relative-attention-values", inputs: ["weights", "v"], output: "value", dtypePolicy },
    ],
    output: "audio_features",
  } as Gemma4AudioProgram;
}

export function sampleTensor(shape: number[], value = 0) {
  return { shape, values: new Float32Array(shape.reduce((total, dimension) => total * dimension, 1)).fill(value) };
}

export function tensorValues(shape: number[], values: number[]) {
  return { shape, values: Float32Array.from(values) };
}
