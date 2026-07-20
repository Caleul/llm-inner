import type { Gemma4AudioProgram } from "./gemma4-audio.js";
import type { Gemma4VisionProgram } from "./gemma4-vision.js";
import type { DenseF32Tensor } from "./types.js";

export type Gemma4RuntimeReductionRequest =
  | {
    scope: "vision";
    operationId: string;
    operation: "attention-score-matmul";
    program: Gemma4VisionProgram;
    operands: readonly [DenseF32Tensor, DenseF32Tensor];
  }
  | {
    scope: "vision";
    operationId: string;
    operation: "attention-value-matmul";
    program: Gemma4VisionProgram;
    operands: readonly [DenseF32Tensor, DenseF32Tensor];
  }
  | {
    scope: "audio";
    operationId: string;
    operation: "chunked-attention-content-matmul" | "relative-attention-position-matmul" | "chunked-relative-attention-values";
    program: Gemma4AudioProgram;
    operands: readonly [DenseF32Tensor, DenseF32Tensor];
  };

/**
 * Executes only a serialized runtime-defined reduction. Implementations do
 * not receive a checkpoint path, tensor catalog, weights, or trace outputs.
 */
export interface Gemma4RuntimeReductionProvider {
  readonly contractId: "torch-2.12.1-cpu-inference-matmul-v1";
  execute(request: Gemma4RuntimeReductionRequest): DenseF32Tensor;
}

export function executeGemma4RuntimeReduction(
  provider: Gemma4RuntimeReductionProvider,
  request: Gemma4RuntimeReductionRequest,
  expectedShape: readonly number[],
): DenseF32Tensor {
  if (provider.contractId !== "torch-2.12.1-cpu-inference-matmul-v1") {
    throw new Error(`${request.operationId}: provedor de redução Gemma 4 não corresponde ao contrato fixado.`);
  }
  const result = provider.execute(request);
  const elements = expectedShape.reduce((total, dimension) => total * dimension, 1);
  if (result.shape.length !== expectedShape.length || result.shape.some((dimension, index) => dimension !== expectedShape[index]) ||
    result.values.length !== elements || result.values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${request.operationId}: provedor de redução Gemma 4 retornou tensor inválido ou shape divergente.`);
  }
  return result;
}
