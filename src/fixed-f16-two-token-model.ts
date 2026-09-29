import { type SafetensorsCatalogReader } from "./safetensors.js";
import { compileFixedTwoTokenAttention, evaluateFixedTwoTokenAttention, type FixedTwoTokenAttentionProgram } from "./fixed-f16-attention-program.js";
import { addF16Bits, compileFixedMlp, evaluateFixedFourLaneDenseHead, evaluateFixedMlp, readFixedF16Vector, rmsNormF16, type FixedMlpProgram } from "./fixed-f16-layer-ops.js";

export interface FixedLayerProgram {
  attention: FixedTwoTokenAttentionProgram;
  inputNorm: number[];
  postAttentionNorm: number[];
  mlp: FixedMlpProgram;
}
export interface FixedTwoTokenModel {
  kind: "fixed-two-token-model";
  embeddingBase64: string;
  headBase64: string;
  vocabSize: number;
  layers: [FixedLayerProgram, FixedLayerProgram];
  finalNorm: number[];
}
const decoded = new WeakMap<FixedTwoTokenModel, { embedding: Buffer; head: Buffer }>();
function weights(model: FixedTwoTokenModel): { embedding: Buffer; head: Buffer } {
  let value = decoded.get(model);
  if (!value) {
    value = { embedding: Buffer.from(model.embeddingBase64, "base64"), head: Buffer.from(model.headBase64, "base64") };
    if (value.embedding.length !== model.vocabSize * 32 || value.head.length !== model.vocabSize * 32) throw new Error("Payload denso F16 inválido.");
    decoded.set(model, value);
  }
  return value;
}

export async function compileFixedTwoTokenModel(
  reader: SafetensorsCatalogReader, cos: readonly (readonly number[])[], sin: readonly (readonly number[])[],
): Promise<FixedTwoTokenModel> {
  const catalog = await reader.inspect();
  const embeddingTensor = catalog.tensors.get("model.embed_tokens.weight");
  const headTensor = catalog.tensors.get("lm_head.weight");
  if (!embeddingTensor || embeddingTensor.storageDtype !== "F16" ||
    embeddingTensor.logicalShape.length !== 2 || embeddingTensor.logicalShape[1] !== 16) {
    throw new Error("Embedding F16 esperado com hidden_size=16.");
  }
  if (!headTensor || headTensor.storageDtype !== "F16" || headTensor.logicalShape.length !== 2 ||
    headTensor.logicalShape[0] !== embeddingTensor.logicalShape[0] || headTensor.logicalShape[1] !== 16) {
    throw new Error("lm_head F16 incompatível com embedding.");
  }
  const embeddingBase64 = (await reader.readTensorBytes(embeddingTensor)).toString("base64");
  const headBase64 = (await reader.readTensorBytes(headTensor)).toString("base64");
  const layers = [];
  for (const layer of [0, 1]) {
    layers.push({
      attention: await compileFixedTwoTokenAttention(reader, layer, cos, sin),
      inputNorm: await readFixedF16Vector(reader, `model.layers.${layer}.input_layernorm.weight`, 16),
      postAttentionNorm: await readFixedF16Vector(reader, `model.layers.${layer}.post_attention_layernorm.weight`, 16),
      mlp: await compileFixedMlp(reader, layer),
    });
  }
  return { kind: "fixed-two-token-model", embeddingBase64, headBase64, vocabSize: embeddingTensor.logicalShape[0]!,
    layers: layers as [FixedLayerProgram, FixedLayerProgram],
    finalNorm: await readFixedF16Vector(reader, "model.norm.weight", 16),
  };
}

export interface FixedModelResult { layers: number[][]; hidden: number[]; logits: number[]; tokens: number[] }
/** The only runtime inputs are two token IDs; all hidden states are calculated internally. */
export function evaluateFixedTwoTokenModel(model: FixedTwoTokenModel, tokenIds: readonly number[]): FixedModelResult {
  if (tokenIds.length !== 2 || tokenIds.some((id) => !Number.isInteger(id) || id < 0 || id >= model.vocabSize)) {
    throw new Error("Dois IDs de token válidos são necessários.");
  }
  const payload = weights(model);
  let hidden = tokenIds.flatMap((id) => Array.from({ length: 16 }, (_, dim) => payload.embedding.readUInt16LE(id * 32 + dim * 2)));
  const layers: number[][] = [];
  for (const layer of model.layers) {
    const normalized = [0, 1].flatMap((token) => rmsNormF16(hidden.slice(token * 16, token * 16 + 16), layer.inputNorm));
    const attention = evaluateFixedTwoTokenAttention(layer.attention, normalized);
    const residual = hidden.map((value, index) => addF16Bits(value, attention[index]!));
    const mlpInput = [0, 1].flatMap((token) => rmsNormF16(residual.slice(token * 16, token * 16 + 16), layer.postAttentionNorm));
    const mlpOutput = [0, 1].flatMap((token) => evaluateFixedMlp(layer.mlp, mlpInput.slice(token * 16, token * 16 + 16)));
    hidden = residual.map((value, index) => addF16Bits(value, mlpOutput[index]!));
    layers.push(hidden);
  }
  hidden = [0, 1].flatMap((token) => rmsNormF16(hidden.slice(token * 16, token * 16 + 16), model.finalNorm));
  const logits = [0, 1].flatMap((token) => evaluateFixedFourLaneDenseHead(payload.head, model.vocabSize, hidden.slice(token * 16, token * 16 + 16)));
  const tokens = [0, 1].map((token) => {
    let best = 0;
    for (let index = 1; index < model.vocabSize; index++) {
      // Finite F16 bit ordering is not numeric ordering for negative values.
      const score = (bits: number) => bits & 0x8000 ? -(bits & 0x7fff) : bits;
      if (score(logits[token * model.vocabSize + index]!) > score(logits[token * model.vocabSize + best]!)) best = index;
    }
    return best;
  });
  return { layers, hidden, logits, tokens };
}
