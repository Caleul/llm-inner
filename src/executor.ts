import type {
  DenseTensor,
  DenseF32Tensor,
  ModelIR,
  Operation,
  ReferenceF32KeyValueCache,
  ReferenceF32ExecutionRequest,
  ReferenceF32ExecutionResult,
  ReferenceKeyValueCache,
  ReferenceExecutionRequest,
  ReferenceExecutionResult,
  TensorRef,
} from "./types.js";

/**
 * Small, deterministic F64 interpreter for the dense decoder subset emitted by
 * the architecture adapters. It intentionally rejects implicit dtype policies,
 * cache inputs, quantized tensors, and RoPE variants whose numerical contract
 * has not yet been encoded here.
 */
export function executeReferenceF64(
  ir: ModelIR,
  request: ReferenceExecutionRequest,
): ReferenceExecutionResult {
  const batch = request.inputIds.length;
  if (batch === 0 || request.inputIds.some((row) => row.length === 0)) {
    throw new Error("inputIds deve conter ao menos um token por batch.");
  }
  const sequence = request.inputIds[0]!.length;
  if (request.inputIds.some((row) => row.length !== sequence)) {
    throw new Error("O executor F64 requer sequências de mesmo comprimento no batch.");
  }
  const positions = request.positionIds ?? request.inputIds.map((row) => row.map((_, index) => index));
  if (positions.length !== batch || positions.some((row) => row.length !== sequence)) {
    throw new Error("positionIds deve ter o mesmo shape de inputIds.");
  }

  const values = new Map<string, DenseTensor>();
  const pastKeyValues = new Map<number, ReferenceKeyValueCache>();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    assertF64Policy(operation);
    switch (operation.op) {
      case "embedding":
        values.set(operation.output, embedding(request.inputIds, tensor(request, operation.weight), operation.scale));
        break;
      case "rms_norm":
        values.set(operation.output, rmsNorm(value(values, operation.input), tensor(request, operation.weight), operation));
        break;
      case "linear":
        if (!operation.transposeWeight) throw new Error(`${operation.id}: executor F64 requer weight no layout [out,in].`);
        values.set(operation.output, linear(value(values, operation.input), tensor(request, operation.weight), operation.bias ? tensor(request, operation.bias) : undefined));
        break;
      case "reshape_heads":
        if (operation.layout !== "BHSD") throw new Error(`${operation.id}: executor F64 requer layout BHSD.`);
        values.set(operation.output, reshapeHeads(value(values, operation.input), operation.numHeads, operation.headDim));
        break;
      case "rotary_embedding":
        values.set(operation.output, rotary(value(values, operation.input), positions, operation));
        break;
      case "scaled_dot_product_attention":
        if (operation.layer === undefined) throw new Error(`${operation.id}: attention sem índice de camada não pode usar cache KV.`);
        if (operation.kvSharing) {
          if (request.pastKeyValues) throw new Error(`${operation.id}: cache KV com compartilhamento entre camadas ainda não é suportado.`);
          values.set(operation.output, attention(value(values, operation.query), value(values, operation.key), value(values, operation.value), operation, request.attentionMask));
          break;
        }
        {
          const currentKey = value(values, operation.key);
          const currentValue = value(values, operation.value);
          const cached = cacheForLayer(request.pastKeyValues, operation.layer, operation, currentKey, currentValue);
          const key = cached ? concatSequence(cached.key, currentKey) : currentKey;
          const valueTensor = cached ? concatSequence(cached.value, currentValue) : currentValue;
          values.set(operation.output, attention(value(values, operation.query), key, valueTensor, operation, request.attentionMask, cached?.key.shape[2] ?? 0));
          pastKeyValues.set(operation.layer, { key, value: valueTensor });
        }
        break;
      case "activation":
        values.set(operation.output, activation(value(values, operation.input), operation.function, operation.approximation));
        break;
      case "elementwise":
        values.set(operation.output, elementwise(operation.inputs.map((name) => value(values, name)), operation.kind, operation.scalar));
        break;
      default: {
        const neverOperation: never = operation;
        throw new Error(`Operação não suportada pelo executor F64: ${JSON.stringify(neverOperation)}`);
      }
    }
  }
  const logits = values.get("softcapped_logits") ?? values.get("logits");
  if (!logits) throw new Error("IR não produziu logits.");
  return { values, logits, pastKeyValues };
}

/**
 * Deterministic scalar binary32 interpreter for the same dense decoder subset
 * as executeReferenceF64. Every stored value and arithmetic boundary is
 * rounded with Math.fround. Transcendentals use the host libm and are rounded
 * back to F32, so this is a declared scalar policy rather than a claim that it
 * is bitwise identical to a particular BLAS, GPU, or framework kernel.
 */
export function executeReferenceF32(
  ir: ModelIR,
  request: ReferenceF32ExecutionRequest,
): ReferenceF32ExecutionResult {
  const batch = request.inputIds.length;
  if (batch === 0 || request.inputIds.some((row) => row.length === 0)) {
    throw new Error("inputIds deve conter ao menos um token por batch.");
  }
  const sequence = request.inputIds[0]!.length;
  if (request.inputIds.some((row) => row.length !== sequence)) {
    throw new Error("O executor F32 requer sequências de mesmo comprimento no batch.");
  }
  const positions = request.positionIds ?? request.inputIds.map((row) => row.map((_, index) => index));
  if (positions.length !== batch || positions.some((row) => row.length !== sequence)) {
    throw new Error("positionIds deve ter o mesmo shape de inputIds.");
  }

  const values = new Map<string, DenseF32Tensor>();
  const pastKeyValues = new Map<number, ReferenceF32KeyValueCache>();
  for (const operation of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    assertF32Policy(operation);
    switch (operation.op) {
      case "embedding":
        values.set(operation.output, embeddingF32(request.inputIds, tensorF32(request, operation.weight), operation.scale));
        break;
      case "rms_norm":
        values.set(operation.output, rmsNormF32(valueF32(values, operation.input), tensorF32(request, operation.weight), operation));
        break;
      case "linear":
        if (!operation.transposeWeight) throw new Error(`${operation.id}: executor F32 requer weight no layout [out,in].`);
        values.set(operation.output, linearF32(valueF32(values, operation.input), tensorF32(request, operation.weight), operation.bias ? tensorF32(request, operation.bias) : undefined));
        break;
      case "reshape_heads":
        if (operation.layout !== "BHSD") throw new Error(`${operation.id}: executor F32 requer layout BHSD.`);
        values.set(operation.output, reshapeHeadsF32(valueF32(values, operation.input), operation.numHeads, operation.headDim));
        break;
      case "rotary_embedding":
        values.set(operation.output, rotaryF32(valueF32(values, operation.input), positions, operation));
        break;
      case "scaled_dot_product_attention":
        if (operation.layer === undefined) throw new Error(`${operation.id}: attention sem índice de camada não pode usar cache KV.`);
        if (operation.kvSharing) {
          if (request.pastKeyValues) throw new Error(`${operation.id}: cache KV com compartilhamento entre camadas ainda não é suportado.`);
          values.set(operation.output, attentionF32(valueF32(values, operation.query), valueF32(values, operation.key), valueF32(values, operation.value), operation, request.attentionMask));
          break;
        }
        {
          const currentKey = valueF32(values, operation.key);
          const currentValue = valueF32(values, operation.value);
          const cached = cacheForLayerF32(request.pastKeyValues, operation.layer, operation, currentKey, currentValue);
          const key = cached ? concatSequenceF32(cached.key, currentKey) : currentKey;
          const valueTensor = cached ? concatSequenceF32(cached.value, currentValue) : currentValue;
          values.set(operation.output, attentionF32(valueF32(values, operation.query), key, valueTensor, operation, request.attentionMask, cached?.key.shape[2] ?? 0));
          pastKeyValues.set(operation.layer, { key, value: valueTensor });
        }
        break;
      case "activation":
        values.set(operation.output, activationF32(valueF32(values, operation.input), operation.function, operation.approximation));
        break;
      case "elementwise":
        values.set(operation.output, elementwiseF32(operation.inputs.map((name) => valueF32(values, name)), operation.kind, operation.scalar));
        break;
      default: {
        const neverOperation: never = operation;
        throw new Error(`Operação não suportada pelo executor F32: ${JSON.stringify(neverOperation)}`);
      }
    }
  }
  const logits = values.get("softcapped_logits") ?? values.get("logits");
  if (!logits) throw new Error("IR não produziu logits.");
  return { values, logits, pastKeyValues };
}

function assertF64Policy(operation: Operation): void {
  const policy = operation.dtypePolicy;
  for (const field of ["computeDtype", "accumulationDtype", "outputDtype"] as const) {
    if (policy[field] !== "F64") {
      throw new Error(`${operation.id}: executor de referência requer política F64 explícita; ${field}=${policy[field] ?? "ausente"}.`);
    }
  }
  for (const [name, dtype] of Object.entries(policy)) {
    if (dtype !== undefined && dtype !== "F64") {
      throw new Error(`${operation.id}: executor de referência suporta somente política F64 explícita; ${name}=${dtype}.`);
    }
  }
  if (operation.op === "linear" && operation.weight.quantization) {
    throw new Error(`${operation.id}: executor F64 ainda não dequantiza ${operation.weight.quantization.family}.`);
  }
  if (operation.op === "scaled_dot_product_attention" && operation.softmaxComputeDtype !== "F64") {
    throw new Error(`${operation.id}: executor de referência suporta somente softmaxComputeDtype=F64.`);
  }
}

function assertF32Policy(operation: Operation): void {
  const policy = operation.dtypePolicy;
  for (const field of ["computeDtype", "accumulationDtype", "outputDtype"] as const) {
    if (policy[field] !== "F32") {
      throw new Error(`${operation.id}: executor de referência requer política F32 explícita; ${field}=${policy[field] ?? "ausente"}.`);
    }
  }
  for (const [name, dtype] of Object.entries(policy)) {
    if (dtype !== undefined && dtype !== "F32") {
      throw new Error(`${operation.id}: executor de referência suporta somente política F32 explícita; ${name}=${dtype}.`);
    }
  }
  if (operation.op === "linear" && operation.weight.quantization) {
    throw new Error(`${operation.id}: executor F32 ainda não dequantiza ${operation.weight.quantization.family}.`);
  }
  if (operation.op === "scaled_dot_product_attention" && operation.softmaxComputeDtype !== "F32") {
    throw new Error(`${operation.id}: executor de referência suporta somente softmaxComputeDtype=F32.`);
  }
}

function tensor(request: ReferenceExecutionRequest, reference: TensorRef): DenseTensor {
  if (reference.quantization) throw new Error(`${reference.name}: tensor quantizado não é suportado pelo executor F64.`);
  const found = request.tensors.get(reference.name);
  if (!found) throw new Error(`Tensor F64 ausente: ${reference.name}`);
  assertShape(found, reference.shape, reference.name);
  return found;
}

function tensorF32(request: ReferenceF32ExecutionRequest, reference: TensorRef): DenseF32Tensor {
  if (reference.quantization) throw new Error(`${reference.name}: tensor quantizado não é suportado pelo executor F32.`);
  const found = request.tensors.get(reference.name);
  if (!found) throw new Error(`Tensor F32 ausente: ${reference.name}`);
  assertShapeF32(found, reference.shape, reference.name);
  return found;
}

function value(values: Map<string, DenseTensor>, name: string): DenseTensor {
  const found = values.get(name);
  if (!found) throw new Error(`Valor intermediário ausente: ${name}`);
  return found;
}

function valueF32(values: Map<string, DenseF32Tensor>, name: string): DenseF32Tensor {
  const found = values.get(name);
  if (!found) throw new Error(`Valor intermediário ausente: ${name}`);
  return found;
}

function dense(shape: number[], values: number[] | Float64Array): DenseTensor {
  const data = values instanceof Float64Array ? values : Float64Array.from(values);
  const size = shape.reduce((total, dimension) => total * dimension, 1);
  if (data.length !== size) throw new Error(`Tensor inválido: shape ${shape.join("x")} requer ${size} valores, recebeu ${data.length}.`);
  return { shape, values: data };
}

function assertShape(actual: DenseTensor, expected: number[], label: string): void {
  if (actual.shape.length !== expected.length || actual.shape.some((dimension, index) => dimension !== expected[index])) {
    throw new Error(`${label}: shape esperado ${expected.join("x")}, recebeu ${actual.shape.join("x")}.`);
  }
}

function denseF32(shape: number[], values: Float32Array): DenseF32Tensor {
  const size = shape.reduce((total, dimension) => total * dimension, 1);
  if (values.length !== size) throw new Error(`Tensor F32 inválido: shape ${shape.join("x")} requer ${size} valores, recebeu ${values.length}.`);
  return { shape, values };
}

function assertShapeF32(actual: DenseF32Tensor, expected: number[], label: string): void {
  if (actual.shape.length !== expected.length || actual.shape.some((dimension, index) => dimension !== expected[index])) {
    throw new Error(`${label}: shape esperado ${expected.join("x")}, recebeu ${actual.shape.join("x")}.`);
  }
}

type AttentionMask = DenseTensor | DenseF32Tensor;

function validateAttentionMask<T extends AttentionMask>(
  mask: T | undefined,
  batch: number,
  heads: number,
  querySequence: number,
  keySequence: number,
  operationId: string,
): T | undefined {
  if (!mask) return undefined;
  const [maskBatch, maskHeads, maskQuery, maskKey] = mask.shape;
  if (
    mask.shape.length !== 4 ||
    maskBatch !== batch ||
    (maskHeads !== 1 && maskHeads !== heads) ||
    maskQuery !== querySequence ||
    maskKey !== keySequence
  ) {
    throw new Error(
      `${operationId}: attentionMask deve ter shape [${batch}, 1|${heads}, ${querySequence}, ${keySequence}], ` +
        `recebeu [${mask.shape.join(", ")}].`,
    );
  }
  for (const entry of mask.values) {
    if (Number.isNaN(entry) || entry === Infinity) {
      throw new Error(`${operationId}: attentionMask aceita somente valores finitos ou -Infinity.`);
    }
  }
  return mask;
}

function maskOffset(mask: AttentionMask, batch: number, head: number, query: number, key: number): number {
  const [, maskHeads, querySequence, keySequence] = mask.shape as [number, number, number, number];
  const selectedHead = maskHeads === 1 ? 0 : head;
  return ((batch * maskHeads + selectedHead) * querySequence + query) * keySequence + key;
}

function embedding(inputIds: number[][], weight: DenseTensor, scale?: number): DenseTensor {
  if (weight.shape.length !== 2) throw new Error("Embedding F64 requer weight 2D.");
  const [vocab, hidden] = weight.shape as [number, number];
  const result = new Float64Array(inputIds.length * inputIds[0]!.length * hidden);
  for (let batch = 0; batch < inputIds.length; batch += 1) for (let sequence = 0; sequence < inputIds[batch]!.length; sequence += 1) {
    const token = inputIds[batch]![sequence]!;
    if (!Number.isInteger(token) || token < 0 || token >= vocab) throw new Error(`Token fora do vocabulário: ${token}.`);
    const offset = (batch * inputIds[batch]!.length + sequence) * hidden;
    for (let index = 0; index < hidden; index += 1) result[offset + index] = weight.values[token * hidden + index]! * (scale ?? 1);
  }
  return dense([inputIds.length, inputIds[0]!.length, hidden], result);
}

function rmsNorm(input: DenseTensor, weight: DenseTensor, operation: Extract<Operation, { op: "rms_norm" }>): DenseTensor {
  const width = input.shape.at(-1);
  if (width === undefined || weight.shape.length !== 1 || weight.shape[0] !== width) throw new Error(`${operation.id}: RMSNorm incompatível.`);
  const result = new Float64Array(input.values.length);
  for (let offset = 0; offset < input.values.length; offset += width) {
    let sum = 0;
    for (let index = 0; index < width; index += 1) sum += input.values[offset + index]! ** 2;
    const scale = 1 / Math.sqrt(sum / width + operation.epsilon);
    for (let index = 0; index < width; index += 1) {
      const multiplier = operation.weightTransform === "one_plus_weight" ? 1 + weight.values[index]! : weight.values[index]!;
      result[offset + index] = input.values[offset + index]! * scale * multiplier;
    }
  }
  return dense([...input.shape], result);
}

function linear(input: DenseTensor, weight: DenseTensor, bias?: DenseTensor): DenseTensor {
  if (input.shape.length < 1 || weight.shape.length !== 2) throw new Error("Linear F64 requer entrada e weight válidos.");
  const features = input.shape.at(-1)!;
  const [outFeatures, inFeatures] = weight.shape as [number, number];
  if (features !== inFeatures) throw new Error(`Linear: entrada ${features} incompatível com weight ${outFeatures}x${inFeatures}.`);
  if (bias && (bias.shape.length !== 1 || bias.shape[0] !== outFeatures)) throw new Error("Linear: bias incompatível.");
  const rows = input.values.length / features;
  const result = new Float64Array(rows * outFeatures);
  for (let row = 0; row < rows; row += 1) for (let output = 0; output < outFeatures; output += 1) {
    let sum = bias?.values[output] ?? 0;
    for (let column = 0; column < inFeatures; column += 1) sum += input.values[row * inFeatures + column]! * weight.values[output * inFeatures + column]!;
    result[row * outFeatures + output] = sum;
  }
  return dense([...input.shape.slice(0, -1), outFeatures], result);
}

function reshapeHeads(input: DenseTensor, heads: number, headDim: number): DenseTensor {
  if (input.shape.length !== 3 || input.shape[2] !== heads * headDim) throw new Error("reshape_heads requer [B,S,H*D].");
  const [batch, sequence] = input.shape as [number, number, number];
  const result = new Float64Array(input.values.length);
  for (let b = 0; b < batch; b += 1) for (let s = 0; s < sequence; s += 1) for (let h = 0; h < heads; h += 1) for (let d = 0; d < headDim; d += 1) {
    result[((b * heads + h) * sequence + s) * headDim + d] = input.values[(b * sequence + s) * heads * headDim + h * headDim + d]!;
  }
  return dense([batch, heads, sequence, headDim], result);
}

function cacheForLayer(
  cache: ReadonlyMap<number, ReferenceKeyValueCache> | undefined,
  layer: number,
  operation: Extract<Operation, { op: "scaled_dot_product_attention" }>,
  currentKey: DenseTensor,
  currentValue: DenseTensor,
): ReferenceKeyValueCache | undefined {
  if (!cache) return undefined;
  const entry = cache.get(layer);
  if (!entry) throw new Error(`${operation.id}: pastKeyValues não contém a camada ${layer}.`);
  assertCacheEntry(entry.key, entry.value, currentKey, currentValue, operation.id);
  return entry;
}

function concatSequence(previous: DenseTensor, current: DenseTensor): DenseTensor {
  const [batch, heads, previousSequence, headDim] = previous.shape as [number, number, number, number];
  const [, , currentSequence] = current.shape as [number, number, number, number];
  const values = new Float64Array(batch * heads * (previousSequence + currentSequence) * headDim);
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) {
    const target = (b * heads + h) * (previousSequence + currentSequence) * headDim;
    values.set(previous.values.subarray((b * heads + h) * previousSequence * headDim, (b * heads + h + 1) * previousSequence * headDim), target);
    values.set(current.values.subarray((b * heads + h) * currentSequence * headDim, (b * heads + h + 1) * currentSequence * headDim), target + previousSequence * headDim);
  }
  return dense([batch, heads, previousSequence + currentSequence, headDim], values);
}

function assertCacheEntry(key: DenseTensor, valueTensor: DenseTensor, currentKey: DenseTensor, currentValue: DenseTensor, operationId: string): void {
  if (key.shape.length !== 4 || valueTensor.shape.length !== 4 || currentKey.shape.length !== 4 || currentValue.shape.length !== 4) throw new Error(`${operationId}: cache KV requer tensores BHSD.`);
  if (key.shape[0] !== currentKey.shape[0] || key.shape[1] !== currentKey.shape[1] || key.shape[3] !== currentKey.shape[3] || valueTensor.shape[0] !== currentValue.shape[0] || valueTensor.shape[1] !== currentValue.shape[1] || valueTensor.shape[3] !== currentValue.shape[3] || key.shape[0] !== valueTensor.shape[0] || key.shape[1] !== valueTensor.shape[1] || key.shape[2] !== valueTensor.shape[2] || key.shape[3] !== valueTensor.shape[3]) {
    throw new Error(`${operationId}: shape do cache KV é incompatível com as projeções atuais.`);
  }
}

function rotary(input: DenseTensor, positions: number[][], operation: Extract<Operation, { op: "rotary_embedding" }>): DenseTensor {
  if (operation.ropeType !== "default" || operation.layout !== "rotate_half" || operation.scaling) throw new Error(`${operation.id}: variante RoPE não suportada pelo executor F64.`);
  if (input.shape.length !== 4 || operation.rotaryDim <= 0 || operation.rotaryDim % 2 !== 0 || operation.rotaryDim > input.shape[3]!) throw new Error(`${operation.id}: dimensão RoPE inválida.`);
  const [batch, heads, sequence, headDim] = input.shape as [number, number, number, number];
  if (positions.length !== batch || positions.some((row) => row.length !== sequence)) throw new Error("Posições incompatíveis com RoPE.");
  const result = Float64Array.from(input.values);
  const half = operation.rotaryDim / 2;
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let s = 0; s < sequence; s += 1) for (let pair = 0; pair < half; pair += 1) {
    const angle = positions[b]![s]! / operation.theta ** ((2 * pair) / operation.rotaryDim);
    const base = ((b * heads + h) * sequence + s) * headDim;
    const first = input.values[base + pair]!;
    const second = input.values[base + pair + half]!;
    result[base + pair] = first * Math.cos(angle) - second * Math.sin(angle);
    result[base + pair + half] = second * Math.cos(angle) + first * Math.sin(angle);
  }
  return dense([...input.shape], result);
}

function attention(query: DenseTensor, key: DenseTensor, valueTensor: DenseTensor, operation: Extract<Operation, { op: "scaled_dot_product_attention" }>, attentionMask?: DenseTensor, pastLength = 0): DenseTensor {
  if (query.shape.length !== 4 || key.shape.length !== 4 || valueTensor.shape.length !== 4) throw new Error(`${operation.id}: attention requer tensores BHSD.`);
  const [batch, queryHeads, querySequence, headDim] = query.shape as [number, number, number, number];
  const [keyBatch, keyHeads, keySequence, keyDim] = key.shape as [number, number, number, number];
  assertShape(valueTensor, [keyBatch, keyHeads, keySequence, keyDim], `${operation.id}: value`);
  if (batch !== keyBatch || queryHeads !== operation.numAttentionHeads || keyHeads !== operation.numKeyValueHeads || headDim !== operation.headDim || keyDim !== headDim) throw new Error(`${operation.id}: topologia attention incompatível.`);
  const result = new Float64Array(batch * querySequence * queryHeads * headDim);
  const group = queryHeads / keyHeads;
  if (!Number.isInteger(group)) throw new Error(`${operation.id}: GQA inválida.`);
  const mask = validateAttentionMask(attentionMask, batch, queryHeads, querySequence, keySequence, operation.id);
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < queryHeads; h += 1) for (let q = 0; q < querySequence; q += 1) {
    const kvHead = Math.floor(h / group);
    const absoluteQuery = pastLength + q;
    const firstKey = operation.slidingWindow === undefined ? 0 : Math.max(0, absoluteQuery - operation.slidingWindow + 1);
    const lastKey = operation.causal ? Math.min(absoluteQuery, keySequence - 1) : keySequence - 1;
    const scores = new Float64Array(keySequence);
    let max = -Infinity;
    for (let k = firstKey; k <= lastKey; k += 1) {
      let dot = 0;
      for (let d = 0; d < headDim; d += 1) dot += query.values[((b * queryHeads + h) * querySequence + q) * headDim + d]! * key.values[((b * keyHeads + kvHead) * keySequence + k) * headDim + d]!;
      const unmasked = operation.scoreSoftcap === undefined ? dot * operation.scale : operation.scoreSoftcap * Math.tanh((dot * operation.scale) / operation.scoreSoftcap);
      const score = unmasked + (mask ? mask.values[maskOffset(mask, b, h, q, k)]! : 0);
      scores[k] = score;
      max = Math.max(max, score);
    }
    if (max === -Infinity) throw new Error(`${operation.id}: attentionMask excluiu todas as chaves da consulta ${q}.`);
    let total = 0;
    for (let k = firstKey; k <= lastKey; k += 1) { scores[k] = Math.exp(scores[k]! - max); total += scores[k]!; }
    for (let d = 0; d < headDim; d += 1) {
      let output = 0;
      for (let k = firstKey; k <= lastKey; k += 1) output += (scores[k]! / total) * valueTensor.values[((b * keyHeads + kvHead) * keySequence + k) * headDim + d]!;
      result[((b * querySequence + q) * queryHeads + h) * headDim + d] = output;
    }
  }
  return dense([batch, querySequence, queryHeads * headDim], result);
}

function activation(input: DenseTensor, functionName: string, approximation?: string): DenseTensor {
  if (functionName === "silu") return dense([...input.shape], input.values.map((x) => x / (1 + Math.exp(-x))));
  if (functionName === "gelu" && approximation === "tanh") return dense([...input.shape], input.values.map((x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)))));
  if (functionName === "gelu" && approximation === "erf") return dense([...input.shape], input.values.map((x) => 0.5 * x * (1 + erf(x / Math.sqrt(2)))));
  throw new Error(`Ativação não suportada pelo executor F64: ${functionName}/${approximation ?? "none"}.`);
}

function elementwise(inputs: DenseTensor[], kind: Extract<Operation, { op: "elementwise" }> ["kind"], scalar?: number): DenseTensor {
  if (inputs.length === 0) throw new Error("Operação elementwise sem entradas.");
  const first = inputs[0]!;
  for (const input of inputs.slice(1)) assertShape(input, first.shape, "elementwise");
  if ((kind === "add" || kind === "multiply") && inputs.length !== 2) throw new Error(`${kind} requer duas entradas.`);
  if ((kind === "scale" || kind === "tanh_softcap") && (inputs.length !== 1 || scalar === undefined)) throw new Error(`${kind} requer escalar.`);
  const result = new Float64Array(first.values.length);
  for (let index = 0; index < result.length; index += 1) {
    if (kind === "add") result[index] = first.values[index]! + inputs[1]!.values[index]!;
    else if (kind === "multiply") result[index] = first.values[index]! * inputs[1]!.values[index]!;
    else if (kind === "scale") result[index] = first.values[index]! * scalar!;
    else result[index] = scalar! * Math.tanh(first.values[index]! / scalar!);
  }
  return dense([...first.shape], result);
}

const f32 = Math.fround;

function embeddingF32(inputIds: number[][], weight: DenseF32Tensor, scale?: number): DenseF32Tensor {
  if (weight.shape.length !== 2) throw new Error("Embedding F32 requer weight 2D.");
  const [vocab, hidden] = weight.shape as [number, number];
  const result = new Float32Array(inputIds.length * inputIds[0]!.length * hidden);
  const f32Scale = f32(scale ?? 1);
  for (let batch = 0; batch < inputIds.length; batch += 1) for (let sequence = 0; sequence < inputIds[batch]!.length; sequence += 1) {
    const token = inputIds[batch]![sequence]!;
    if (!Number.isInteger(token) || token < 0 || token >= vocab) throw new Error(`Token fora do vocabulário: ${token}.`);
    const offset = (batch * inputIds[batch]!.length + sequence) * hidden;
    for (let index = 0; index < hidden; index += 1) result[offset + index] = f32(f32(weight.values[token * hidden + index]!) * f32Scale);
  }
  return denseF32([inputIds.length, inputIds[0]!.length, hidden], result);
}

function rmsNormF32(input: DenseF32Tensor, weight: DenseF32Tensor, operation: Extract<Operation, { op: "rms_norm" }>): DenseF32Tensor {
  const width = input.shape.at(-1);
  if (width === undefined || weight.shape.length !== 1 || weight.shape[0] !== width) throw new Error(`${operation.id}: RMSNorm incompatível.`);
  const result = new Float32Array(input.values.length);
  const epsilon = f32(operation.epsilon);
  for (let offset = 0; offset < input.values.length; offset += width) {
    let sum = f32(0);
    for (let index = 0; index < width; index += 1) sum = f32(sum + f32(input.values[offset + index]! * input.values[offset + index]!));
    const mean = f32(sum / f32(width));
    const scale = f32(1 / f32(Math.sqrt(f32(mean + epsilon))));
    for (let index = 0; index < width; index += 1) {
      const multiplier = operation.weightTransform === "one_plus_weight" ? f32(1 + weight.values[index]!) : weight.values[index]!;
      result[offset + index] = f32(f32(input.values[offset + index]! * scale) * multiplier);
    }
  }
  return denseF32([...input.shape], result);
}

function linearF32(input: DenseF32Tensor, weight: DenseF32Tensor, bias?: DenseF32Tensor): DenseF32Tensor {
  if (input.shape.length < 1 || weight.shape.length !== 2) throw new Error("Linear F32 requer entrada e weight válidos.");
  const features = input.shape.at(-1)!;
  const [outFeatures, inFeatures] = weight.shape as [number, number];
  if (features !== inFeatures) throw new Error(`Linear: entrada ${features} incompatível com weight ${outFeatures}x${inFeatures}.`);
  if (bias && (bias.shape.length !== 1 || bias.shape[0] !== outFeatures)) throw new Error("Linear: bias incompatível.");
  const rows = input.values.length / features;
  const result = new Float32Array(rows * outFeatures);
  for (let row = 0; row < rows; row += 1) for (let output = 0; output < outFeatures; output += 1) {
    let sum = f32(bias?.values[output] ?? 0);
    for (let column = 0; column < inFeatures; column += 1) sum = f32(sum + f32(input.values[row * inFeatures + column]! * weight.values[output * inFeatures + column]!));
    result[row * outFeatures + output] = sum;
  }
  return denseF32([...input.shape.slice(0, -1), outFeatures], result);
}

function reshapeHeadsF32(input: DenseF32Tensor, heads: number, headDim: number): DenseF32Tensor {
  if (input.shape.length !== 3 || input.shape[2] !== heads * headDim) throw new Error("reshape_heads requer [B,S,H*D].");
  const [batch, sequence] = input.shape as [number, number, number];
  const result = new Float32Array(input.values.length);
  for (let b = 0; b < batch; b += 1) for (let s = 0; s < sequence; s += 1) for (let h = 0; h < heads; h += 1) for (let d = 0; d < headDim; d += 1) {
    result[((b * heads + h) * sequence + s) * headDim + d] = input.values[(b * sequence + s) * heads * headDim + h * headDim + d]!;
  }
  return denseF32([batch, heads, sequence, headDim], result);
}

function cacheForLayerF32(
  cache: ReadonlyMap<number, ReferenceF32KeyValueCache> | undefined,
  layer: number,
  operation: Extract<Operation, { op: "scaled_dot_product_attention" }>,
  currentKey: DenseF32Tensor,
  currentValue: DenseF32Tensor,
): ReferenceF32KeyValueCache | undefined {
  if (!cache) return undefined;
  const entry = cache.get(layer);
  if (!entry) throw new Error(`${operation.id}: pastKeyValues não contém a camada ${layer}.`);
  assertCacheEntryF32(entry.key, entry.value, currentKey, currentValue, operation.id);
  return entry;
}

function concatSequenceF32(previous: DenseF32Tensor, current: DenseF32Tensor): DenseF32Tensor {
  const [batch, heads, previousSequence, headDim] = previous.shape as [number, number, number, number];
  const [, , currentSequence] = current.shape as [number, number, number, number];
  const values = new Float32Array(batch * heads * (previousSequence + currentSequence) * headDim);
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) {
    const target = (b * heads + h) * (previousSequence + currentSequence) * headDim;
    values.set(previous.values.subarray((b * heads + h) * previousSequence * headDim, (b * heads + h + 1) * previousSequence * headDim), target);
    values.set(current.values.subarray((b * heads + h) * currentSequence * headDim, (b * heads + h + 1) * currentSequence * headDim), target + previousSequence * headDim);
  }
  return denseF32([batch, heads, previousSequence + currentSequence, headDim], values);
}

function assertCacheEntryF32(key: DenseF32Tensor, valueTensor: DenseF32Tensor, currentKey: DenseF32Tensor, currentValue: DenseF32Tensor, operationId: string): void {
  if (key.shape.length !== 4 || valueTensor.shape.length !== 4 || currentKey.shape.length !== 4 || currentValue.shape.length !== 4) throw new Error(`${operationId}: cache KV requer tensores BHSD.`);
  if (key.shape[0] !== currentKey.shape[0] || key.shape[1] !== currentKey.shape[1] || key.shape[3] !== currentKey.shape[3] || valueTensor.shape[0] !== currentValue.shape[0] || valueTensor.shape[1] !== currentValue.shape[1] || valueTensor.shape[3] !== currentValue.shape[3] || key.shape[0] !== valueTensor.shape[0] || key.shape[1] !== valueTensor.shape[1] || key.shape[2] !== valueTensor.shape[2] || key.shape[3] !== valueTensor.shape[3]) {
    throw new Error(`${operationId}: shape do cache KV é incompatível com as projeções atuais.`);
  }
}

function rotaryF32(input: DenseF32Tensor, positions: number[][], operation: Extract<Operation, { op: "rotary_embedding" }>): DenseF32Tensor {
  if (operation.ropeType !== "default" || operation.layout !== "rotate_half" || operation.scaling) throw new Error(`${operation.id}: variante RoPE não suportada pelo executor F32.`);
  if (input.shape.length !== 4 || operation.rotaryDim <= 0 || operation.rotaryDim % 2 !== 0 || operation.rotaryDim > input.shape[3]!) throw new Error(`${operation.id}: dimensão RoPE inválida.`);
  const [batch, heads, sequence, headDim] = input.shape as [number, number, number, number];
  if (positions.length !== batch || positions.some((row) => row.length !== sequence)) throw new Error("Posições incompatíveis com RoPE.");
  const result = Float32Array.from(input.values);
  const half = operation.rotaryDim / 2;
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < heads; h += 1) for (let s = 0; s < sequence; s += 1) for (let pair = 0; pair < half; pair += 1) {
    const angle = f32(positions[b]![s]! / f32(operation.theta ** ((2 * pair) / operation.rotaryDim)));
    const cosine = f32(Math.cos(angle));
    const sine = f32(Math.sin(angle));
    const base = ((b * heads + h) * sequence + s) * headDim;
    const first = input.values[base + pair]!;
    const second = input.values[base + pair + half]!;
    result[base + pair] = f32(f32(first * cosine) - f32(second * sine));
    result[base + pair + half] = f32(f32(second * cosine) + f32(first * sine));
  }
  return denseF32([...input.shape], result);
}

function attentionF32(query: DenseF32Tensor, key: DenseF32Tensor, valueTensor: DenseF32Tensor, operation: Extract<Operation, { op: "scaled_dot_product_attention" }>, attentionMask?: DenseF32Tensor, pastLength = 0): DenseF32Tensor {
  if (query.shape.length !== 4 || key.shape.length !== 4 || valueTensor.shape.length !== 4) throw new Error(`${operation.id}: attention requer tensores BHSD.`);
  const [batch, queryHeads, querySequence, headDim] = query.shape as [number, number, number, number];
  const [keyBatch, keyHeads, keySequence, keyDim] = key.shape as [number, number, number, number];
  assertShapeF32(valueTensor, [keyBatch, keyHeads, keySequence, keyDim], `${operation.id}: value`);
  if (batch !== keyBatch || queryHeads !== operation.numAttentionHeads || keyHeads !== operation.numKeyValueHeads || headDim !== operation.headDim || keyDim !== headDim) throw new Error(`${operation.id}: topologia attention incompatível.`);
  const result = new Float32Array(batch * querySequence * queryHeads * headDim);
  const group = queryHeads / keyHeads;
  if (!Number.isInteger(group)) throw new Error(`${operation.id}: GQA inválida.`);
  const mask = validateAttentionMask(attentionMask, batch, queryHeads, querySequence, keySequence, operation.id);
  const scale = f32(operation.scale);
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < queryHeads; h += 1) for (let q = 0; q < querySequence; q += 1) {
    const kvHead = Math.floor(h / group);
    const absoluteQuery = pastLength + q;
    const firstKey = operation.slidingWindow === undefined ? 0 : Math.max(0, absoluteQuery - operation.slidingWindow + 1);
    const lastKey = operation.causal ? Math.min(absoluteQuery, keySequence - 1) : keySequence - 1;
    const scores = new Float32Array(keySequence);
    let max = -Infinity;
    for (let k = firstKey; k <= lastKey; k += 1) {
      let dot = f32(0);
      for (let d = 0; d < headDim; d += 1) dot = f32(dot + f32(query.values[((b * queryHeads + h) * querySequence + q) * headDim + d]! * key.values[((b * keyHeads + kvHead) * keySequence + k) * headDim + d]!));
      const scaled = f32(dot * scale);
      const unmasked = operation.scoreSoftcap === undefined ? scaled : f32(f32(operation.scoreSoftcap) * f32(Math.tanh(f32(scaled / f32(operation.scoreSoftcap)))));
      const score = f32(unmasked + (mask ? mask.values[maskOffset(mask, b, h, q, k)]! : 0));
      scores[k] = score;
      max = Math.max(max, score);
    }
    if (max === -Infinity) throw new Error(`${operation.id}: attentionMask excluiu todas as chaves da consulta ${q}.`);
    let total = f32(0);
    for (let k = firstKey; k <= lastKey; k += 1) { scores[k] = f32(Math.exp(f32(scores[k]! - f32(max)))); total = f32(total + scores[k]!); }
    for (let d = 0; d < headDim; d += 1) {
      let output = f32(0);
      for (let k = firstKey; k <= lastKey; k += 1) output = f32(output + f32(f32(scores[k]! / total) * valueTensor.values[((b * keyHeads + kvHead) * keySequence + k) * headDim + d]!));
      result[((b * querySequence + q) * queryHeads + h) * headDim + d] = output;
    }
  }
  return denseF32([batch, querySequence, queryHeads * headDim], result);
}

function activationF32(input: DenseF32Tensor, functionName: string, approximation?: string): DenseF32Tensor {
  const values = new Float32Array(input.values.length);
  for (let index = 0; index < values.length; index += 1) {
    const x = input.values[index]!;
    if (functionName === "silu") values[index] = f32(x / f32(1 + f32(Math.exp(-x))));
    else if (functionName === "gelu" && approximation === "tanh") values[index] = f32(f32(0.5 * x) * f32(1 + f32(Math.tanh(f32(Math.sqrt(2 / Math.PI) * f32(x + f32(0.044715 * f32(x * f32(x * x)))))))));
    else if (functionName === "gelu" && approximation === "erf") values[index] = f32(f32(0.5 * x) * f32(1 + f32(erf(f32(x / f32(Math.sqrt(2)))))));
    else throw new Error(`Ativação não suportada pelo executor F32: ${functionName}/${approximation ?? "none"}.`);
  }
  return denseF32([...input.shape], values);
}

function elementwiseF32(inputs: DenseF32Tensor[], kind: Extract<Operation, { op: "elementwise" }> ["kind"], scalar?: number): DenseF32Tensor {
  if (inputs.length === 0) throw new Error("Operação elementwise sem entradas.");
  const first = inputs[0]!;
  for (const input of inputs.slice(1)) assertShapeF32(input, first.shape, "elementwise");
  if ((kind === "add" || kind === "multiply") && inputs.length !== 2) throw new Error(`${kind} requer duas entradas.`);
  if ((kind === "scale" || kind === "tanh_softcap") && (inputs.length !== 1 || scalar === undefined)) throw new Error(`${kind} requer escalar.`);
  const result = new Float32Array(first.values.length);
  const f32Scalar = scalar === undefined ? undefined : f32(scalar);
  for (let index = 0; index < result.length; index += 1) {
    if (kind === "add") result[index] = f32(first.values[index]! + inputs[1]!.values[index]!);
    else if (kind === "multiply") result[index] = f32(first.values[index]! * inputs[1]!.values[index]!);
    else if (kind === "scale") result[index] = f32(first.values[index]! * f32Scalar!);
    else result[index] = f32(f32Scalar! * f32(Math.tanh(f32(first.values[index]! / f32Scalar!))));
  }
  return denseF32([...first.shape], result);
}

function erf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  return sign * (1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-x * x));
}
