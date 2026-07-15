import type {
  DenseTensor,
  ModelIR,
  Operation,
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
        values.set(operation.output, attention(value(values, operation.query), value(values, operation.key), value(values, operation.value), operation));
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
  return { values, logits };
}

function assertF64Policy(operation: Operation): void {
  const policy = operation.dtypePolicy;
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

function tensor(request: ReferenceExecutionRequest, reference: TensorRef): DenseTensor {
  if (reference.quantization) throw new Error(`${reference.name}: tensor quantizado não é suportado pelo executor F64.`);
  const found = request.tensors.get(reference.name);
  if (!found) throw new Error(`Tensor F64 ausente: ${reference.name}`);
  assertShape(found, reference.shape, reference.name);
  return found;
}

function value(values: Map<string, DenseTensor>, name: string): DenseTensor {
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

function attention(query: DenseTensor, key: DenseTensor, valueTensor: DenseTensor, operation: Extract<Operation, { op: "scaled_dot_product_attention" }>): DenseTensor {
  if (query.shape.length !== 4 || key.shape.length !== 4 || valueTensor.shape.length !== 4) throw new Error(`${operation.id}: attention requer tensores BHSD.`);
  const [batch, queryHeads, querySequence, headDim] = query.shape as [number, number, number, number];
  const [keyBatch, keyHeads, keySequence, keyDim] = key.shape as [number, number, number, number];
  assertShape(valueTensor, [keyBatch, keyHeads, keySequence, keyDim], `${operation.id}: value`);
  if (batch !== keyBatch || queryHeads !== operation.numAttentionHeads || keyHeads !== operation.numKeyValueHeads || headDim !== operation.headDim || keyDim !== headDim) throw new Error(`${operation.id}: topologia attention incompatível.`);
  const result = new Float64Array(batch * querySequence * queryHeads * headDim);
  const group = queryHeads / keyHeads;
  if (!Number.isInteger(group)) throw new Error(`${operation.id}: GQA inválida.`);
  for (let b = 0; b < batch; b += 1) for (let h = 0; h < queryHeads; h += 1) for (let q = 0; q < querySequence; q += 1) {
    const kvHead = Math.floor(h / group);
    const firstKey = operation.slidingWindow === undefined ? 0 : Math.max(0, q - operation.slidingWindow + 1);
    const lastKey = operation.causal ? Math.min(q, keySequence - 1) : keySequence - 1;
    const scores = new Float64Array(keySequence);
    let max = -Infinity;
    for (let k = firstKey; k <= lastKey; k += 1) {
      let dot = 0;
      for (let d = 0; d < headDim; d += 1) dot += query.values[((b * queryHeads + h) * querySequence + q) * headDim + d]! * key.values[((b * keyHeads + kvHead) * keySequence + k) * headDim + d]!;
      const score = operation.scoreSoftcap === undefined ? dot * operation.scale : operation.scoreSoftcap * Math.tanh((dot * operation.scale) / operation.scoreSoftcap);
      scores[k] = score;
      max = Math.max(max, score);
    }
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

function erf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value);
  const t = 1 / (1 + 0.3275911 * x);
  return sign * (1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-x * x));
}
