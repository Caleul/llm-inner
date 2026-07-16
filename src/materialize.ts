import type { TensorBridge } from "./bridge.js";
import type { DenseF32Tensor, DenseTensor, ModelCatalog, ModelIR, Operation, QuantizationSpec, TensorInfo, TensorRef } from "./types.js";
import { adaptGgufDecoderCatalog } from "./gguf-llama.js";

export interface DenseF32Reader {
  readDenseAsF32(tensor: TensorInfo): Promise<DenseF32Tensor>;
}

/** Deliberately separate from F32 materialization: no storage widening occurs. */
export interface DenseF64Reader {
  readDenseF64(tensor: TensorInfo): Promise<DenseTensor>;
}

interface NativeMlxAffineReader extends DenseF32Reader {
  readMlxAffineAsF32(tensor: TensorInfo, scales: TensorInfo, biases?: TensorInfo): Promise<DenseF32Tensor>;
}

/**
 * Materialize the complete set of constants referenced by an IR for the
 * scalar F32 executor. This is deliberately separate from lowering: a model
 * configured with an unknown compute policy is still not executable merely
 * because its storage can be decoded into F32.
 *
 * The catalog is rechecked against every IR TensorRef. This prevents callers
 * from accidentally combining an IR with a same-named tensor from another
 * checkpoint, shard layout, or quantization contract.
 */
export async function materializeReferenceF32Constants(
  ir: ModelIR,
  catalog: ModelCatalog,
  reader: DenseF32Reader,
  bridge?: Pick<TensorBridge, "readMlxDequantizedF32">,
): Promise<ReadonlyMap<string, DenseF32Tensor>> {
  const adaptedCatalog = adaptGgufDecoderCatalog(catalog);
  const references = referencedTensors(ir);
  const constants = new Map<string, DenseF32Tensor>();
  for (const reference of references.values()) {
    const catalogued = adaptedCatalog.tensors.get(reference.name);
    if (!catalogued) throw new Error(`IR referencia tensor ausente do catálogo: ${reference.name}.`);
    assertReferenceMatchesCatalog(reference, catalogued);

    let materialized: DenseF32Tensor;
    if (catalogued.quantization) {
      if (adaptedCatalog.format === "mlx-safetensors" && catalogued.quantization.family === "mlx") {
        const scales = catalogued.quantization.scaleTensor ? adaptedCatalog.tensors.get(catalogued.quantization.scaleTensor) : undefined;
        const biases = catalogued.quantization.biasTensor ? adaptedCatalog.tensors.get(catalogued.quantization.biasTensor) : undefined;
        if (catalogued.quantization.mode === "affine" && supportsNativeMlxAffine(reader) && scales && (!catalogued.quantization.biasTensor || biases)) {
          materialized = await reader.readMlxAffineAsF32(catalogued, scales, biases);
        } else {
          if (!bridge) throw new Error(`${reference.name}: quantização MLX ${catalogued.quantization.mode} requer TensorBridge com mlx.core.dequantize; affine nativo exige reader e tensors de parâmetros verificados.`);
          materialized = await bridge.readMlxDequantizedF32(adaptedCatalog, reference.name);
        }
      } else if (adaptedCatalog.format === "gguf" && catalogued.quantization.family === "gguf" && (catalogued.quantization.mode === "q2_k" || catalogued.quantization.mode === "q3_k" || catalogued.quantization.mode === "q4_0" || catalogued.quantization.mode === "q4_1" || catalogued.quantization.mode === "q4_k" || catalogued.quantization.mode === "q5_0" || catalogued.quantization.mode === "q5_1" || catalogued.quantization.mode === "q5_k" || catalogued.quantization.mode === "q6_k" || catalogued.quantization.mode === "q8_0" || catalogued.quantization.mode === "q8_1" || catalogued.quantization.mode === "q8_k")) {
        materialized = await reader.readDenseAsF32(catalogued);
      } else {
        throw new Error(`${reference.name}: quantização ${catalogued.quantization.family}/${catalogued.quantization.mode} não possui materializador F32 verificado.`);
      }
    } else {
      materialized = await reader.readDenseAsF32(catalogued);
    }
    assertMaterializedTensor(reference, materialized);
    constants.set(reference.name, materialized);
  }
  return constants;
}

/**
 * Materialize the exact dense F64 constants required by the F64 executor.
 * Quantized and lower-precision storage are rejected rather than being widened
 * into a different accumulation/rounding contract.
 */
export async function materializeReferenceF64Constants(
  ir: ModelIR,
  catalog: ModelCatalog,
  reader: DenseF64Reader,
): Promise<ReadonlyMap<string, DenseTensor>> {
  const adaptedCatalog = adaptGgufDecoderCatalog(catalog);
  const constants = new Map<string, DenseTensor>();
  for (const reference of referencedTensors(ir).values()) {
    const catalogued = adaptedCatalog.tensors.get(reference.name);
    if (!catalogued) throw new Error(`IR referencia tensor ausente do catálogo: ${reference.name}.`);
    assertReferenceMatchesCatalog(reference, catalogued);
    if (catalogued.storageDtype !== "F64" || catalogued.quantization) {
      throw new Error(`${reference.name}: executor F64 requer storage F64 denso não quantizado; recebeu ${catalogued.storageDtype}${catalogued.quantization ? " quantizado" : ""}.`);
    }
    const materialized = await reader.readDenseF64(catalogued);
    const elements = reference.shape.reduce((product, dimension) => product * dimension, 1);
    if (!sameShape(reference.shape, materialized.shape) || materialized.values.length !== elements) {
      throw new Error(`${reference.name}: materializador F64 retornou shape incompatível [${materialized.shape.join(", ")}].`);
    }
    constants.set(reference.name, materialized);
  }
  return constants;
}

function supportsNativeMlxAffine(reader: DenseF32Reader): reader is NativeMlxAffineReader {
  return "readMlxAffineAsF32" in reader && typeof reader.readMlxAffineAsF32 === "function";
}

function referencedTensors(ir: ModelIR): Map<string, TensorRef> {
  const references = new Map<string, TensorRef>();
  for (const operation of allOperations(ir)) {
    switch (operation.op) {
      case "embedding":
      case "per_layer_embedding":
        registerReference(references, operation.weight);
        break;
      case "rms_norm":
        if (operation.weight) registerReference(references, operation.weight);
        break;
      case "linear":
        registerReference(references, operation.weight);
        if (operation.bias) registerReference(references, operation.bias);
        break;
      case "tensor_scale":
        registerReference(references, operation.scalar);
        break;
      default:
        break;
    }
  }
  return references;
}

function allOperations(ir: ModelIR): Operation[] {
  return [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
}

function registerReference(references: Map<string, TensorRef>, reference: TensorRef): void {
  const existing = references.get(reference.name);
  if (existing && !sameReference(existing, reference)) {
    throw new Error(`IR contém referências incompatíveis para o tensor ${reference.name}.`);
  }
  references.set(reference.name, reference);
}

function assertReferenceMatchesCatalog(reference: TensorRef, tensor: TensorInfo): void {
  if (
    reference.storageDtype !== tensor.storageDtype ||
    !sameShape(reference.shape, tensor.logicalShape) ||
    !sameQuantization(reference.quantization, tensor.quantization)
  ) {
    throw new Error(`${reference.name}: referência do IR diverge do catálogo de origem; recuse a materialização cruzada.`);
  }
}

function assertMaterializedTensor(reference: TensorRef, tensor: DenseF32Tensor): void {
  const elements = reference.shape.reduce((product, dimension) => product * dimension, 1);
  if (!sameShape(reference.shape, tensor.shape) || tensor.values.length !== elements) {
    throw new Error(`${reference.name}: materializador F32 retornou shape incompatível [${tensor.shape.join(", ")}].`);
  }
  if (!sameQuantization(reference.quantization, tensor.sourceQuantization)) {
    throw new Error(`${reference.name}: proveniência de quantização do materializador diverge da referência do IR.`);
  }
}

function sameReference(left: TensorRef, right: TensorRef): boolean {
  return left.name === right.name &&
    left.storageDtype === right.storageDtype &&
    sameShape(left.shape, right.shape) &&
    sameQuantization(left.quantization, right.quantization);
}

function sameShape(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((dimension, index) => dimension === right[index]);
}

function sameQuantization(left: QuantizationSpec | undefined, right: QuantizationSpec | undefined): boolean {
  if (!left || !right) return left === right;
  return left.family === right.family &&
    left.mode === right.mode &&
    left.bits === right.bits &&
    left.groupSize === right.groupSize &&
    left.tensorType === right.tensorType &&
    left.scaleTensor === right.scaleTensor &&
    left.biasTensor === right.biasTensor &&
    left.globalScaleTensor === right.globalScaleTensor;
}
