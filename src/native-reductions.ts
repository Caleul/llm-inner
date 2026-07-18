import type { ReductionSchedule } from "./types.js";

const f32 = Math.fround;

/** PyTorch generic reduced-precision GEMM `sum()` with four ILP accumulators. */
export function pytorchCpuBf16GemmIlp4F32(
  length: number,
  left: (index: number) => number,
  right: (index: number) => number,
): number {
  if (!Number.isSafeInteger(length) || length < 0) throw new Error(`PyTorch BF16 GEMM requer comprimento inteiro não negativo; recebeu ${length}.`);
  const partial = new Float32Array(4);
  let index = 0;
  for (; index + 4 <= length; index += 4) for (let lane = 0; lane < 4; lane += 1) {
    partial[lane] = f32(partial[lane]! + f32(left(index + lane) * right(index + lane)));
  }
  for (; index < length; index += 1) partial[0] = f32(partial[0]! + f32(left(index) * right(index)));
  for (let lane = 1; lane < 4; lane += 1) partial[0] = f32(partial[0]! + partial[lane]!);
  return partial[0]!;
}

export interface PytorchCpuBf16WelfordMoments {
  mean: number;
  variance: number;
}

/** Scalar transcript of PyTorch's BF16 `RowwiseMoments` ADVSIMD path. */
export function pytorchCpuBf16WelfordMomentsF32(
  length: number,
  value: (index: number) => number,
  reduction: Extract<ReductionSchedule, { kind: "pytorch-cpu-bf16-welford" }>,
): PytorchCpuBf16WelfordMoments {
  assertPytorchCpuBf16WelfordReduction(reduction);
  if (!Number.isSafeInteger(length) || length <= 0 || length % reduction.inputVectorLanes !== 0 ||
    length / reduction.inputVectorLanes > reduction.chunkVectors) {
    throw new Error(`PyTorch BF16 Welford requer 1..${reduction.chunkVectors} vetores completos de ${reduction.inputVectorLanes} lanes; recebeu ${length}.`);
  }
  const vectorCount = length / reduction.inputVectorLanes;
  const lowMean = new Float32Array(4), highMean = new Float32Array(4);
  const lowM2 = new Float32Array(4), highM2 = new Float32Array(4);
  for (let vector = 0; vector < vectorCount; vector += 1) {
    const reciprocal = f32(1 / f32(vector + 1));
    for (let lane = 0; lane < 4; lane += 1) {
      welfordLaneUpdate(lowMean, lowM2, lane, value(vector * 8 + lane), reciprocal);
      welfordLaneUpdate(highMean, highM2, lane, value(vector * 8 + 4 + lane), reciprocal);
    }
  }
  const mergedMean = new Float32Array(4), mergedM2 = new Float32Array(4);
  for (let lane = 0; lane < 4; lane += 1) {
    const delta = f32(highMean[lane]! - lowMean[lane]!);
    const ratioDelta = f32(f32(0.5) * delta);
    mergedMean[lane] = f32(lowMean[lane]! + ratioDelta);
    mergedM2[lane] = f32(f32(delta * f32(vectorCount)) * ratioDelta + f32(lowM2[lane]! + highM2[lane]!));
  }
  let count = 0, mean = f32(0), m2 = f32(0);
  ({ count, mean, m2 } = foldMomentLanes(vectorCount * 2, mergedMean, mergedM2, count, mean, m2));
  return { mean, variance: f32(m2 / f32(length)) };
}

export function assertPytorchCpuBf16WelfordReduction(
  reduction: Extract<ReductionSchedule, { kind: "pytorch-cpu-bf16-welford" }>,
): void {
  if (reduction.inputVectorLanes !== 8 || reduction.accumulatorVectorLanes !== 4 || reduction.chunkVectors !== 16 ||
    reduction.vectorMergeOrder !== "low-then-high" || reduction.laneFold !== "ascending" ||
    reduction.secondPass !== "x-times-scale-plus-bias-times-gamma") {
    throw new Error("Agenda PyTorch CPU BF16 Welford inválida.");
  }
}

function welfordLaneUpdate(mean: Float32Array, m2: Float32Array, lane: number, sample: number, reciprocal: number): void {
  const delta = f32(sample - mean[lane]!);
  mean[lane] = f32(mean[lane]! + delta * reciprocal);
  const remainder = f32(sample - mean[lane]!);
  m2[lane] = f32(m2[lane]! + delta * remainder);
}

function foldMomentLanes(
  countToAdd: number,
  meansToAdd: Float32Array,
  m2ToAdd: Float32Array,
  count: number,
  mean: number,
  m2: number,
): { count: number; mean: number; m2: number } {
  for (let lane = 0; lane < 4; lane += 1) {
    const total = count + countToAdd;
    const ratio = f32(countToAdd / total);
    const delta = f32(meansToAdd[lane]! - mean);
    const priorDelta = f32(delta * f32(count));
    const scaledDelta = f32(ratio * delta);
    mean = f32(mean + scaledDelta);
    m2 = f32(priorDelta * scaledDelta + f32(m2 + m2ToAdd[lane]!));
    count = total;
  }
  return { count, mean, m2 };
}

/**
 * Scalar replay of PyTorch's ARM `bf16_dot_with_fp32_arith` non-BFDOT path.
 *
 * Complete 32-term iterations accumulate into eight four-lane F32 registers.
 * The source reduces that register tree, then adds an independent eight-term
 * vector tail and finally the ascending scalar tail. Keeping those three
 * regions separate matters for attention, whose dynamic key length is not
 * generally divisible by the fully-unrolled width used by dense linears.
 */
export function armNeonBf16DotF32(
  length: number,
  left: (index: number) => number,
  right: (index: number) => number,
  reduction: Extract<ReductionSchedule, { kind: "arm-neon-bf16-dot-fma" }>,
): number {
  assertArmNeonBf16DotReduction(reduction);
  if (!Number.isSafeInteger(length) || length < 0) throw new Error(`ARM NEON BF16 dot requer comprimento inteiro não negativo; recebeu ${length}.`);

  const registers = new Float32Array(reduction.laneCount);
  const complete = length - (length % reduction.laneCount);
  for (let index = 0; index < complete; index += 1) {
    const lane = index % reduction.laneCount;
    registers[lane] = f32(registers[lane]! + left(index) * right(index));
  }
  let result = reduceRegisterTree(registers, reduction.lanesPerRegister, reduction.horizontalFold);

  const vectorTailWidth = reduction.lanesPerRegister * 2;
  const vectorTailEnd = length - (length % vectorTailWidth);
  const tail = new Float32Array(reduction.lanesPerRegister);
  for (let first = complete; first < vectorTailEnd; first += vectorTailWidth) {
    for (let lane = 0; lane < reduction.lanesPerRegister; lane += 1) {
      tail[lane] = f32(tail[lane]! + left(first + lane) * right(first + lane));
      tail[lane] = f32(tail[lane]! + left(first + reduction.lanesPerRegister + lane) * right(first + reduction.lanesPerRegister + lane));
    }
  }
  result = f32(result + foldF32(tail, reduction.horizontalFold));

  for (let index = vectorTailEnd; index < length; index += 1) {
    result = f32(result + left(index) * right(index));
  }
  return result;
}

export function assertArmNeonBf16DotReduction(
  reduction: Extract<ReductionSchedule, { kind: "arm-neon-bf16-dot-fma" }>,
): void {
  if ((reduction.laneCount !== 32 && reduction.laneCount !== 64) || reduction.registerCount !== 8 ||
    (reduction.lanesPerRegister !== 4 && reduction.lanesPerRegister !== 8) ||
    reduction.laneCount !== reduction.registerCount * reduction.lanesPerRegister ||
    reduction.inputLane !== "index-modulo-vector-lane-count" ||
    (reduction.horizontalFold !== "ascending" && reduction.horizontalFold !== "pairwise")) {
    throw new Error("Agenda ARM NEON BF16 dot inválida.");
  }
}

function reduceRegisterTree(registers: Float32Array, width: number, order: "ascending" | "pairwise"): number {
  const reduced = Float32Array.from(registers);
  for (let register = 0; register < 4; register += 1) for (let lane = 0; lane < width; lane += 1) {
    const index = register * width + lane;
    reduced[index] = f32(reduced[index]! + reduced[(register + 4) * width + lane]!);
  }
  for (let register = 0; register < 2; register += 1) for (let lane = 0; lane < width; lane += 1) {
    const index = register * width + lane;
    reduced[index] = f32(reduced[index]! + reduced[(register + 2) * width + lane]!);
  }
  for (let lane = 0; lane < width; lane += 1) reduced[lane] = f32(reduced[lane]! + reduced[width + lane]!);
  return foldF32(reduced.subarray(0, width), order);
}

function foldF32(values: Float32Array, order: "ascending" | "pairwise"): number {
  if (order === "ascending") {
    let result = f32(0);
    for (const value of values) result = f32(result + value);
    return result;
  }
  let current = Float32Array.from(values);
  while (current.length > 1) {
    const next = new Float32Array(Math.ceil(current.length / 2));
    for (let index = 0; index < current.length; index += 2) {
      next[index / 2] = index + 1 < current.length ? f32(current[index]! + current[index + 1]!) : current[index]!;
    }
    current = next;
  }
  return current[0] ?? f32(0);
}
