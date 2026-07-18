import type { ReductionSchedule } from "./types.js";

const f32 = Math.fround;

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
