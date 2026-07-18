import type { ReductionSchedule } from "./types.js";

type LinearReductionSchedule = Exclude<
  ReductionSchedule,
  { kind: "pytorch-cpu-f32-cascade-sum" | "pytorch-cpu-bf16-welford" }
>;

/**
 * Returns true only when the declared reduction consumes the mathematical
 * product directly at an F32 accumulation boundary. A scalar audit must not
 * pre-round these products merely because the learned operand is stored as an
 * F32 value after BF16/F16 decoding.
 */
export function gemma4LiteralReductionUsesExactProducts(schedule: ReductionSchedule): boolean {
  return schedule.kind === "ordered-fma" || schedule.kind === "interleaved-fma-lanes" ||
    schedule.kind === "tiled-fma-lanes" || schedule.kind === "arm-neon-bf16-dot-fma" ||
    schedule.kind === "arm-neon-bf16-bfdot-fma" ||
    (schedule.kind === "blocked-f32-terms" || schedule.kind === "blocked-tiled-f32-lanes") &&
      schedule.productBoundary === "fused-fma";
}

export function gemma4LiteralScalarProductFormula(
  schedule: ReductionSchedule,
  inputIndex: number,
  input: string,
  learnedLiteral: string,
): string {
  const product = gemma4LiteralReductionUsesExactProducts(schedule) ? "exact_product" : "F32";
  return `product[${inputIndex}] = ${product}(${input} * ${learnedLiteral})`;
}

/**
 * Serializes each supported linear reduction as indexed scalar assignments.
 * Every learned term is referenced only through product[i], whose definition
 * is emitted with the concrete decoded numeric literal by the caller. This is
 * intentionally shared by text, vision and audio linear audit views so a
 * native schedule cannot drift between modalities.
 */
export function buildGemma4LiteralLinearReductionAssignments(
  schedule: LinearReductionSchedule,
  width: number,
  accumulationDtype: string | undefined,
): string[] {
  if (!Number.isSafeInteger(width) || width <= 0) throw new Error(`Redução linear literal requer width positivo; recebeu ${width}.`);
  switch (schedule.kind) {
    case "ordered-scalar": {
      const accumulator = accumulationDtype === "F64" ? "F64" : accumulationDtype === "F32" ? "F32" : undefined;
      if (!accumulator) throw new Error(`ordered-scalar requer accumulationDtype F32 ou F64; recebeu ${accumulationDtype ?? "missing"}.`);
      return [
        `acc[0] = ${accumulator}(0)`,
        `acc[i+1] = ${accumulator}(acc[i] + product[i]), i=0..${width - 1} ascending`,
        `reduced = acc[${width}]`,
      ];
    }
    case "ordered-fma": return [
      "acc[0] = F32(0)",
      `acc[i+1] = F32(acc[i] + product[i]), i=0..${width - 1} ascending; product[i] is exact until this F32 boundary`,
      `reduced = acc[${width}]`,
    ];
    case "arm-neon-bf16-dot-fma": return armNeonFmaAssignments(schedule, width);
    case "arm-neon-bf16-bfdot-fma": return armNeonBfdotAssignments(schedule, width);
    case "blocked-f32-terms": {
      const blocks = Math.ceil(width / schedule.termsPerBlock);
      const index = schedule.termOrder === "ascending"
        ? "i=b*termsPerBlock+j"
        : "i=min((b+1)*termsPerBlock,width)-1-j";
      return [
        `block[b,0] = F32(0), b=0..${blocks - 1}`,
        `block[b,j+1] = F32(block[b,j] + product[i]), ${index}, j=0..min(${schedule.termsPerBlock},${width}-b*${schedule.termsPerBlock})-1 ascending traversal in ${schedule.termOrder} input order`,
        "acc[0] = F32(0)",
        `acc[b+1] = F32(acc[b] + block[b,min(${schedule.termsPerBlock},${width}-b*${schedule.termsPerBlock})]), b=0..${blocks - 1} ascending`,
        `reduced = acc[${blocks}]`,
      ];
    }
    case "interleaved-f32-lanes": case "interleaved-fma-lanes": return [
      `lane[0,l] = F32(0), l=0..${schedule.laneCount - 1}`,
      `lane[i+1,l] = l==(i mod ${schedule.laneCount}) ? F32(lane[i,l] + product[i]) : lane[i,l], i=0..${width - 1} ascending, l=0..${schedule.laneCount - 1}`,
      ...foldAssignments(`lane[${width},{lane}]`, schedule.laneCount, schedule.laneReductionOrder, "lane_fold", "reduced"),
    ];
    case "tiled-f32-lanes": case "tiled-fma-lanes": {
      const tileWidth = schedule.laneCount * schedule.termsPerLane;
      return [
        `lane[0,l] = F32(0), l=0..${schedule.laneCount - 1}`,
        `mapped_lane(i) = floor((i mod ${tileWidth})/${schedule.termsPerLane})`,
        `lane[i+1,l] = l==mapped_lane(i) ? F32(lane[i,l] + product[i]) : lane[i,l], i=0..${width - 1} ascending, l=0..${schedule.laneCount - 1}`,
        ...foldAssignments(`lane[${width},{lane}]`, schedule.laneCount, schedule.laneReductionOrder, "lane_fold", "reduced"),
      ];
    }
    case "blocked-tiled-f32-lanes": {
      const tileWidth = schedule.laneCount * schedule.termsPerLane;
      const blocks = Math.ceil(width / tileWidth);
      return [
        `block_lane[b,0,l] = F32(0), b=0..${blocks - 1}, l=0..${schedule.laneCount - 1}`,
        `block_lane[b,j+1,l] = l==floor(j/${schedule.termsPerLane}) ? F32(block_lane[b,j,l] + product[b*${tileWidth}+j]) : block_lane[b,j,l], j=0..min(${tileWidth},${width}-b*${tileWidth})-1 ascending`,
        `block_result[b] = fold_${schedule.laneReductionOrder}_F32(block_lane[b,min(${tileWidth},${width}-b*${tileWidth}),0..${schedule.laneCount - 1}])`,
        "acc[0] = F32(0)",
        `acc[b+1] = F32(acc[b] + block_result[b]), b=0..${blocks - 1} ascending`,
        `reduced = acc[${blocks}]`,
      ];
    }
  }
}

function armNeonFmaAssignments(
  schedule: Extract<ReductionSchedule, { kind: "arm-neon-bf16-dot-fma" }>,
  width: number,
): string[] {
  const completeEnd = width - width % schedule.laneCount;
  const vectorTailWidth = schedule.lanesPerRegister * 2;
  const vectorTailEnd = width - width % vectorTailWidth;
  const vectorTailBlocks = (vectorTailEnd - completeEnd) / vectorTailWidth;
  const scalarTail = width - vectorTailEnd;
  return [
    `register[0,r,l] = F32(0), r=0..${schedule.registerCount - 1}, l=0..${schedule.lanesPerRegister - 1}`,
    `mapped_register(i) = floor((i mod ${schedule.laneCount})/${schedule.lanesPerRegister}); mapped_lane(i) = i mod ${schedule.lanesPerRegister}`,
    ...(completeEnd === 0 ? [] : [`register[i+1,r,l] = (r==mapped_register(i) && l==mapped_lane(i)) ? F32(register[i,r,l] + product[i]) : register[i,r,l], i=0..${completeEnd - 1} ascending; product[i] is exact until this F32 boundary`]),
    `tree_04[r,l] = F32(register[${completeEnd},r,l] + register[${completeEnd},r+4,l]), r=0..3, l=0..${schedule.lanesPerRegister - 1}`,
    `tree_02[r,l] = F32(tree_04[r,l] + tree_04[r+2,l]), r=0..1, l=0..${schedule.lanesPerRegister - 1}`,
    `tree_01[l] = F32(tree_02[0,l] + tree_02[1,l]), l=0..${schedule.lanesPerRegister - 1}`,
    ...foldAssignments("tree_01[{lane}]", schedule.lanesPerRegister, schedule.horizontalFold, "register_fold", "register_result"),
    `vector_tail[0,l] = F32(0), l=0..${schedule.lanesPerRegister - 1}`,
    ...(vectorTailBlocks === 0 ? [] : [`vector_tail[b+1,l] = F32(F32(vector_tail[b,l] + product[${completeEnd}+b*${vectorTailWidth}+l]) + product[${completeEnd}+b*${vectorTailWidth}+${schedule.lanesPerRegister}+l]), b=0..${vectorTailBlocks - 1} ascending, l=0..${schedule.lanesPerRegister - 1}; products are exact until each F32 boundary`]),
    ...foldAssignments(`vector_tail[${vectorTailBlocks},{lane}]`, schedule.lanesPerRegister, schedule.horizontalFold, "vector_tail_fold", "vector_tail_result"),
    "tail_acc[0] = F32(register_result + vector_tail_result)",
    ...(scalarTail === 0 ? [] : [`tail_acc[j+1] = F32(tail_acc[j] + product[${vectorTailEnd}+j]), j=0..${scalarTail - 1} ascending; product is exact until this F32 boundary`]),
    `reduced = tail_acc[${scalarTail}]`,
  ];
}

function armNeonBfdotAssignments(
  schedule: Extract<ReductionSchedule, { kind: "arm-neon-bf16-bfdot-fma" }>,
  width: number,
): string[] {
  const pairCount = Math.ceil(width / schedule.termsPerLane);
  const iterationTerms = schedule.activeRegisterCount * schedule.termsPerInstruction;
  return [
    `register[0,r,l] = F32(0), r=0..${schedule.registerCount - 1}, l=0..${schedule.lanesPerRegister - 1}`,
    `pair_register(p) = floor(((p*${schedule.termsPerLane}) mod ${iterationTerms})/${schedule.termsPerInstruction}); pair_lane(p) = floor((((p*${schedule.termsPerLane}) mod ${iterationTerms}) mod ${schedule.termsPerInstruction})/${schedule.termsPerLane})`,
    `register[p+1,r,l] = (r==pair_register(p) && l==pair_lane(p)) ? F32(F32(register[p,r,l] + product[p*${schedule.termsPerLane}]) + (p*${schedule.termsPerLane}+1<${width} ? product[p*${schedule.termsPerLane}+1] : F32(0))) : register[p,r,l], p=0..${pairCount - 1} ascending`,
    `tree_04[r,l] = F32(register[${pairCount},r,l] + register[${pairCount},r+4,l]), r=0..3, l=0..${schedule.lanesPerRegister - 1}`,
    `tree_02[r,l] = F32(tree_04[r,l] + tree_04[r+2,l]), r=0..1, l=0..${schedule.lanesPerRegister - 1}`,
    `tree_01[l] = F32(tree_02[0,l] + tree_02[1,l]), l=0..${schedule.lanesPerRegister - 1}`,
    ...foldAssignments("tree_01[{lane}]", schedule.lanesPerRegister, schedule.horizontalFold, "register_fold", "reduced"),
  ];
}

function foldAssignments(
  source: string,
  count: number,
  order: "ascending" | "descending" | "balanced-pairwise" | "pairwise",
  prefix: string,
  result: string,
): string[] {
  if (order === "ascending" || order === "descending") {
    const sourceIndex = order === "ascending" ? "j" : `${count - 1}-j`;
    return [
      `${prefix}[0] = F32(0)`,
      `${prefix}[j+1] = F32(${prefix}[j] + ${source.replace("{lane}", sourceIndex)}), j=0..${count - 1} ascending over source lanes in ${order} order`,
      `${result} = ${prefix}[${count}]`,
    ];
  }
  const assignments = [`${prefix}_0[j] = ${source.replace("{lane}", "j")}, j=0..${count - 1}`];
  let current = count, level = 0;
  while (current > 1) {
    const next = Math.ceil(current / 2);
    assignments.push(`${prefix}_${level + 1}[j] = 2*j+1<${current} ? F32(${prefix}_${level}[2*j] + ${prefix}_${level}[2*j+1]) : ${prefix}_${level}[2*j], j=0..${next - 1}`);
    current = next;
    level += 1;
  }
  assignments.push(`${result} = ${prefix}_${level}[0]`);
  return assignments;
}
