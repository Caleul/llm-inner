import type { ReductionSchedule } from "./types.js";

type CascadeReduction = Extract<ReductionSchedule, { kind: "pytorch-cpu-f32-cascade-sum" }>;
type WelfordReduction = Extract<ReductionSchedule, { kind: "pytorch-cpu-bf16-welford" }>;

/**
 * Exact indexed transcript of PyTorch's contiguous F32 cascade reduction used
 * by every native Gemma 4 RMSNorm. `square[i]` is declared by the caller so
 * this state machine is shared by text, vision and audio without hiding the
 * coordinate-to-input mapping.
 */
export function buildGemma4LiteralCascadeSquareReductionAssignments(
  schedule: CascadeReduction,
  width: number,
): string[] {
  assertCascadeSchedule(schedule, width);
  const unitWidth = schedule.vectorLanes * schedule.ilpFactor;
  const unitCount = width / unitWidth;
  const levelPower = Math.max(Math.log2(schedule.minimumLevelStep), Math.ceil(Math.log2(unitCount)) >> 2);
  const levelStep = 2 ** levelPower;
  const levelMask = levelStep - 1;
  return [
    `unit_width = ${unitWidth}; unit_count = ${unitCount}; level_power = ${levelPower}; level_step = ${levelStep}; level_mask = ${levelMask}`,
    `cascade[level,register,lane] = F32(0), level=0..${schedule.cascadeLevels - 1}, register=0..${schedule.ilpFactor - 1}, lane=0..${schedule.vectorLanes - 1}`,
    "unit = 0",
    `while unit+level_step<=unit_count in ascending chunk order: for local=0..level_step-1 ascending, cascade[0,register,lane]=F32(cascade[0,register,lane]+square[unit*unit_width+register*${schedule.vectorLanes}+lane]) for register=0..${schedule.ilpFactor - 1}, lane=0..${schedule.vectorLanes - 1}; then unit=unit+1`,
    `after each complete chunk: for level=1..${schedule.cascadeLevels - 1} ascending, cascade[level,register,lane]=F32(cascade[level,register,lane]+cascade[level-1,register,lane]) and cascade[level-1,register,lane]=F32(0) for every register,lane; stop this level loop after the first level where (unit & (level_mask << (level*level_power))) != 0`,
    `for remaining unit while unit<unit_count ascending: cascade[0,register,lane]=F32(cascade[0,register,lane]+square[unit*unit_width+register*${schedule.vectorLanes}+lane]) for register=0..${schedule.ilpFactor - 1}, lane=0..${schedule.vectorLanes - 1}; then unit=unit+1`,
    `for level=1..${schedule.cascadeLevels - 1} ascending, register=0..${schedule.ilpFactor - 1}, lane=0..${schedule.vectorLanes - 1}: cascade[0,register,lane]=F32(cascade[0,register,lane]+cascade[level,register,lane])`,
    `for register=1..${schedule.ilpFactor - 1} ascending, lane=0..${schedule.vectorLanes - 1}: cascade[0,0,lane]=F32(cascade[0,0,lane]+cascade[0,register,lane])`,
    "lane_acc[0] = F32(0)",
    `lane_acc[lane+1] = F32(lane_acc[lane]+cascade[0,0,lane]), lane=0..${schedule.vectorLanes - 1} ascending`,
    `sum = lane_acc[${schedule.vectorLanes}]`,
  ];
}

/**
 * Exact scalar state transitions for PyTorch's reduced-precision ARM
 * RowwiseMoments kernel. The caller supplies the concrete indexed `x[c]`
 * declaration and the learned gamma literal used by the second pass.
 */
export function buildGemma4LiteralWelfordAssignments(
  schedule: WelfordReduction,
  channels: number,
): string[] {
  assertWelfordSchedule(schedule, channels);
  const vectorCount = channels / schedule.inputVectorLanes;
  const mergedCount = vectorCount * 2;
  return [
    `vector_count = ${vectorCount}; merged_lane_count = ${mergedCount}`,
    `low_mean[0,lane]=F32(0); low_m2[0,lane]=F32(0); high_mean[0,lane]=F32(0); high_m2[0,lane]=F32(0), lane=0..${schedule.accumulatorVectorLanes - 1}`,
    `reciprocal[v] = F32(1/F32(v+1)), v=0..${vectorCount - 1} ascending`,
    `low_delta[v,lane]=F32(x[v*${schedule.inputVectorLanes}+lane]-low_mean[v,lane]); low_mean[v+1,lane]=F32(low_mean[v,lane]+low_delta[v,lane]*reciprocal[v]); low_remainder[v,lane]=F32(x[v*${schedule.inputVectorLanes}+lane]-low_mean[v+1,lane]); low_m2[v+1,lane]=F32(low_m2[v,lane]+low_delta[v,lane]*low_remainder[v,lane]), v=0..${vectorCount - 1} ascending, lane=0..${schedule.accumulatorVectorLanes - 1}`,
    `high_delta[v,lane]=F32(x[v*${schedule.inputVectorLanes}+${schedule.accumulatorVectorLanes}+lane]-high_mean[v,lane]); high_mean[v+1,lane]=F32(high_mean[v,lane]+high_delta[v,lane]*reciprocal[v]); high_remainder[v,lane]=F32(x[v*${schedule.inputVectorLanes}+${schedule.accumulatorVectorLanes}+lane]-high_mean[v+1,lane]); high_m2[v+1,lane]=F32(high_m2[v,lane]+high_delta[v,lane]*high_remainder[v,lane]), v=0..${vectorCount - 1} ascending, lane=0..${schedule.accumulatorVectorLanes - 1}`,
    `merge_delta[lane]=F32(high_mean[${vectorCount},lane]-low_mean[${vectorCount},lane]); merge_ratio_delta[lane]=F32(F32(0.5)*merge_delta[lane]), lane=0..${schedule.accumulatorVectorLanes - 1}`,
    `merged_mean[lane]=F32(low_mean[${vectorCount},lane]+merge_ratio_delta[lane]); merged_m2[lane]=F32(F32(merge_delta[lane]*F32(${vectorCount}))*merge_ratio_delta[lane]+F32(low_m2[${vectorCount},lane]+high_m2[${vectorCount},lane])), lane=0..${schedule.accumulatorVectorLanes - 1}`,
    "fold_count[0]=0; fold_mean[0]=F32(0); fold_m2[0]=F32(0)",
    `fold_total[lane]=fold_count[lane]+${mergedCount}; fold_ratio[lane]=F32(${mergedCount}/fold_total[lane]); fold_delta[lane]=F32(merged_mean[lane]-fold_mean[lane]); fold_prior_delta[lane]=F32(fold_delta[lane]*F32(fold_count[lane])); fold_scaled_delta[lane]=F32(fold_ratio[lane]*fold_delta[lane]), lane=0..${schedule.accumulatorVectorLanes - 1} ascending`,
    `fold_mean[lane+1]=F32(fold_mean[lane]+fold_scaled_delta[lane]); fold_m2[lane+1]=F32(fold_prior_delta[lane]*fold_scaled_delta[lane]+F32(fold_m2[lane]+merged_m2[lane])); fold_count[lane+1]=fold_total[lane], lane=0..${schedule.accumulatorVectorLanes - 1} ascending`,
    `mean = fold_mean[${schedule.accumulatorVectorLanes}]`,
    `variance = F32(fold_m2[${schedule.accumulatorVectorLanes}]/F32(${channels}))`,
  ];
}

/** Normative generic programs embedded in schema-v15's formula language. */
export function gemma4LiteralNormalizationReductionPrograms(): {
  pytorchCpuF32CascadeSum: string[];
  pytorchCpuBf16Welford: string[];
} {
  return {
    pytorchCpuF32CascadeSum: [
      "unit_width=schedule.vectorLanes*schedule.ilpFactor; unit_count=width/unit_width; level_power=max(log2(schedule.minimumLevelStep),ceil(log2(unit_count))>>2); level_step=2**level_power; level_mask=level_step-1",
      "cascade[level,register,lane]=F32(0), level=0..schedule.cascadeLevels-1, register=0..schedule.ilpFactor-1, lane=0..schedule.vectorLanes-1; unit=0",
      "while unit+level_step<=unit_count: for local=0..level_step-1 ascending, cascade[0,register,lane]=F32(cascade[0,register,lane]+square[unit*unit_width+register*schedule.vectorLanes+lane]) for register,lane ascending; unit=unit+1",
      "after each complete chunk: for level=1..schedule.cascadeLevels-1 ascending, cascade[level,register,lane]=F32(cascade[level,register,lane]+cascade[level-1,register,lane]); cascade[level-1,register,lane]=F32(0) for every register,lane; break after the first level where (unit & (level_mask << (level*level_power))) != 0",
      "while unit<unit_count ascending: cascade[0,register,lane]=F32(cascade[0,register,lane]+square[unit*unit_width+register*schedule.vectorLanes+lane]) for register,lane ascending; unit=unit+1",
      "for level=1..schedule.cascadeLevels-1 ascending, register,lane ascending: cascade[0,register,lane]=F32(cascade[0,register,lane]+cascade[level,register,lane])",
      "for register=1..schedule.ilpFactor-1 ascending, lane ascending: cascade[0,0,lane]=F32(cascade[0,0,lane]+cascade[0,register,lane])",
      "lane_acc[0]=F32(0); lane_acc[lane+1]=F32(lane_acc[lane]+cascade[0,0,lane]), lane=0..schedule.vectorLanes-1 ascending; sum=lane_acc[schedule.vectorLanes]",
    ],
    pytorchCpuBf16Welford: [
      "vector_count=width/schedule.inputVectorLanes; low_mean[0,lane]=low_m2[0,lane]=high_mean[0,lane]=high_m2[0,lane]=F32(0), lane=0..schedule.accumulatorVectorLanes-1",
      "reciprocal[v]=F32(1/F32(v+1)), v=0..vector_count-1 ascending",
      "low_delta[v,lane]=F32(x[v*schedule.inputVectorLanes+lane]-low_mean[v,lane]); low_mean[v+1,lane]=F32(low_mean[v,lane]+low_delta[v,lane]*reciprocal[v]); low_remainder[v,lane]=F32(x[v*schedule.inputVectorLanes+lane]-low_mean[v+1,lane]); low_m2[v+1,lane]=F32(low_m2[v,lane]+low_delta[v,lane]*low_remainder[v,lane])",
      "high_delta[v,lane]=F32(x[v*schedule.inputVectorLanes+schedule.accumulatorVectorLanes+lane]-high_mean[v,lane]); high_mean[v+1,lane]=F32(high_mean[v,lane]+high_delta[v,lane]*reciprocal[v]); high_remainder[v,lane]=F32(x[v*schedule.inputVectorLanes+schedule.accumulatorVectorLanes+lane]-high_mean[v+1,lane]); high_m2[v+1,lane]=F32(high_m2[v,lane]+high_delta[v,lane]*high_remainder[v,lane])",
      "merge_delta[lane]=F32(high_mean[vector_count,lane]-low_mean[vector_count,lane]); merge_ratio_delta[lane]=F32(F32(0.5)*merge_delta[lane]); merged_mean[lane]=F32(low_mean[vector_count,lane]+merge_ratio_delta[lane]); merged_m2[lane]=F32(F32(merge_delta[lane]*F32(vector_count))*merge_ratio_delta[lane]+F32(low_m2[vector_count,lane]+high_m2[vector_count,lane]))",
      "fold_count[0]=0; fold_mean[0]=F32(0); fold_m2[0]=F32(0); fold_total[lane]=fold_count[lane]+2*vector_count; fold_ratio[lane]=F32((2*vector_count)/fold_total[lane]); fold_delta[lane]=F32(merged_mean[lane]-fold_mean[lane]); fold_prior_delta[lane]=F32(fold_delta[lane]*F32(fold_count[lane])); fold_scaled_delta[lane]=F32(fold_ratio[lane]*fold_delta[lane])",
      "fold_mean[lane+1]=F32(fold_mean[lane]+fold_scaled_delta[lane]); fold_m2[lane+1]=F32(fold_prior_delta[lane]*fold_scaled_delta[lane]+F32(fold_m2[lane]+merged_m2[lane])); fold_count[lane+1]=fold_total[lane], lane=0..schedule.accumulatorVectorLanes-1 ascending",
      "mean=fold_mean[schedule.accumulatorVectorLanes]; variance=F32(fold_m2[schedule.accumulatorVectorLanes]/F32(width)); inv_std=F32(1/ARM_SQRT_F32(F32(variance+epsilon))); bias=F32(-inv_std*mean); result=F32(F32(F32(x*inv_std)+bias)*gamma); output=assignment_output_cast(result)",
    ],
  };
}

function assertCascadeSchedule(schedule: CascadeReduction, width: number): void {
  if (!Number.isSafeInteger(width) || width <= 0 || width % 16 !== 0 || schedule.vectorLanes !== 4 ||
    schedule.ilpFactor !== 4 || schedule.cascadeLevels !== 4 || schedule.minimumLevelStep !== 16 ||
    schedule.registerFold !== "ascending" || schedule.laneFold !== "ascending") {
    throw new Error(`Agenda literal PyTorch CPU cascade inválida para width=${width}.`);
  }
}

function assertWelfordSchedule(schedule: WelfordReduction, channels: number): void {
  if (!Number.isSafeInteger(channels) || channels <= 0 || channels % schedule.inputVectorLanes !== 0 ||
    channels / schedule.inputVectorLanes > schedule.chunkVectors || schedule.inputVectorLanes !== 8 ||
    schedule.accumulatorVectorLanes !== 4 || schedule.chunkVectors !== 16 ||
    schedule.vectorMergeOrder !== "low-then-high" || schedule.laneFold !== "ascending" ||
    schedule.secondPass !== "x-times-scale-plus-bias-times-gamma") {
    throw new Error(`Agenda literal PyTorch CPU BF16 Welford inválida para channels=${channels}.`);
  }
}
