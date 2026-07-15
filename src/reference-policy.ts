import type { ModelIR, Operation } from "./types.js";

/**
 * The scalar executors are an explicit candidate policy, never an inferred
 * property of a checkpoint's storage. Keeping this mutation in one place
 * prevents trace replay and performance measurement from drifting apart.
 */
export function applyReferenceF32Policy(ir: ModelIR): void {
  for (const operation of allOperations(ir)) {
    operation.dtypePolicy = { computeDtype: "F32", accumulationDtype: "F32", outputDtype: "F32" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F32";
  }
}

export function applyReferenceF64Policy(ir: ModelIR): void {
  for (const operation of allOperations(ir)) {
    operation.dtypePolicy = { computeDtype: "F64", accumulationDtype: "F64", outputDtype: "F64" };
    if (operation.op === "scaled_dot_product_attention") operation.softmaxComputeDtype = "F64";
  }
}

function allOperations(ir: ModelIR): Operation[] {
  return [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue];
}
