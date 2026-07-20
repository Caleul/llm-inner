import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type { Gemma4LiteralCalculationScope } from "./gemma4-literal-domains.js";

export type Gemma4LiteralRuntimeReductionOperationClass =
  | "vision-attention-score"
  | "vision-attention-value"
  | "audio-content-attention-score"
  | "audio-position-attention-score"
  | "audio-attention-value";

export type Gemma4RuntimeReductionTowerParameter =
  | "attentionHeads"
  | "headDim"
  | "attentionChunkSize"
  | "attentionContextLeft"
  | "attentionContextRight";

export type Gemma4RuntimeReductionDimension =
  | { kind: "constant"; value: number }
  | { kind: "tensor-axis"; tensor: string; axis: number }
  | { kind: "tower-parameter"; name: Gemma4RuntimeReductionTowerParameter }
  | { kind: "add" | "subtract" | "multiply" | "ceil-divide" | "exact-divide"; left: Gemma4RuntimeReductionDimension; right: Gemma4RuntimeReductionDimension };

export type Gemma4RuntimeReductionInvocationStage =
  | { id: string; operation: "cast"; input: string; output: string; dtype: "BF16" | "F32" }
  | { id: string; operation: "reshape"; input: string; output: string; shape: Gemma4RuntimeReductionDimension[] }
  | { id: string; operation: "permute"; input: string; output: string; axes: number[] }
  | { id: string; operation: "pad-axis-zero"; input: string; output: string; axis: number; before: Gemma4RuntimeReductionDimension; after: Gemma4RuntimeReductionDimension }
  | { id: string; operation: "unfold"; input: string; output: string; axis: number; size: Gemma4RuntimeReductionDimension; step: Gemma4RuntimeReductionDimension }
  | { id: string; operation: "move-axis"; input: string; output: string; source: number; destination: number }
  | { id: string; operation: "contiguous"; input: string; output: string }
  | { id: string; operation: "matmul"; left: string; right: string; output: string }
  | { id: string; operation: "slice-axis"; input: string; output: string; axis: number; start: Gemma4RuntimeReductionDimension; endExclusive: Gemma4RuntimeReductionDimension };

export interface Gemma4RuntimeReductionInvocationProgram {
  kind: "gemma4-runtime-reduction-invocation-program";
  schemaVersion: 1;
  id: Gemma4LiteralRuntimeReductionOperationClass;
  scope: "vision" | "audio";
  graphOperation: string;
  orderedOperands: [
    { ordinal: 0; name: "operand_0"; role: string; dtype: "BF16" | "F32"; layout: string },
    { ordinal: 1; name: "operand_1"; role: string; dtype: "BF16" | "F32"; layout: string },
  ];
  towerParameters: Gemma4RuntimeReductionTowerParameter[];
  stages: Gemma4RuntimeReductionInvocationStage[];
  output: { value: string; dtype: "F32"; layout: string };
}

const c = (value: number): Gemma4RuntimeReductionDimension => ({ kind: "constant", value });
const axis = (tensor: string, value: number): Gemma4RuntimeReductionDimension => ({ kind: "tensor-axis", tensor, axis: value });
const tower = (name: Gemma4RuntimeReductionTowerParameter): Gemma4RuntimeReductionDimension => ({ kind: "tower-parameter", name });
const arithmetic = (
  kind: Extract<Gemma4RuntimeReductionDimension, { left: Gemma4RuntimeReductionDimension }>["kind"],
  left: Gemma4RuntimeReductionDimension,
  right: Gemma4RuntimeReductionDimension,
): Gemma4RuntimeReductionDimension => ({ kind, left, right });
const add = (left: Gemma4RuntimeReductionDimension, right: Gemma4RuntimeReductionDimension): Gemma4RuntimeReductionDimension => arithmetic("add", left, right);
const subtract = (left: Gemma4RuntimeReductionDimension, right: Gemma4RuntimeReductionDimension): Gemma4RuntimeReductionDimension => arithmetic("subtract", left, right);
const multiply = (left: Gemma4RuntimeReductionDimension, right: Gemma4RuntimeReductionDimension): Gemma4RuntimeReductionDimension => arithmetic("multiply", left, right);
const ceilDivide = (left: Gemma4RuntimeReductionDimension, right: Gemma4RuntimeReductionDimension): Gemma4RuntimeReductionDimension => arithmetic("ceil-divide", left, right);
const exactDivide = (left: Gemma4RuntimeReductionDimension, right: Gemma4RuntimeReductionDimension): Gemma4RuntimeReductionDimension => arithmetic("exact-divide", left, right);

const heads = tower("attentionHeads"), dim = tower("headDim"), chunk = tower("attentionChunkSize");
const past = subtract(tower("attentionContextLeft"), c(1));
const future = tower("attentionContextRight");
const context = add(add(chunk, past), future);

function queryStages(): Gemma4RuntimeReductionInvocationStage[] {
  const sequence = axis("operand_0", 1);
  const paddedSequence = multiply(ceilDivide(sequence, chunk), chunk);
  return [
    { id: "query_split_heads", operation: "reshape", input: "operand_0", output: "query_split", shape: [axis("operand_0", 0), sequence, heads, dim] },
    { id: "query_pad_to_chunk", operation: "pad-axis-zero", input: "query_split", output: "query_padded", axis: 1, before: c(0), after: subtract(paddedSequence, sequence) },
    { id: "query_chunk", operation: "reshape", input: "query_padded", output: "query_chunked", shape: [axis("query_padded", 0), exactDivide(axis("query_padded", 1), chunk), chunk, heads, dim] },
    { id: "query_bmm_layout", operation: "permute", input: "query_chunked", output: "query_bmm", axes: [0, 3, 1, 2, 4] },
  ];
}

function contextStages(input: "operand_1", role: "key" | "value"): Gemma4RuntimeReductionInvocationStage[] {
  const split = `${role}_split`, padded = `${role}_padded`, unfolded = `${role}_unfolded`, windows = `${role}_windows`;
  return [
    { id: `${role}_split_heads`, operation: "reshape", input, output: split, shape: [axis(input, 0), axis(input, 1), heads, dim] },
    { id: `${role}_context_pad`, operation: "pad-axis-zero", input: split, output: padded, axis: 1, before: past, after: add(future, subtract(chunk, c(1))) },
    { id: `${role}_context_unfold`, operation: "unfold", input: padded, output: unfolded, axis: 1, size: context, step: chunk },
    { id: `${role}_context_axis`, operation: "move-axis", input: unfolded, output: windows, source: 4, destination: 2 },
    { id: `${role}_context_contiguous`, operation: "contiguous", input: windows, output: `${role}_context`, },
  ];
}

export function gemma4RuntimeReductionInvocationPrograms(): Gemma4RuntimeReductionInvocationProgram[] {
  return [
    {
      kind: "gemma4-runtime-reduction-invocation-program", schemaVersion: 1,
      id: "vision-attention-score", scope: "vision", graphOperation: "attention-score-matmul",
      orderedOperands: [
        { ordinal: 0, name: "operand_0", role: "post-rope-query", dtype: "BF16", layout: "B,H,Q,D" },
        { ordinal: 1, name: "operand_1", role: "post-rope-key", dtype: "BF16", layout: "B,H,K,D" },
      ],
      towerParameters: ["attentionHeads", "headDim"],
      stages: [
        { id: "query_bf16", operation: "cast", input: "operand_0", output: "query_bf16", dtype: "BF16" },
        { id: "key_bf16", operation: "cast", input: "operand_1", output: "key_bf16", dtype: "BF16" },
        { id: "key_transpose", operation: "permute", input: "key_bf16", output: "key_transposed", axes: [0, 1, 3, 2] },
        { id: "native_matmul", operation: "matmul", left: "query_bf16", right: "key_transposed", output: "native_output" },
        { id: "output_f32", operation: "cast", input: "native_output", output: "output", dtype: "F32" },
      ],
      output: { value: "output", dtype: "F32", layout: "B,H,Q,K" },
    },
    {
      kind: "gemma4-runtime-reduction-invocation-program", schemaVersion: 1,
      id: "vision-attention-value", scope: "vision", graphOperation: "attention-value-matmul",
      orderedOperands: [
        { ordinal: 0, name: "operand_0", role: "attention-weights", dtype: "BF16", layout: "B,H,Q,K" },
        { ordinal: 1, name: "operand_1", role: "normalized-value", dtype: "BF16", layout: "B,H,K,D" },
      ],
      towerParameters: ["attentionHeads", "headDim"],
      stages: [
        { id: "weights_bf16", operation: "cast", input: "operand_0", output: "weights_bf16", dtype: "BF16" },
        { id: "value_bf16", operation: "cast", input: "operand_1", output: "value_bf16", dtype: "BF16" },
        { id: "native_matmul", operation: "matmul", left: "weights_bf16", right: "value_bf16", output: "native_context" },
        { id: "head_transpose", operation: "permute", input: "native_context", output: "transposed_context", axes: [0, 2, 1, 3] },
        { id: "contiguous_context", operation: "contiguous", input: "transposed_context", output: "contiguous_context" },
        { id: "merge_heads", operation: "reshape", input: "contiguous_context", output: "merged_context", shape: [axis("contiguous_context", 0), axis("contiguous_context", 1), multiply(heads, dim)] },
        { id: "output_f32", operation: "cast", input: "merged_context", output: "output", dtype: "F32" },
      ],
      output: { value: "output", dtype: "F32", layout: "B,Q,H*D" },
    },
    {
      kind: "gemma4-runtime-reduction-invocation-program", schemaVersion: 1,
      id: "audio-content-attention-score", scope: "audio", graphOperation: "chunked-attention-content-matmul",
      orderedOperands: [
        { ordinal: 0, name: "operand_0", role: "scaled-query", dtype: "F32", layout: "B,S,H*D" },
        { ordinal: 1, name: "operand_1", role: "scaled-key", dtype: "F32", layout: "B,S,H*D" },
      ],
      towerParameters: ["attentionHeads", "headDim", "attentionChunkSize", "attentionContextLeft", "attentionContextRight"],
      stages: [
        ...queryStages(), ...contextStages("operand_1", "key"),
        { id: "key_bmm_layout", operation: "permute", input: "key_context", output: "key_bmm", axes: [0, 3, 1, 4, 2] },
        { id: "native_matmul", operation: "matmul", left: "query_bmm", right: "key_bmm", output: "output" },
      ],
      output: { value: "output", dtype: "F32", layout: "B,H,blocks,chunk,context" },
    },
    {
      kind: "gemma4-runtime-reduction-invocation-program", schemaVersion: 1,
      id: "audio-position-attention-score", scope: "audio", graphOperation: "relative-attention-position-matmul",
      orderedOperands: [
        { ordinal: 0, name: "operand_0", role: "scaled-query", dtype: "F32", layout: "B,S,H*D" },
        { ordinal: 1, name: "operand_1", role: "relative-key", dtype: "F32", layout: "1,R,H*D" },
      ],
      towerParameters: ["attentionHeads", "headDim", "attentionChunkSize", "attentionContextLeft", "attentionContextRight"],
      stages: [
        ...queryStages(),
        { id: "query_flatten_chunks", operation: "reshape", input: "query_bmm", output: "query_flat", shape: [axis("query_bmm", 0), axis("query_bmm", 1), multiply(axis("query_bmm", 2), axis("query_bmm", 3)), axis("query_bmm", 4)] },
        { id: "relative_split_heads", operation: "reshape", input: "operand_1", output: "relative_split", shape: [axis("operand_1", 1), heads, dim] },
        { id: "relative_bmm_layout", operation: "permute", input: "relative_split", output: "relative_bmm", axes: [1, 2, 0] },
        { id: "native_matmul", operation: "matmul", left: "query_flat", right: "relative_bmm", output: "flat_output" },
        { id: "restore_chunks", operation: "reshape", input: "flat_output", output: "output", shape: [axis("query_bmm", 0), axis("query_bmm", 1), axis("query_bmm", 2), axis("query_bmm", 3), axis("relative_bmm", 2)] },
      ],
      output: { value: "output", dtype: "F32", layout: "B,H,blocks,chunk,R" },
    },
    {
      kind: "gemma4-runtime-reduction-invocation-program", schemaVersion: 1,
      id: "audio-attention-value", scope: "audio", graphOperation: "chunked-relative-attention-values",
      orderedOperands: [
        { ordinal: 0, name: "operand_0", role: "attention-weights", dtype: "F32", layout: "B,H,blocks,chunk,context" },
        { ordinal: 1, name: "operand_1", role: "value", dtype: "F32", layout: "B,S,H*D" },
      ],
      towerParameters: ["attentionHeads", "headDim", "attentionChunkSize", "attentionContextLeft", "attentionContextRight"],
      stages: [
        ...contextStages("operand_1", "value"),
        { id: "value_bmm_layout", operation: "permute", input: "value_context", output: "value_bmm", axes: [0, 3, 1, 2, 4] },
        { id: "native_matmul", operation: "matmul", left: "operand_0", right: "value_bmm", output: "native_context" },
        { id: "context_sequence_layout", operation: "permute", input: "native_context", output: "sequence_context", axes: [0, 2, 3, 1, 4] },
        { id: "merge_context_heads", operation: "reshape", input: "sequence_context", output: "merged_context", shape: [axis("sequence_context", 0), multiply(axis("sequence_context", 1), axis("sequence_context", 2)), multiply(heads, dim)] },
        { id: "trim_padding", operation: "slice-axis", input: "merged_context", output: "output", axis: 1, start: c(0), endExclusive: axis("operand_1", 1) },
      ],
      output: { value: "output", dtype: "F32", layout: "B,S,H*D" },
    },
  ];
}

export function validateGemma4RuntimeReductionInvocationPrograms(programs: readonly Gemma4RuntimeReductionInvocationProgram[]): void {
  if (!isDeepStrictEqual(programs, gemma4RuntimeReductionInvocationPrograms())) {
    throw new Error("Programas de invocação das reduções Gemma 4 estão incompletos ou divergentes.");
  }
}

export function gemma4RuntimeReductionInvocationProgram(
  programs: readonly Gemma4RuntimeReductionInvocationProgram[],
  scope: Gemma4LiteralCalculationScope,
  operation: string,
): Gemma4RuntimeReductionInvocationProgram {
  const matches = programs.filter((program) => program.scope === scope && program.graphOperation === operation);
  if (matches.length !== 1) throw new Error(`Redução runtime-defined inesperada em ${scope}:${operation}; programa de invocação único ausente.`);
  return matches[0]!;
}

export function gemma4RuntimeReductionInvocationProgramSha256(program: Gemma4RuntimeReductionInvocationProgram): string {
  return createHash("sha256").update(JSON.stringify(program), "utf8").digest("hex");
}

export function gemma4LiteralRuntimeReductionOperationClass(
  scope: Gemma4LiteralCalculationScope,
  operation: string,
): Gemma4LiteralRuntimeReductionOperationClass {
  return gemma4RuntimeReductionInvocationProgram(gemma4RuntimeReductionInvocationPrograms(), scope, operation).id;
}
