import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { open, rm } from "node:fs/promises";
import type { Gemma4ParametricExactRealProgram, Gemma4ParametricOutputFunction } from "./gemma4-parametric-global-real-program.js";
import { parametricNodeDependencies, type Gemma4ParametricRealNode } from "./gemma4-parametric-real-expression.js";
import type { OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";

export interface Gemma4RealFlatExpansionEstimate {
  family: string;
  dimension: number;
  symbolicNodeOccurrences: string;
  decimalDigits: number;
  conservativeMinimumBytes: string;
  reductionsUnrolled: false;
}

/** Counts literal substitutions. Finite reductions remain symbolic, making this a strict conservative estimate. */
export function estimateGemma4FlatExpansion(program: Gemma4ParametricExactRealProgram, output: Gemma4ParametricOutputFunction): Gemma4RealFlatExpansionEstimate {
  const nodes = new Map(program.expressionGraph.nodes.map((node) => [node.id, node]));
  const functions = new Map(program.operationFunctions.map((entry) => [entry.functionId, entry]));
  const memo = new Map<string, bigint>(), active = new Set<string>();
  const count = (id: string): bigint => {
    const cached = memo.get(id); if (cached !== undefined) return cached;
    if (active.has(id)) throw new Error(`${id}: ciclo durante estimativa de expansão.`);
    const node = nodes.get(id); if (!node) throw new Error(`${id}: nó ausente durante estimativa de expansão.`);
    active.add(id);
    let result = 1n;
    for (const dependency of parametricNodeDependencies(node)) result += count(dependency);
    if (node.kind === "function-call") {
      const function_ = functions.get(node.functionId); if (!function_) throw new Error(`${node.functionId}: função ausente durante estimativa.`);
      result += count(function_.root);
    }
    active.delete(id); memo.set(id, result); return result;
  };
  const occurrences = count(output.root);
  return {
    family: output.name, dimension: output.fixedDimension, symbolicNodeOccurrences: occurrences.toString(),
    decimalDigits: occurrences.toString().length, conservativeMinimumBytes: (occurrences * 2n).toString(), reductionsUnrolled: false,
  };
}

export function gemma4SsaExpression(node: Gemma4ParametricRealNode): string {
  const list = (operator: string, values: readonly string[]) => `(${values.join(` ${operator} `)})`;
  switch (node.kind) {
    case "integer-constant": return String(node.value);
    case "integer-parameter": case "integer-bound-index": return node.name;
    case "integer-add": return `(${node.left} + ${node.right})`; case "integer-subtract": return `(${node.left} - ${node.right})`;
    case "integer-multiply": return `(${node.left} * ${node.right})`; case "integer-floor-divide": return `floor(${node.left} / ${node.right})`;
    case "integer-modulo": return `mod(${node.left}, ${node.right})`; case "tensor-axis": return `axis(${JSON.stringify(node.tensor)}, ${node.axis})`;
    case "stable-true-prefix-rank": return `stable_true_prefix_rank(${JSON.stringify(node.tensor)}, ${node.equals}, ${node.batch}, ${node.sequence})`;
    case "rational": return `${node.value.numerator}/${node.value.denominator}`; case "boolean": return String(node.value);
    case "negative-infinity": return "-Infinity"; case "positive-infinity": return "Infinity";
    case "input-element": return `input(${JSON.stringify(node.tensor)}, [${node.coordinates.join(", ")}])`;
    case "learned-rational-element": return `constant(${JSON.stringify(node.tensor)}, [${node.coordinates.join(", ")}], ${JSON.stringify(node.decoderId)})`;
    case "function-call": return `${node.functionId}(${node.arguments.join(", ")})`;
    case "add": return list("+", node.arguments); case "multiply": return list("*", node.arguments);
    case "minimum": return `min(${node.arguments.join(", ")})`; case "maximum": return `max(${node.arguments.join(", ")})`;
    case "divide": return `(${node.numerator} / ${node.denominator})`; case "integer-power": return `(${node.base} ** ${node.exponent})`;
    case "power": return `(${node.base} ** ${node.exponent})`; case "unary-function": return `${node.function}(${node.argument})`;
    case "compare": return `(${node.left} ${comparisonOperator(node.comparison)} ${node.right})`;
    case "select": return `select(${node.condition}, ${node.whenTrue}, ${node.whenFalse})`;
    case "finite-sum": case "finite-maximum": return `${node.kind === "finite-sum" ? "sum" : "reduce_max"}(${node.index}=${node.startInclusive}..${node.endExclusive}, ${node.body}${node.predicate ? `, if=${node.predicate}` : ""})`;
  }
}

/** Streams the 270 MB instruction section without constructing another giant JSON string. */
export async function writeGemma4RealSsaExport(artifact: OpenGemma4CompositeLiteralArtifact, outputPath: string): Promise<{ bytes: number; sha256: string; estimates: Gemma4RealFlatExpansionEstimate[] }> {
  const handle = await open(outputPath, "wx"); await handle.close();
  const stream = createWriteStream(outputPath, { flags: "w" }); const hash = createHash("sha256"); let bytes = 0;
  const write = async (value: string): Promise<void> => { const data = Buffer.from(value); hash.update(data); bytes += data.length; if (!stream.write(data)) await new Promise<void>((accept) => stream.once("drain", accept)); };
  const program = artifact.realSimplifiedProgram;
  const representativeOutputs = [...new Set(program.outputFunctions.map((entry) => entry.name))].map((name) => program.outputFunctions.find((entry) => entry.name === name)!);
  const estimates = representativeOutputs.map((entry) => estimateGemma4FlatExpansion(program, entry));
  try {
    await write(`{"kind":"gemma4-real-simplified-ssa","schemaVersion":1,"semantics":${JSON.stringify(program.semantics)},`);
    await write(`"constantMemory":{"kind":"authenticated-gemma4-literal-artifact","artifact":${JSON.stringify(artifact.artifact)},"artifactBytes":${artifact.artifactBytes},"sourceIdentity":${JSON.stringify(artifact.sourceIdentity)},"integrityManifest":${JSON.stringify(artifact.integrityManifest)}},`);
    await write(`"inputBoundaries":${JSON.stringify(program.inputBoundaries)},"flatExpansionEstimates":${JSON.stringify(estimates)},"statements":[`);
    for (let index = 0; index < program.expressionGraph.nodes.length; index += 1) {
      const node = program.expressionGraph.nodes[index]!;
      if (index) await write(","); await write(JSON.stringify({ target: node.id, expression: gemma4SsaExpression(node), node }));
    }
    await write(`],"functions":${JSON.stringify(program.operationFunctions)},"outputs":[`);
    for (let index = 0; index < program.outputFunctions.length; index += 1) {
      const entry = program.outputFunctions[index]!; if (index) await write(",");
      await write(JSON.stringify({ assignment: `calc_${entry.name}_${entry.fixedDimension}`, value: entry.root, parameters: entry.parameters, coordinate: entry.coordinate, finalQuantization: entry.finalQuantization }));
    }
    await write(`],"coverage":${JSON.stringify(program.coverage)}}\n`);
    await new Promise<void>((accept, reject) => { stream.once("error", reject); stream.end(accept); });
    return { bytes, sha256: hash.digest("hex"), estimates };
  } catch (error) {
    stream.destroy();
    await new Promise<void>((accept) => stream.closed ? accept() : stream.once("close", accept));
    await rm(outputPath, { force: true });
    throw error;
  }
}

function comparisonOperator(value: "equal" | "not-equal" | "less" | "less-equal" | "greater" | "greater-equal"): string {
  return ({ equal: "==", "not-equal": "!=", less: "<", "less-equal": "<=", greater: ">", "greater-equal": ">=" } as const)[value];
}
