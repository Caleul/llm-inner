import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { buildModelIR, resolveModelContextLimit } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { pruneFixedF16ScalarDeclarations } from "./fixed-f16-source-prune.js";
import { inlineNextFixedF16ScalarCache } from "./fixed-f16-cache-inline.js";
import type { Operation } from "./types.js";
import {
  compileFixedF16CachedAttentionSource,
  compileFixedF16CachedEmbeddingSource, compileFixedF16CachedMlpSource,
  compileFixedF16CachedWideLinearSource, compileFixedF16OutputRowFormula,
  compileFixedF16RmsNormParametricFormulas,
  composeCachedScalarSource, composeCachedScalarSources, scalarSourceFromFactoredRmsNorm, scalarSourceFromFormulas,
  substituteFixedF16ParametricFormulas,
  type FixedF16CachedScalarSource,
} from "./fixed-f16-parametric-formulas.js";

type Linear = Extract<Operation, { op: "linear" }>;
type Norm = Extract<Operation, { op: "rms_norm" }>;
type Rotary = Extract<Operation, { op: "rotary_embedding" }>;
type Attention = Extract<Operation, { op: "scaled_dot_product_attention" }>;
type Elementwise = Extract<Operation, { op: "elementwise" }>;

function requireForward(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(`Forward F16 não suportado: ${message}`);
}

function tensor(op: Linear | Norm): string {
  requireForward(op.weight?.storageDtype === "F16", `${op.id}: peso F16 ausente`);
  return op.weight.name;
}

function linear(op: Operation): Linear {
  requireForward(op.op === "linear" && !op.bias && op.transposeWeight,
    `${op.id}: projeção sem bias/transposta necessária`);
  tensor(op);
  return op;
}

function norm(op: Operation): Norm {
  requireForward(op.op === "rms_norm" && op.weightTransform === "direct" && op.axis === -1,
    `${op.id}: RMSNorm direta no último eixo necessária`);
  tensor(op);
  return op;
}

/** The compiled model starts at the vectors entering its first transformer layer. */
export type FixedF16InputBoundary = "embeddings" | "token-ids";

/** Lower a supported ordered decoder forward into a checkpoint-independent scalar function. */
export async function compileFixedF16ScalarModelFromDirectory(
  directory: string, inputBoundary: FixedF16InputBoundary = "embeddings", outputDimension?: number,
): Promise<FixedF16CachedScalarSource> {
  const opened = await openCatalog(directory, false);
  const reader = new SafetensorsCatalogReader(directory);
  try {
    requireForward(opened.catalog.format === "safetensors", "contêiner Safetensors necessário");
    const ir = await buildModelIR(opened.catalog, { outputRows: 0, inputTerms: 0, includeWeights: false });
    requireForward(ir.prelude.length === 1 && ir.prelude[0]?.op === "embedding" &&
      ir.prelude[0].scale === undefined && ir.prelude[0].weight.storageDtype === "F16",
    "embedding F16 sem escala necessário");
    requireForward(inputBoundary === "embeddings" || inputBoundary === "token-ids", "fronteira de entrada inválida");
    const maxSequenceLength = resolveModelContextLimit(ir.config, ir.architecture.modelType);
    requireForward(typeof maxSequenceLength === "number" && Number.isSafeInteger(maxSequenceLength) &&
      maxSequenceLength > 0, "limite de contexto definido pela configuração necessário");
    let source: FixedF16CachedScalarSource | undefined;
    for (const layer of ir.layers) {
      const ops = layer.operations;
      requireForward(ops.map((op) => op.op).join(",") === [
        "rms_norm", "linear", "linear", "linear", "reshape_heads", "reshape_heads", "reshape_heads",
        "rotary_embedding", "rotary_embedding", "scaled_dot_product_attention", "linear", "elementwise",
        "rms_norm", "linear", "linear", "activation", "elementwise", "linear", "elementwise",
      ].join(","), `camada ${layer.index}: sequência de operações não suportada`);
      const inputNorm = norm(ops[0]!);
      const q = linear(ops[1]!), k = linear(ops[2]!), v = linear(ops[3]!);
      const qHeads = ops[4]!, kHeads = ops[5]!, vHeads = ops[6]!;
      const qRope = ops[7] as Rotary, kRope = ops[8] as Rotary;
      const attention = ops[9] as Attention;
      const o = linear(ops[10]!);
      const firstAdd = ops[11] as Elementwise;
      const postNorm = norm(ops[12]!);
      const gate = linear(ops[13]!), up = linear(ops[14]!);
      const activation = ops[15]!;
      const multiply = ops[16] as Elementwise;
      const down = linear(ops[17]!);
      const secondAdd = ops[18] as Elementwise;
      requireForward(q.input === inputNorm.output && k.input === inputNorm.output && v.input === inputNorm.output &&
        qHeads.op === "reshape_heads" && kHeads.op === "reshape_heads" && vHeads.op === "reshape_heads" &&
        qHeads.input === q.output && kHeads.input === k.output && vHeads.input === v.output &&
        qHeads.layout === "BHSD" && kHeads.layout === "BHSD" && vHeads.layout === "BHSD" &&
        qHeads.numHeads === attention.numAttentionHeads &&
        kHeads.numHeads === attention.numKeyValueHeads && vHeads.numHeads === attention.numKeyValueHeads &&
        qHeads.headDim === attention.headDim && kHeads.headDim === attention.headDim &&
        vHeads.headDim === attention.headDim &&
        qRope.layout === "rotate_half" && kRope.layout === "rotate_half" &&
        qRope.input === qHeads.output && kRope.input === kHeads.output &&
        qRope.rotaryDim === attention.headDim && kRope.rotaryDim === attention.headDim &&
        attention.query === qRope.output && attention.key === kRope.output && attention.value === vHeads.output &&
        qRope.theta === kRope.theta && !qRope.scaling && !kRope.scaling &&
        attention.causal && !attention.slidingWindow && !attention.scoreSoftcap &&
        attention.scale === 1 / Math.sqrt(attention.headDim) &&
        o.input === attention.output && firstAdd.kind === "add" &&
        firstAdd.inputs.includes(inputNorm.input) && firstAdd.inputs.includes(o.output) &&
        postNorm.input === firstAdd.output && gate.input === postNorm.output && up.input === postNorm.output &&
        activation.op === "activation" && activation.function === "silu" && activation.input === gate.output &&
        multiply.kind === "multiply" && multiply.inputs.includes(activation.output) && multiply.inputs.includes(up.output) &&
        down.input === multiply.output && secondAdd.kind === "add" &&
        secondAdd.inputs.includes(firstAdd.output) && secondAdd.inputs.includes(down.output),
      `camada ${layer.index}: dependências ou semântica numérica divergentes`);
      const compiledNorm = await compileFixedF16RmsNormParametricFormulas(reader, tensor(inputNorm), inputNorm.epsilon);
      const compiledAttention = await compileFixedF16CachedAttentionSource(reader,
        { q: tensor(q), k: tensor(k), v: tensor(v), o: tensor(o) }, attention.numAttentionHeads,
        attention.numKeyValueHeads, attention.headDim, qRope.theta, "f32-interleaved-four-lane-pairwise");
      const normalizedAttention = composeCachedScalarSources(compiledAttention, scalarSourceFromFactoredRmsNorm(compiledNorm));
      const firstResidual = { ...normalizedAttention, formulas: normalizedAttention.formulas.map((formula, dimension) =>
        `add16(x[t][${dimension}],${formula})`) };
      const compiledPostNorm = await compileFixedF16RmsNormParametricFormulas(reader, tensor(postNorm), postNorm.epsilon);
      const compiledMlp = await compileFixedF16CachedMlpSource(reader,
        { gate: tensor(gate), up: tensor(up), down: tensor(down) }, "f32-interleaved-four-lane-pairwise");
      const postMlp = composeCachedScalarSources(compiledMlp, scalarSourceFromFactoredRmsNorm(compiledPostNorm));
      const secondResidual = { ...postMlp, formulas: postMlp.formulas.map((formula, dimension) =>
        `add16(x[t][${dimension}],${formula})`) };
      const local = composeCachedScalarSources(secondResidual, firstResidual);
      source = source ? composeCachedScalarSources(local, source) : local;
    }
    requireForward(source && ir.epilogue.length === 2, "normalização final e projeção de logits necessárias");
    const finalNorm = norm(ir.epilogue[0]!);
    const head = linear(ir.epilogue[1]!);
    requireForward(head.input === finalNorm.output, "projeção final não recebe a RMSNorm final");
    const finalNormFormula = await compileFixedF16RmsNormParametricFormulas(
      reader, tensor(finalNorm), finalNorm.epsilon);
    let logits: FixedF16CachedScalarSource;
    if (outputDimension === undefined) {
      const normalized = composeCachedScalarSources(scalarSourceFromFactoredRmsNorm(finalNormFormula), source);
      logits = composeCachedScalarSources(await compileFixedF16CachedWideLinearSource(reader, tensor(head)), normalized);
    } else {
      requireForward(finalNormFormula.rmsFactor, "fator RMSNorm final necessário");
      const factor = finalNormFormula.rmsFactor;
      const compactNorm = { ...finalNormFormula,
        formulas: factor.compactFormulas.map((formula) => formula.replace(/\brms_factor\b/g, "final_rms_factor")) };
      const output = substituteFixedF16ParametricFormulas(
        await compileFixedF16OutputRowFormula(reader, tensor(head), outputDimension), compactNorm, null);
      const expanded = { ...output, formulas: output.formulas.map((formula) =>
        `(() => { const final_rms_factor=${factor.expression}; return ${formula}; })()`) };
      const inlined = substituteFixedF16ParametricFormulas(expanded, {
        kind: "fixed-f16-parametric-formulas", inputSize: source.inputSize,
        outputSize: source.outputSize, formulas: source.formulas,
        arithmetic: "f32-interleaved-four-lane-pairwise",
      }, null);
      let reduced = pruneFixedF16ScalarDeclarations({ ...source, outputSize: 1, formulas: inlined.formulas });
      while (true) {
        const next = inlineNextFixedF16ScalarCache(reduced);
        if (next.inlined === null) break;
        reduced = next.source;
      }
      logits = reduced;
    }
    if (inputBoundary === "embeddings") return { ...logits, maxSequenceLength };
    const embedding = await compileFixedF16CachedEmbeddingSource(reader, ir.prelude[0].weight.name);
    return { ...composeCachedScalarSources(logits, embedding), maxSequenceLength };
  } finally {
    await reader.close();
    await opened.close();
  }
}

/** Start backward substitution at one final output dimension. */
export async function compileFixedF16ScalarDimensionFromDirectory(
  directory: string, dimension: number,
): Promise<FixedF16CachedScalarSource> {
  return compileFixedF16ScalarModelFromDirectory(directory, "embeddings", dimension);
}

/** Persist only numeric source and literals; evaluation does not reopen the model directory. */
export async function writeFixedF16ScalarArtifact(
  directory: string, output: string, inputBoundary: FixedF16InputBoundary = "embeddings",
): Promise<void> {
  const source = await compileFixedF16ScalarModelFromDirectory(directory, inputBoundary);
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify({ schemaVersion: 1, numericProfile: "pytorch-cpu-eager-f16-four-lane",
    inputBoundary, source }), "utf8");
}

export async function readFixedF16ScalarArtifact(path: string): Promise<FixedF16CachedScalarSource> {
  const artifact: unknown = JSON.parse(await readFile(path, "utf8"));
  if (!artifact || typeof artifact !== "object" || !("schemaVersion" in artifact) ||
    artifact.schemaVersion !== 1 || !("numericProfile" in artifact) ||
    artifact.numericProfile !== "pytorch-cpu-eager-f16-four-lane" || !("source" in artifact)) {
    throw new Error("Artefato escalar F16 incompatível.");
  }
  if ("inputBoundary" in artifact && artifact.inputBoundary !== "embeddings" &&
    artifact.inputBoundary !== "token-ids") throw new Error("Fronteira de entrada escalar inválida.");
  const source = artifact.source;
  if (!source || typeof source !== "object" || !("kind" in source) ||
    source.kind !== "fixed-f16-cached-scalar-source" || !("formulas" in source) ||
    !Array.isArray(source.formulas) || !("declarations" in source) ||
    typeof source.declarations !== "string" || !("inputSize" in source) ||
    typeof source.inputSize !== "number" || !("outputSize" in source) ||
    typeof source.outputSize !== "number" || !("nextCacheId" in source) ||
    typeof source.nextCacheId !== "number") throw new Error("Fonte escalar F16 inválida.");
  if ("maxSequenceLength" in source && (!Number.isSafeInteger(source.maxSequenceLength) ||
    (source.maxSequenceLength as number) <= 0)) throw new Error("Limite de contexto inválido.");
  return source as FixedF16CachedScalarSource;
}
