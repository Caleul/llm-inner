import { buildModelIR } from "./architecture.js";
import { openCatalog } from "./catalog.js";
import { SafetensorsCatalogReader } from "./safetensors.js";
import type { Operation } from "./types.js";
import {
  addFixedF16Residual, compileFixedF16AttentionParametricFormulas,
  compileFixedF16EmbeddingParametricFormulas, compileFixedF16MlpParametricFormulas,
  compileFixedF16ParametricFormulas, compileFixedF16RmsNormParametricFormulas,
  composeCachedScalarSource, composeCachedScalarSources, scalarSourceFromFormulas,
  substituteFactoredRmsNorm, type FixedF16CachedScalarSource,
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

/** Lower a supported ordered decoder forward into a checkpoint-independent scalar function. */
export async function compileFixedF16ScalarModelFromDirectory(directory: string): Promise<FixedF16CachedScalarSource> {
  const opened = await openCatalog(directory, false);
  const reader = new SafetensorsCatalogReader(directory);
  try {
    requireForward(opened.catalog.format === "safetensors", "contêiner Safetensors necessário");
    const ir = await buildModelIR(opened.catalog, { outputRows: 0, inputTerms: 0, includeWeights: false });
    requireForward(ir.prelude.length === 1 && ir.prelude[0]?.op === "embedding" &&
      ir.prelude[0].scale === undefined && ir.prelude[0].weight.storageDtype === "F16",
    "embedding F16 sem escala necessário");
    const embedding = await compileFixedF16EmbeddingParametricFormulas(reader, ir.prelude[0].weight.name);
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
      const compiledAttention = await compileFixedF16AttentionParametricFormulas(reader,
        { q: tensor(q), k: tensor(k), v: tensor(v), o: tensor(o) }, attention.numAttentionHeads,
        attention.numKeyValueHeads, attention.headDim, qRope.theta, "f32-interleaved-four-lane-pairwise");
      const firstResidual = addFixedF16Residual(substituteFactoredRmsNorm(compiledAttention, compiledNorm));
      const compiledPostNorm = await compileFixedF16RmsNormParametricFormulas(reader, tensor(postNorm), postNorm.epsilon);
      const compiledMlp = await compileFixedF16MlpParametricFormulas(reader,
        { gate: tensor(gate), up: tensor(up), down: tensor(down) }, "f32-interleaved-four-lane-pairwise");
      const postMlp = substituteFactoredRmsNorm(compiledMlp, compiledPostNorm);
      const secondResidual = { ...postMlp, formulas: postMlp.formulas.map((formula, dimension) =>
        `add16(x[t][${dimension}],${formula})`) };
      const local = composeCachedScalarSource(secondResidual, scalarSourceFromFormulas(firstResidual));
      source = source ? composeCachedScalarSources(local, source) : local;
    }
    requireForward(source && ir.epilogue.length === 2, "normalização final e projeção de logits necessárias");
    const finalNorm = norm(ir.epilogue[0]!);
    const head = linear(ir.epilogue[1]!);
    requireForward(head.input === finalNorm.output, "projeção final não recebe a RMSNorm final");
    const normalized = composeCachedScalarSource(await compileFixedF16RmsNormParametricFormulas(
      reader, tensor(finalNorm), finalNorm.epsilon), source);
    const logits = composeCachedScalarSource(await compileFixedF16ParametricFormulas(
      reader, tensor(head), "f32-interleaved-four-lane-pairwise"), normalized);
    return composeCachedScalarSources(logits, scalarSourceFromFormulas(embedding));
  } finally {
    await reader.close();
    await opened.close();
  }
}
