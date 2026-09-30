import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm, access, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { SafetensorsCatalogReader } from "../src/safetensors.js";
import { compileFixedF16Projection, evaluateFixedF16Projection, f16BitsToDyadic, roundDyadicToF16IfElse, roundDyadicToF32IfElse } from "../src/fixed-f16-projection.js";
import { evaluateFixedScalarFunctions, scalarizeFixedF16Projection, substituteFixedScalarFunctions } from "../src/fixed-f16-scalar-functions.js";
import { evaluateFixedTwoTokenAttentionScores, evaluateFixedTwoTokenAttentionValues } from "../src/fixed-f16-attention-scores.js";
import { softmaxTwoF16IfElse } from "../src/fixed-f16-two-way-softmax.js";
import { addFixedF16Residual, compileFixedF16AttentionParametricFormulas, compileFixedF16CachedAttentionSource, compileFixedF16CachedMlpSource, compileFixedF16EmbeddingParametricFormulas, compileFixedF16MlpParametricFormulas, compileFixedF16ParametricFormulas, compileFixedF16RmsNormParametricFormulas, composeCachedScalarSource, composeCachedScalarSources, estimateFixedF16ParametricSubstitutionCharacters, evaluateFixedF16CachedScalarSource, evaluateFixedF16ParametricFormulas, prepareFixedF16CachedScalarDimensionSource, prepareFixedF16CachedScalarSource, scalarSourceFromFactoredRmsNorm, scalarSourceFromFormulas, substituteFactoredRmsNorm, substituteFixedF16ParametricFormulas } from "../src/fixed-f16-parametric-formulas.js";

test("MLP escalar com cache gate/up preserva valores F16 para n variável", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-linear.json"), "utf8")) as
    { mlps: Record<string, { inputs: number[][]; outputs: Record<string, number[][]> }> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const source = await compileFixedF16CachedMlpSource(reader, fixtureMlpTensors(layer), "f32-interleaved-four-lane-pairwise");
      for (const n of [1, 2, 3, 4]) {
        assert.deepEqual(evaluateFixedF16CachedScalarSource(source, fixture.mlps[String(layer)]!.inputs.slice(0, n)),
          fixture.mlps[String(layer)]!.outputs[String(n)], `layer=${layer}, n=${n}`);
      }
    }
  } finally { await reader.close(); }
});

test("atenção escalar com cache Q/K/V preserva valores F16 para n variável e escores extremos", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-attention.json"), "utf8")) as
    { cases: Record<string, { layers: Record<string, { input: number[][][]; output: number[][][] }> }>;
      stress: Record<string, Record<string, { input: number[][][]; output: number[][][] }>> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const source = await compileFixedF16CachedAttentionSource(reader, fixtureAttentionTensors(layer),
        4, 4, 4, 10_000, "f32-interleaved-four-lane-pairwise");
      for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
        const { input, output } = fixture.cases[String(n)]!.layers[String(layer)]!;
        assert.deepEqual(evaluateFixedF16CachedScalarSource(source, input[0]!), output[0], `layer=${layer}, n=${n}`);
      }
      for (const scale of [10, 100, 1000]) {
        const { input, output } = fixture.stress[String(scale)]![String(layer)]!;
        assert.deepEqual(evaluateFixedF16CachedScalarSource(source, input[0]!), output[0], `layer=${layer}, scale=${scale}`);
      }
    }
  } finally { await reader.close(); }
});
import { compileFixedF16ScalarModelFromDirectory, readFixedF16ScalarArtifact, writeFixedF16ScalarArtifact } from "../src/fixed-f16-ir-scalar-compiler.js";

test("artefato escalar salvo executa logits sem consultar o checkpoint", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-attention.json"), "utf8")) as
    { token_cases: Record<string, { ids: number[]; logits: number[][][] }>;
      random_token_cases: Record<string, { ids: number[]; logits: number[][][] }> };
  const temporary = await mkdtemp(join(tmpdir(), "scalar-artifact-"));
  try {
    const artifact = join(temporary, "model.json");
    await writeFixedF16ScalarArtifact(directory, artifact);
    const source = await readFixedF16ScalarArtifact(artifact);
    assert.equal(source.inputSize, 16);
    const reader = new SafetensorsCatalogReader(directory);
    const embedding = await compileFixedF16EmbeddingParametricFormulas(reader, "model.embed_tokens.weight");
    await reader.close();
    const selected = prepareFixedF16CachedScalarDimensionSource(source, 17);
    for (const { ids, logits } of [
      fixture.token_cases["1"]!, fixture.token_cases["4"]!, fixture.token_cases["8"]!,
      ...Object.values(fixture.random_token_cases),
    ]) {
      const vectors = ids.map((id) => evaluateFixedF16CachedScalarSource(scalarSourceFromFormulas(embedding), [[id]])[0]!);
      assert.deepEqual(evaluateFixedF16CachedScalarSource(source, vectors), logits[0], `n=${ids.length}`);
      assert.deepEqual(selected(vectors), logits[0]!.map((row) => row[17]!), `dim=17, n=${ids.length}`);
    }
  } finally { await rm(temporary, { recursive: true, force: true }); }
});

test("descobre o forward do pacote e gera logits escalares sem nomes de tensores fornecidos pelo teste", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-attention.json"), "utf8")) as
    { token_cases: Record<string, { ids: number[]; logits: number[][][] }>;
      random_token_cases: Record<string, { ids: number[]; logits: number[][][] }> };
  const source = await compileFixedF16ScalarModelFromDirectory(directory, "token-ids");
  const metrics = { calls: {} as Record<string, number> };
  const run = prepareFixedF16CachedScalarSource(source, metrics);
  for (const n of [1, 4, 8]) {
    const { ids, logits } = fixture.token_cases[String(n)]!;
    assert.deepEqual(run(ids.map((id) => [id])), logits[0], `n=${n}`);
  }
  for (const [seed, { ids, logits }] of Object.entries(fixture.random_token_cases)) {
    assert.deepEqual(run(ids.map((id) => [id])), logits[0], `seed=${seed}, n=${ids.length}`);
  }
  assert.ok((metrics.calls["Math.fround"] ?? 0) > 0);
  assert.ok((metrics.calls.f16Bits ?? 0) > 0);
  assert.ok((metrics.calls["Math.exp"] ?? 0) > 0);
});

const fixtureAttentionTensors = (index: number) => {
  const base = `model.layers.${index}.self_attn.`;
  return { q: `${base}q_proj.weight`, k: `${base}k_proj.weight`,
    v: `${base}v_proj.weight`, o: `${base}o_proj.weight` };
};
const fixtureMlpTensors = (index: number) => {
  const base = `model.layers.${index}.mlp.`;
  return { gate: `${base}gate_proj.weight`, up: `${base}up_proj.weight`, down: `${base}down_proj.weight` };
};

test("duas camadas completas preservam valores finais por dimensão com cache escalar", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-attention.json"), "utf8")) as
    { cases: Record<string, { layers: Record<string, { layer_input: number[][][]; layer_output: number[][][] }>;
      logits: number[][][] }>;
      token_cases: Record<string, { ids: number[]; logits: number[][][] }> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    let source;
    for (const index of [0, 1]) {
      const inputNorm = await compileFixedF16RmsNormParametricFormulas(reader, `model.layers.${index}.input_layernorm.weight`, 1e-6);
      const attention = await compileFixedF16AttentionParametricFormulas(reader, fixtureAttentionTensors(index), 4, 4, 4, 10_000, "f32-interleaved-four-lane-pairwise");
      const firstResidual = addFixedF16Residual(substituteFactoredRmsNorm(attention, inputNorm));
      const postNorm = await compileFixedF16RmsNormParametricFormulas(reader, `model.layers.${index}.post_attention_layernorm.weight`, 1e-6);
      const mlp = await compileFixedF16MlpParametricFormulas(reader, fixtureMlpTensors(index), "f32-interleaved-four-lane-pairwise");
      const postMlp = substituteFactoredRmsNorm(mlp, postNorm);
      const secondResidual = { ...postMlp, formulas: postMlp.formulas.map((formula, d) => `add16(x[t][${d}],${formula})`) };
      const local = composeCachedScalarSource(secondResidual, scalarSourceFromFormulas(firstResidual));
      source = source ? composeCachedScalarSources(local, source) : local;
      for (const n of [1, 2, 3, 4]) {
        const { layer_input } = fixture.cases[String(n)]!.layers["0"]!;
        const { layer_output } = fixture.cases[String(n)]!.layers[String(index)]!;
        assert.deepEqual(evaluateFixedF16CachedScalarSource(source, layer_input[0]!), layer_output[0], `camada ${index}, n=${n}`);
      }
    }
    const finalNorm = await compileFixedF16RmsNormParametricFormulas(reader, "model.norm.weight", 1e-6);
    const normalized = composeCachedScalarSource(finalNorm, source!);
    const head = await compileFixedF16ParametricFormulas(reader, "lm_head.weight", "f32-interleaved-four-lane-pairwise");
    const logits = composeCachedScalarSource(head, normalized);
    const first = fixture.cases["1"]!;
    assert.deepEqual(evaluateFixedF16CachedScalarSource(logits, first.layers["0"]!.layer_input[0]!),
      first.logits[0], "todos os logits finais, n=1");
    const eight = fixture.cases["8"]!;
    assert.deepEqual(evaluateFixedF16CachedScalarSource(logits, eight.layers["0"]!.layer_input[0]!),
      eight.logits[0], "todos os logits finais, n=8");
    const embedding = await compileFixedF16EmbeddingParametricFormulas(reader, "model.embed_tokens.weight");
    const fromIds = composeCachedScalarSources(logits, scalarSourceFromFormulas(embedding));
    for (const n of [1, 4]) {
      const { ids, logits: expected } = fixture.token_cases[String(n)]!;
      assert.deepEqual(evaluateFixedF16CachedScalarSource(fromIds, ids.map((id) => [id])),
        expected[0], `IDs até todos os logits, n=${n}`);
    }
  } finally { await reader.close(); }
});

test("atenção escalar paramétrica compara dimensões finais com PyTorch para n=1..8 e escores extremos", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-attention.json"), "utf8")) as
    { cases: Record<string, { layers: Record<string, { layer_input: number[][][]; post_norm_input: number[][][]; input: number[][][]; output: number[][][] }> }>;
      stress: Record<string, Record<string, { input: number[][][]; output: number[][][] }>> };
  const reader = new SafetensorsCatalogReader(directory);
  try { for (const layer of [0, 1]) {
    const program = await compileFixedF16AttentionParametricFormulas(reader, fixtureAttentionTensors(layer), 4, 4, 4, 10_000, "f32-interleaved-four-lane-pairwise");
    for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const { input, output } = fixture.cases[String(n)]!.layers[String(layer)]!;
      assert.deepEqual(evaluateFixedF16ParametricFormulas(program, input[0]!), output[0], `camada=${layer}, n=${n}`);
    }
    for (const scale of [10, 100, 1000]) {
      const { input, output } = fixture.stress[String(scale)]![String(layer)]!;
      assert.deepEqual(evaluateFixedF16ParametricFormulas(program, input[0]!), output[0], `camada=${layer}, escala=${scale}`);
    }
    const norm = await compileFixedF16RmsNormParametricFormulas(reader, `model.layers.${layer}.input_layernorm.weight`, 1e-6);
    const fused = substituteFactoredRmsNorm(program, norm);
    const residual = addFixedF16Residual(fused);
    for (const n of [1, 2, 3, 4]) {
      const { layer_input, post_norm_input, output } = fixture.cases[String(n)]!.layers[String(layer)]!;
      assert.deepEqual(evaluateFixedF16ParametricFormulas(fused, layer_input[0]!), output[0], `norm+atenção camada=${layer}, n=${n}`);
      assert.deepEqual(evaluateFixedF16ParametricFormulas(residual, layer_input[0]!), post_norm_input[0], `residual camada=${layer}, n=${n}`);
    }
  } } finally { await reader.close(); }
});

test("if/else F16 respeita limites, empate par, sinal e overflow", () => {
  assert.equal(roundDyadicToF16IfElse({ coefficient: 2049n, exponent: -11 }), 0x3c00); // midpoint 1, 1+2^-10
  assert.equal(roundDyadicToF16IfElse({ coefficient: 2051n, exponent: -11 }), 0x3c02); // midpoint: lower ímpar
  assert.equal(roundDyadicToF16IfElse({ coefficient: -2049n, exponent: -11 }), 0xbc00);
  assert.equal(roundDyadicToF16IfElse({ coefficient: 1n, exponent: -25 }), 0); // metade do menor subnormal
  assert.equal(roundDyadicToF16IfElse({ coefficient: -3n, exponent: -25 }), 0x8002);
  assert.equal(roundDyadicToF16IfElse({ coefficient: 65520n, exponent: 0 }), 0x7c00);
  assert.equal(f16BitsToDyadic(0x3c00).coefficient, 1024n);
  assert.equal(roundDyadicToF32IfElse({ coefficient: 0x1000001n, exponent: -24 }), 0x3f800000); // empate F32: 1 é par
  assert.equal(roundDyadicToF32IfElse({ coefficient: 0x1000003n, exponent: -24 }), 0x3f800002);
  assert.equal(roundDyadicToF32IfElse({ coefficient: 1n, exponent: -150 }), 0);
});

test("substitui pesos de cada dimensão e compõe duas projeções com arredondamento entre elas", async () => {
  const directory = await mkdtemp(join(tmpdir(), "fixed-f16-"));
  const data = Buffer.alloc(16);
  [0x4000, 0x0000, 0x0000, 0x3800, 0x3800, 0x0000, 0x0000, 0x3c00].forEach((bits, index) => data.writeUInt16LE(bits, index * 2));
  const header = Buffer.from(JSON.stringify({ "layer.0.weight": { dtype: "F16", shape: [2, 2], data_offsets: [0, 8] }, "layer.1.weight": { dtype: "F16", shape: [2, 2], data_offsets: [8, 16] } }));
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(header.length));
  await writeFile(join(directory, "model.safetensors"), Buffer.concat([length, header, data]));
  await writeFile(join(directory, "config.json"), JSON.stringify({ model_type: "fixture" }));
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const first = await compileFixedF16Projection(reader, "layer.0.weight");
    const second = await compileFixedF16Projection(reader, "layer.1.weight");
    assert.deepEqual(first.rows.map((row) => row.terms.map((term) => term.weightBits)), [[0x4000], [0x3800]]);
    const intermediate = evaluateFixedF16Projection(first, [0x3c00, 0x4000]);
    assert.deepEqual(intermediate, [0x4000, 0x3c00]);
    assert.deepEqual(evaluateFixedF16Projection(second, intermediate), [0x3c00, 0x3c00]);
  } finally { await reader.close(); await rm(directory, { recursive: true, force: true }); }
});

test("q_proj real: 16 dimensões coincidem com a referência PyTorch F16 no vetor fixo", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const program = await compileFixedF16Projection(reader, "model.layers.0.self_attn.q_proj.weight");
    const input = [11264, 12288, 12800, 13312, 13568, 13824, 14080, 14336, 14464, 14592, 14720, 14848, 14976, 15104, 15232, 15360];
    const expected = [0x1feb, 0x1d71, 0x2c2a, 0x279b, 0x9eb6, 0x2b30, 0x292e, 0xa8d3, 0x241d, 0x2c7f, 0xa2db, 0xa0db, 0xa5eb, 0x2a03, 0x29af, 0xa61d];
    assert.equal(program.inputSize, 16); assert.equal(program.outputSize, 16);
    assert.deepEqual(evaluateFixedF16Projection(program, input), expected);
  } finally { await reader.close(); }
});

test("Q/K/V reais das duas camadas: 192 resultados do forward de dois tokens coincidem bit a bit", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const reference = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-projections.json"), "utf8")) as
    Record<string, { input: number[][][]; output: number[][][] }>;
  const reader = new SafetensorsCatalogReader(directory);
  let compared = 0;
  try {
    for (const [tensor, values] of Object.entries(reference)) {
      const program = await compileFixedF16Projection(reader, tensor);
      for (let token = 0; token < 2; token++) {
        const actual = evaluateFixedF16Projection(program, values.input[0]![token]!);
        const expected = values.output[0]![token]!;
        assert.deepEqual(actual, expected, `${tensor}, token ${token}`);
        compared += expected.length;
      }
    }
    assert.equal(compared, 192);
  } finally { await reader.close(); }
});

test("fórmulas escalares com pesos substituídos aceitam n tokens e compõem projeções sem estágios", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const reference = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-projections.json"), "utf8")) as
    Record<string, { input: number[][][]; output: number[][][] }>;
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const qName = "model.layers.0.self_attn.q_proj.weight";
    const kName = "model.layers.0.self_attn.k_proj.weight";
    const q = await compileFixedF16ParametricFormulas(reader, qName);
    const k = await compileFixedF16ParametricFormulas(reader, kName);
    const kProjection = await compileFixedF16Projection(reader, kName);
    const composed = substituteFixedF16ParametricFormulas(k, q);
    assert.equal(estimateFixedF16ParametricSubstitutionCharacters(k, q), BigInt(composed.formulas.join("").length));
    const causalConsumer = { kind: "fixed-f16-parametric-formulas" as const, inputSize: 16, outputSize: 1,
      formulas: ["f16Bits(causalSum(t, (j) => f16(x[j][0])))"], arithmetic: "f32-ascending-products-and-sum" as const };
    const causalClosed = substituteFixedF16ParametricFormulas(causalConsumer, q);
    const rows = [...reference[qName]!.input[0]!, reference[qName]!.input[0]![0]!];
    assert.deepEqual(evaluateFixedF16ParametricFormulas(q, rows.slice(0, 2)), reference[qName]!.output[0]);
    for (const length of [0, 1, 2, 3]) {
      const input = rows.slice(0, length);
      const projected = evaluateFixedF16ParametricFormulas(q, input);
      const expected = projected.map((row) => evaluateFixedF16Projection(kProjection, row));
      assert.deepEqual(evaluateFixedF16ParametricFormulas(composed, input), expected, `n=${length}`);
      assert.deepEqual(evaluateFixedF16ParametricFormulas(causalClosed, input),
        evaluateFixedF16ParametricFormulas(causalConsumer, projected), `n=${length}, redução causal sem estágio`);
      assert.doesNotMatch(composed.formulas.join("\n"), /weightBits|q_proj|k_proj|function|=>/);
    }
  } finally { await reader.close(); }
});

test("fórmulas paramétricas de quatro acumuladores reproduzem PyTorch para n=1..4", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-linear.json"), "utf8")) as
    { cases: Record<string, { inputs: number[][]; outputs: Record<string, number[][]> }> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const [name, values] of Object.entries(fixture.cases).filter(([name]) => name.includes("_proj"))) {
      const program = await compileFixedF16ParametricFormulas(reader, `${name}.weight`, "f32-interleaved-four-lane-pairwise");
      for (let length = 1; length <= 4; length++) {
        assert.deepEqual(evaluateFixedF16ParametricFormulas(program, values.inputs.slice(0, length)),
          values.outputs[String(length)], `${name}, n=${length}`);
      }
      assert.doesNotMatch(program.formulas.join("\n"), /weightBits|\.weight|layer\./);
    }
  } finally { await reader.close(); }
});

test("RMSNorm paramétrica F16 reproduz valores PyTorch por dimensão para n=1..4", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-linear.json"), "utf8")) as
    { cases: Record<string, { inputs: number[][]; outputs: Record<string, number[][]> }> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const [name, values] of Object.entries(fixture.cases).filter(([name]) => name.includes("layernorm"))) {
      const program = await compileFixedF16RmsNormParametricFormulas(reader, `${name}.weight`, 1e-6);
      for (let length = 1; length <= 4; length++) {
        assert.deepEqual(evaluateFixedF16ParametricFormulas(program, values.inputs.slice(0, length)),
          values.outputs[String(length)], `${name}, n=${length}`);
      }
    }
  } finally { await reader.close(); }
});

test("substituição literal de RMSNorm em Q preserva o resultado PyTorch para n variável", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-linear.json"), "utf8")) as
    { composites: { layer0_norm_q: { inputs: number[][]; outputs: Record<string, number[][]> } } };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const norm = await compileFixedF16RmsNormParametricFormulas(reader, "model.layers.0.input_layernorm.weight", 1e-6);
    const q = await compileFixedF16ParametricFormulas(reader, "model.layers.0.self_attn.q_proj.weight", "f32-interleaved-four-lane-pairwise");
    const composed = substituteFixedF16ParametricFormulas(q, norm);
    assert.ok(composed.formulas.join("").length > q.formulas.join("").length);
    for (let length = 1; length <= 4; length++) {
      assert.deepEqual(evaluateFixedF16ParametricFormulas(composed, fixture.composites.layer0_norm_q.inputs.slice(0, length)),
        fixture.composites.layer0_norm_q.outputs[String(length)], `n=${length}`);
    }
  } finally { await reader.close(); }
});

test("MLP completa é uma fórmula plana por dimensão para n=1..4", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-parametric-linear.json"), "utf8")) as
    { mlps: Record<string, { inputs: number[][]; outputs: Record<string, number[][]> }> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const program = await compileFixedF16MlpParametricFormulas(reader, fixtureMlpTensors(layer), "f32-interleaved-four-lane-pairwise");
      assert.equal(program.outputSize, 16);
      assert.doesNotMatch(program.formulas.join("\n"), /gate_proj|up_proj|down_proj|weightBits/);
      for (let length = 1; length <= 4; length++) {
        assert.deepEqual(evaluateFixedF16ParametricFormulas(program, fixture.mlps[String(layer)]!.inputs.slice(0, length)),
          fixture.mlps[String(layer)]!.outputs[String(length)], `camada ${layer}, n=${length}`);
      }
    }
    const norm = await compileFixedF16RmsNormParametricFormulas(reader, "model.layers.0.post_attention_layernorm.weight", 1e-6);
    const mlp = await compileFixedF16MlpParametricFormulas(reader, fixtureMlpTensors(0), "f32-interleaved-four-lane-pairwise");
    const required = estimateFixedF16ParametricSubstitutionCharacters(mlp, norm);
    assert.ok(required > 10_000_000n);
    assert.throws(() => substituteFixedF16ParametricFormulas(mlp, norm, 10_000_000), /Nenhuma fórmula parcial/);
    const factored = substituteFactoredRmsNorm(mlp, norm, 10_000_000);
    assert.ok(BigInt(factored.formulas.join("").length) < required / 5n);
    const fixtureWithComposite = fixture as typeof fixture &
      { composites: { layer0_post_norm_mlp: { inputs: number[][]; outputs: Record<string, number[][]> } } };
    for (let length = 1; length <= 4; length++) {
      assert.deepEqual(evaluateFixedF16ParametricFormulas(factored,
        fixtureWithComposite.composites.layer0_post_norm_mlp.inputs.slice(0, length)),
      fixtureWithComposite.composites.layer0_post_norm_mlp.outputs[String(length)], `RMSNorm→MLP n=${length}`);
    }
  } finally { await reader.close(); }
});

test("atenção de um token: compõe V e O nas duas camadas com paridade de bits", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const reference = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-single-token-attention.json"), "utf8")) as
    Record<string, { input: number[][][]; output: number[][][] }>;
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const valueKey = `model.layers.${layer}.self_attn.v_proj.weight`;
      const outputKey = `model.layers.${layer}.self_attn.o_proj.weight`;
      const valueProgram = await compileFixedF16Projection(reader, valueKey);
      const outputProgram = await compileFixedF16Projection(reader, outputKey);
      const values = evaluateFixedF16Projection(valueProgram, reference[valueKey]!.input[0]![0]!);
      assert.deepEqual(values, reference[valueKey]!.output[0]![0]!, `V camada ${layer}`);
      assert.deepEqual(values, reference[outputKey]!.input[0]![0]!, `entrada O camada ${layer}`);
      assert.deepEqual(evaluateFixedF16Projection(outputProgram, values), reference[outputKey]!.output[0]![0]!, `O camada ${layer}`);
      const closed = substituteFixedScalarFunctions(scalarizeFixedF16Projection(outputProgram), scalarizeFixedF16Projection(valueProgram));
      assert.equal(closed.outputs.length, 16);
      assert.equal(closed.inputSize, 16);
      assert.deepEqual(evaluateFixedScalarFunctions(closed, reference[valueKey]!.input[0]![0]!), reference[outputKey]!.output[0]![0]!, `funções compostas camada ${layer}`);
      assert.doesNotMatch(JSON.stringify(closed), /self_attn|v_proj|o_proj|weightBits/);
    }
  } finally { await reader.close(); }
});

test("RoPE e scores mascarados de dois tokens coincidem por head e camada", async () => {
  const projections = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-projections.json"), "utf8")) as
    Record<string, { output: number[][][] }>;
  const stages = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-stages.json"), "utf8")) as
    Record<string, { cos: number[][][]; sin: number[][][]; q_rotated: number[][][][]; k_rotated: number[][][][];
      score: number[][][][]; scaled: number[][][][]; masked: number[][][][]; probabilities: number[][][][] }>;
  for (const layer of [0, 1]) {
    const base = `model.layers.${layer}.self_attn.`;
    const fixture = stages[String(layer)]!;
    const actual = evaluateFixedTwoTokenAttentionScores(
      projections[`${base}q_proj.weight`]!.output[0]!, projections[`${base}k_proj.weight`]!.output[0]!,
      fixture.cos[0]!, fixture.sin[0]!,
    );
    assert.deepEqual(actual.qRotated, fixture.q_rotated[0], `RoPE Q camada ${layer}`);
    assert.deepEqual(actual.kRotated, fixture.k_rotated[0], `RoPE K camada ${layer}`);
    assert.deepEqual(actual.score, fixture.score[0], `QK camada ${layer}`);
    assert.deepEqual(actual.scaled, fixture.scaled[0], `escala camada ${layer}`);
    assert.deepEqual(actual.masked, fixture.masked[0], `máscara camada ${layer}`);
    const probabilities = actual.masked.map((head) => head.map((row) => softmaxTwoF16IfElse(row[0]!, row[1]!)));
    assert.deepEqual(probabilities, fixture.probabilities[0], `softmax camada ${layer}`);
  }
});

test("atenção completa de dois tokens atravessa AV e O com paridade de bits", async (context) => {
  const projections = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-projections.json"), "utf8")) as
    Record<string, { input: number[][][]; output: number[][][] }>;
  const stages = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-stages.json"), "utf8")) as
    Record<string, { cos: number[][][]; sin: number[][][]; probabilities: number[][][][] }>;
  const outputs = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-output.json"), "utf8")) as
    Record<string, { input: number[][][]; output: number[][][] }>;
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const base = `model.layers.${layer}.self_attn.`;
      const projected: Record<string, number[][]> = {};
      for (const name of ["q_proj", "k_proj", "v_proj"]) {
        const tensor = `${base}${name}.weight`;
        const program = await compileFixedF16Projection(reader, tensor);
        projected[name] = projections[tensor]!.input[0]!.map((input) => evaluateFixedF16Projection(program, input));
        assert.deepEqual(projected[name], projections[tensor]!.output[0], `${name} camada ${layer}`);
      }
      const fixture = stages[String(layer)]!;
      const scores = evaluateFixedTwoTokenAttentionScores(projected.q_proj!, projected.k_proj!, fixture.cos[0]!, fixture.sin[0]!);
      const probabilities = scores.masked.map((head) => head.map((row) => softmaxTwoF16IfElse(row[0]!, row[1]!)));
      assert.deepEqual(probabilities, fixture.probabilities[0], `softmax camada ${layer}`);
      const value = projected.v_proj!;
      const attention = evaluateFixedTwoTokenAttentionValues(probabilities, value);
      assert.deepEqual(attention, outputs[String(layer)]!.input[0], `AV camada ${layer}`);
      const projection = await compileFixedF16Projection(reader, `model.layers.${layer}.self_attn.o_proj.weight`);
      for (let token = 0; token < 2; token++) {
        assert.deepEqual(evaluateFixedF16Projection(projection, attention[token]!), outputs[String(layer)]!.output[0]![token]!, `O camada ${layer}, token ${token}`);
      }
    }
  } finally { await reader.close(); }
});
