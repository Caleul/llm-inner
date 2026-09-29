import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, writeFile, rm, access, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { SafetensorsCatalogReader } from "../src/safetensors.js";
import { compileFixedF16Projection, evaluateFixedF16Projection, f16BitsToDyadic, roundDyadicToF16IfElse, roundDyadicToF32IfElse } from "../src/fixed-f16-projection.js";
import { evaluateFixedScalarFunctions, scalarizeFixedF16Projection, substituteFixedScalarFunctions } from "../src/fixed-f16-scalar-functions.js";
import { evaluateFixedTwoTokenAttentionScores, evaluateFixedTwoTokenAttentionValues } from "../src/fixed-f16-attention-scores.js";
import { softmaxTwoF16IfElse } from "../src/fixed-f16-two-way-softmax.js";
import { compileFixedTwoTokenAttention, evaluateFixedTwoTokenAttention } from "../src/fixed-f16-attention-program.js";
import { addF16Bits, compileFixedMlp, evaluateFixedFourLaneProjection, evaluateFixedMlp, readFixedF16Vector, rmsNormF16 } from "../src/fixed-f16-layer-ops.js";
import { compileFixedTwoTokenModel, evaluateFixedTwoTokenModel } from "../src/fixed-f16-two-token-model.js";

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

test("softmax de dois logits cobre diferenças amplas e compara 212 pares com PyTorch CPU", async () => {
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-softmax-pytorch-cpu.json"), "utf8")) as
    { torch: string; cases: Array<{ score: [number, number]; probability: [number, number] }> };
  assert.equal(fixture.cases.length, 212);
  for (const [index, sample] of fixture.cases.entries()) {
    assert.deepEqual(softmaxTwoF16IfElse(...sample.score), sample.probability, `caso ${index}, scores ${sample.score}`);
  }
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

test("cada saída O de dois tokens depende somente da entrada da camada e de pesos literais", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const projections = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-projections.json"), "utf8")) as
    Record<string, { input: number[][][] }>;
  const stages = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-stages.json"), "utf8")) as
    Record<string, { cos: number[][][]; sin: number[][][] }>;
  const outputs = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-output.json"), "utf8")) as
    Record<string, { output: number[][][] }>;
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const program = await compileFixedTwoTokenAttention(reader, layer, stages[String(layer)]!.cos[0]!, stages[String(layer)]!.sin[0]!);
      const input = projections[`model.layers.${layer}.self_attn.q_proj.weight`]!.input[0]!.flat();
      assert.equal(program.outputs.length, 32);
      assert.ok(program.nodes.every((node) => node.op !== "input" || node.index < 32));
      assert.deepEqual(evaluateFixedTwoTokenAttention(program, input), outputs[String(layer)]!.output[0]!.flat(), `O fechado camada ${layer}`);
      assert.doesNotMatch(JSON.stringify(program), /q_proj|k_proj|v_proj|o_proj|weightBits/);
    }
  } finally { await reader.close(); }
});

test("RMSNorm das duas camadas coincide com o forward capturado", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-full-forward.json"), "utf8")) as
    { cases: Record<string, Record<string, { input: number[]; output: number[] }>> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const [caseName, records] of Object.entries(fixture.cases)) for (const layer of [0, 1]) {
      for (const name of ["input_layernorm", "post_attention_layernorm"]) {
        const key = `model.layers.${layer}.${name}`;
        const weight = await readFixedF16Vector(reader, `${key}.weight`, 16);
        const record = records[key]!;
        for (let token = 0; token < 2; token++) {
          assert.deepEqual(rmsNormF16(record.input.slice(token * 16, token * 16 + 16), weight),
            record.output.slice(token * 16, token * 16 + 16), `${caseName}, ${key}, token ${token}`);
        }
      }
    }
  } finally { await reader.close(); }
});

test("projeções gate, up e down da MLP coincidem com o forward capturado", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-full-forward.json"), "utf8")) as
    { cases: Record<string, Record<string, { input: number[]; output: number[] }>> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) for (const name of ["gate_proj", "up_proj", "down_proj"]) {
      const key = `model.layers.${layer}.mlp.${name}`;
      const program = await compileFixedF16Projection(reader, `${key}.weight`);
      for (const [caseName, records] of Object.entries(fixture.cases)) {
        const record = records[key]!;
        for (let token = 0; token < 2; token++) {
          assert.deepEqual(evaluateFixedFourLaneProjection(program, record.input.slice(token * program.inputSize, (token + 1) * program.inputSize)),
            record.output.slice(token * program.outputSize, (token + 1) * program.outputSize), `${caseName}, ${key}, token ${token}`);
        }
      }
    }
  } finally { await reader.close(); }
});

test("MLP completa coincide com o forward capturado", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-full-forward.json"), "utf8")) as
    { cases: Record<string, Record<string, { input: number[]; output: number[] }>> };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    for (const layer of [0, 1]) {
      const program = await compileFixedMlp(reader, layer);
      const key = `model.layers.${layer}.mlp`;
      for (const [caseName, records] of Object.entries(fixture.cases)) {
        const record = records[key]!;
        for (let token = 0; token < 2; token++) {
          assert.deepEqual(evaluateFixedMlp(program, record.input.slice(token * 16, token * 16 + 16)),
            record.output.slice(token * 16, token * 16 + 16), `${caseName}, ${key}, token ${token}`);
        }
      }
    }
  } finally { await reader.close(); }
});

test("embedding, duas camadas e logits são calculados somente dos IDs de token", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-full-forward.json"), "utf8")) as
    { cases: Record<string, Record<string, { input: number[]; output: number[] }> & { hidden_states: number[][]; argmax: number[] }> };
  const stages = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-stages.json"), "utf8")) as
    Record<string, { cos: number[][][]; sin: number[][][] }>;
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const model = await compileFixedTwoTokenModel(reader, stages["0"]!.cos[0]!, stages["0"]!.sin[0]!);
    const hydrated = JSON.parse(JSON.stringify(model)) as typeof model;
    for (const [name, records] of Object.entries(fixture.cases)) {
      const ids = name.split(",").map(Number);
      const result = evaluateFixedTwoTokenModel(hydrated, ids);
      assert.deepEqual(result.tokens, evaluateFixedTwoTokenModel(model, ids).tokens, `${name}, JSON recarregado`);
      assert.deepEqual(result.layers[0], records.hidden_states[1], `${name}, camada 0`);
      const layer1 = hydrated.layers[1];
      const normalized1 = [0, 1].flatMap((token) => rmsNormF16(result.layers[0]!.slice(token * 16, token * 16 + 16), layer1.inputNorm));
      assert.deepEqual(normalized1, records["model.layers.1.input_layernorm"]!.output, `${name}, norm de entrada camada 1`);
      for (const projectionName of ["q_proj", "k_proj", "v_proj"]) {
        const key = `model.layers.1.self_attn.${projectionName}`;
        const projection = await compileFixedF16Projection(reader, `${key}.weight`);
        const actual = [0, 1].flatMap((token) => evaluateFixedFourLaneProjection(projection, normalized1.slice(token * 16, token * 16 + 16)));
        assert.deepEqual(actual, records[key]!.output, `${name}, ${projectionName} camada 1`);
      }
      const attention1 = evaluateFixedTwoTokenAttention(layer1.attention, normalized1);
      assert.deepEqual(attention1, records["model.layers.1.self_attn"]!.output, `${name}, atenção camada 1`);
      const residual1 = result.layers[0]!.map((value, index) => addF16Bits(value, attention1[index]!));
      assert.deepEqual(residual1, records["model.layers.1.post_attention_layernorm"]!.input, `${name}, residual atenção camada 1`);
      const mlpInput1 = [0, 1].flatMap((token) => rmsNormF16(residual1.slice(token * 16, token * 16 + 16), layer1.postAttentionNorm));
      assert.deepEqual(mlpInput1, records["model.layers.1.post_attention_layernorm"]!.output, `${name}, norm MLP camada 1`);
      const mlp1 = [0, 1].flatMap((token) => evaluateFixedMlp(layer1.mlp, mlpInput1.slice(token * 16, token * 16 + 16)));
      assert.deepEqual(mlp1, records["model.layers.1.mlp"]!.output, `${name}, MLP camada 1`);
      const finalResidual = records["model.layers.1.post_attention_layernorm"]!.input;
      const finalMlp = records["model.layers.1.mlp"]!.output;
      assert.deepEqual(result.layers[1], finalResidual.map((value, index) => addF16Bits(value, finalMlp[index]!)), `${name}, camada 1`);
      assert.deepEqual(result.hidden, records["model.norm"]!.output, `${name}, norm final`);
      assert.deepEqual(result.logits, records.lm_head!.output, `${name}, logits`);
      assert.deepEqual(result.tokens, records.argmax, `${name}, argmax`);
    }
  } finally { await reader.close(); }
});

test("32 pares do vocabulário inteiro preservam hashes de camada e logits PyTorch", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const fixture = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-32-prompt-hashes.json"), "utf8")) as
    { cases: Array<{ ids: number[]; layer0_sha256: string; final_hidden_sha256: string; logits_sha256: string; next_token: number }> };
  const stages = JSON.parse(await readFile(resolve("test/fixtures/tiny-random-llama-two-token-attention-stages.json"), "utf8")) as
    Record<string, { cos: number[][][]; sin: number[][][] }>;
  const hash = (values: readonly number[]) => {
    const bytes = Buffer.allocUnsafe(values.length * 2);
    values.forEach((value, index) => bytes.writeUInt16LE(value, index * 2));
    return createHash("sha256").update(bytes).digest("hex");
  };
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const model = await compileFixedTwoTokenModel(reader, stages["0"]!.cos[0]!, stages["0"]!.sin[0]!);
    assert.equal(fixture.cases.length, 32);
    for (const sample of fixture.cases) {
      const actual = evaluateFixedTwoTokenModel(model, sample.ids);
      assert.equal(hash(actual.layers[0]!), sample.layer0_sha256, `${sample.ids}: camada 0`);
      assert.equal(hash(actual.hidden), sample.final_hidden_sha256, `${sample.ids}: norm final`);
      assert.equal(hash(actual.logits), sample.logits_sha256, `${sample.ids}: logits`);
      assert.equal(actual.tokens[1], sample.next_token, `${sample.ids}: próximo token`);
    }
  } finally { await reader.close(); }
});
