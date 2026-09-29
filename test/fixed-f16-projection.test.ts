import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm, access, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { SafetensorsCatalogReader } from "../src/safetensors.js";
import { compileFixedF16Projection, evaluateFixedF16Projection, f16BitsToDyadic, roundDyadicToF16IfElse, roundDyadicToF32IfElse } from "../src/fixed-f16-projection.js";
import { evaluateFixedScalarFunctions, scalarizeFixedF16Projection, substituteFixedScalarFunctions } from "../src/fixed-f16-scalar-functions.js";
import { evaluateFixedTwoTokenAttentionScores } from "../src/fixed-f16-attention-scores.js";

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
      score: number[][][][]; scaled: number[][][][]; masked: number[][][][] }>;
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
  }
});
