import assert from "node:assert/strict";
import test from "node:test";
import { access, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { compileF16LiteralMultiplyBranches, compileF16SiluBranches, compileFiniteF16UnaryBranches } from "../src/fixed-f16-conditional-linear.js";
import { compileFixedF16Projection, f16BitsToDyadic, multiplyDyadic, roundDyadicToF16IfElse } from "../src/fixed-f16-projection.js";
import { SafetensorsCatalogReader } from "../src/safetensors.js";
import { sleefExpF32 } from "../src/sleef-f32.js";

test("multiplicação literal F16 vira somente ramos com folhas lineares constantes", () => {
  const weightBits = 0x3800;
  const rule = compileF16LiteralMultiplyBranches(weightBits);
  assert.ok(rule.intervals > 1);
  assert.ok(rule.characters > 0);
  assert.doesNotMatch(rule.source, /Math\.|f16Bits|scalar_cache_/);
  const evaluate = new Function(`return (${rule.source});`)() as (bits: number) => number;
  const weight = f16BitsToDyadic(weightBits);
  for (let bits = 0; bits <= 0xffff; bits++) {
    if ((bits & 0x7c00) === 0x7c00) continue;
    const expected = roundDyadicToF16IfElse(multiplyDyadic(f16BitsToDyadic(bits), weight));
    assert.equal(evaluate(bits), expected, `bits=${bits}`);
  }
  assert.throws(() => evaluate(0x7c00), /F16 não finito/);
  assert.throws(() => evaluate(0xfc00), /F16 não finito/);
});

test("redutor finito reduz ReLU F16 a dois caminhos afins", () => {
  const rule = compileFiniteF16UnaryBranches((bits) => bits < 0x8000 ? bits : 0);
  assert.ok(rule.intervals <= 3);
  assert.ok(rule.characters < 300);
  const evaluate = new Function(`return (${rule.source});`)() as (bits: number) => number;
  for (const bits of [0, 1, 0x3c00, 0x7bff, 0x8000, 0xbc00, 0xfbff]) {
    assert.equal(evaluate(bits), bits < 0x8000 ? bits : 0);
  }
});

test("peso literal do fixture passa pela mesma regra condicional", async (context) => {
  const directory = resolve("artifacts/tiny-random-llama");
  try { await access(join(directory, "model.safetensors")); } catch { context.skip("checkpoint opcional ausente"); return; }
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const projection = await compileFixedF16Projection(reader, "model.layers.0.self_attn.q_proj.weight");
    const weightBits = projection.rows[0]!.terms[0]!.weightBits;
    const rule = compileF16LiteralMultiplyBranches(weightBits);
    const evaluate = new Function(`return (${rule.source});`)() as (bits: number) => number;
    const weight = f16BitsToDyadic(weightBits);
    for (let bits = 0; bits <= 0xffff; bits++) {
      if ((bits & 0x7c00) === 0x7c00) continue;
      assert.equal(evaluate(bits), roundDyadicToF16IfElse(multiplyDyadic(f16BitsToDyadic(bits), weight)));
    }
  } finally { await reader.close(); }
});

test("SiLU F16 do PyTorch capturada é substituída por ramos lineares sem exp", async () => {
  const rule = compileF16SiluBranches();
  assert.doesNotMatch(rule.source, /Math\.|exp\(|f16Bits|scalar_cache_/);
  const evaluate = new Function(`return (${rule.source});`)() as (bits: number) => number;
  const expected = await readFile(resolve("numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin"));
  assert.equal(expected.length, 65536 * 2);
  for (let bits = 0; bits <= 0xffff; bits++) {
    if ((bits & 0x7c00) === 0x7c00) continue;
    assert.equal(evaluate(bits), expected.readUInt16LE(bits * 2), `bits=${bits}`);
  }
});

test("exp F32 de SLEEF reproduz divergências observadas contra Math.exp", () => {
  const cases = [
    [3262825344, 305935162], [3184947456, 1063685354],
    [3243791216, 901008368], [3246524984, 867911208],
    [3254988800, 667789498], [3222659602, 1036314212],
  ];
  const buffer = new DataView(new ArrayBuffer(4));
  for (const [inputBits, expectedBits] of cases) {
    buffer.setUint32(0, inputBits!, true);
    const input = buffer.getFloat32(0, true);
    buffer.setFloat32(0, sleefExpF32(input), true);
    assert.equal(buffer.getUint32(0, true), expectedBits);
  }
});
