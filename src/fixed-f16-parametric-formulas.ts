import { compileFixedF16Projection, f16BitsToDyadic, f32BitsToDyadic, roundDyadicToF16IfElse, type FixedF16Projection } from "./fixed-f16-projection.js";
import { type SafetensorsCatalogReader } from "./safetensors.js";

/** Flat source formulas: F_d(x,t), valid for every t in an arbitrary-length input. */
export interface FixedF16ParametricFormulas {
  kind: "fixed-f16-parametric-formulas";
  inputSize: number;
  outputSize: number;
  formulas: string[];
  arithmetic: "f32-ascending-products-and-sum";
}

function literal(bits: number): string {
  const value = f16BitsToDyadic(bits);
  const numeric = Number(value.coefficient) * 2 ** value.exponent;
  if (!Number.isFinite(numeric)) throw new Error("Peso F16 não finito.");
  return Object.is(numeric, -0) ? "-0" : String(numeric);
}

/** Decode learned weights at compile time; the emitted formulas contain only literals and x[t][i]. */
export function scalarizeFixedF16ProjectionForAnyLength(projection: FixedF16Projection): FixedF16ParametricFormulas {
  if (projection.arithmetic !== "f32-ascending-products-and-sum" || projection.rounding !== "binary16-nearest-ties-to-even-conditional") {
    throw new Error("Política numérica incompatível com a fórmula F16 paramétrica.");
  }
  const formulas = projection.rows.map((row) => {
    let accumulator = "0";
    for (const term of row.terms) {
      const weight = literal(term.weightBits);
      const product = weight === "1" ? `f16(x[t][${term.input}])` :
        weight === "-1" ? `(-f16(x[t][${term.input}]))` :
        `Math.fround(f16(x[t][${term.input}]) * ${weight})`;
      accumulator = `Math.fround(${accumulator} + ${product})`;
    }
    return `f16Bits(${accumulator})`;
  });
  return { kind: "fixed-f16-parametric-formulas", inputSize: projection.inputSize,
    outputSize: projection.outputSize, formulas, arithmetic: projection.arithmetic };
}

export async function compileFixedF16ParametricFormulas(reader: SafetensorsCatalogReader, tensor: string): Promise<FixedF16ParametricFormulas> {
  return scalarizeFixedF16ProjectionForAnyLength(await compileFixedF16Projection(reader, tensor));
}

/** Physically substitute every previous-dimension formula; no producer call or stage input remains. */
export function substituteFixedF16ParametricFormulas(
  consumer: FixedF16ParametricFormulas, producer: FixedF16ParametricFormulas, maxCharacters = 10_000_000,
): FixedF16ParametricFormulas {
  if (consumer.inputSize !== producer.outputSize) throw new Error("Dimensões incompatíveis na substituição.");
  let size = 0;
  const formulas = consumer.formulas.map((formula) => {
    const expanded = formula.replace(/x\[([a-z][a-z0-9]*)\]\[(\d+)\]/g, (_match, position: string, rawIndex: string) => {
      const index = Number(rawIndex);
      const replacement = producer.formulas[index];
      if (replacement === undefined) throw new Error(`Dimensão ${index} ausente no produtor.`);
      return `(${replacement.replace(/x\[t\]/g, `x[${position}]`)})`;
    });
    size += expanded.length;
    if (size > maxCharacters) throw new Error(`Expansão literal excede ${maxCharacters} caracteres; nenhuma fórmula parcial será retornada.`);
    return expanded;
  });
  return { kind: "fixed-f16-parametric-formulas", inputSize: producer.inputSize,
    outputSize: consumer.outputSize, formulas, arithmetic: consumer.arithmetic };
}

function f16(bits: number): number {
  const value = f16BitsToDyadic(bits);
  return Number(value.coefficient) * 2 ** value.exponent;
}
function f16Bits(value: number): number {
  const buffer = new DataView(new ArrayBuffer(4));
  buffer.setFloat32(0, value, true);
  return roundDyadicToF16IfElse(f32BitsToDyadic(buffer.getUint32(0, true)));
}

function causalSum(token: number, term: (key: number) => number): number {
  let accumulator = Math.fround(0);
  for (let key = 0; key <= token; key++) accumulator = Math.fround(accumulator + Math.fround(term(key)));
  return accumulator;
}

/** Execute source-generated formulas for every token; n is determined only at call time. */
export function evaluateFixedF16ParametricFormulas(program: FixedF16ParametricFormulas, input: readonly (readonly number[])[]): number[][] {
  if (input.some((row) => row.length !== program.inputSize)) throw new Error("Dimensão de entrada incompatível.");
  const functions = program.formulas.map((formula) => {
    if (!/^f16Bits\([\s\S]*\)$/.test(formula) || /\b(?:weight|projection|layer|eval|require|import)\b/.test(formula)) {
      throw new Error("Fórmula escalar inválida ou não substituída.");
    }
    return new Function("x", "t", "f16", "f16Bits", "causalSum", `return ${formula};`) as
      (x: readonly (readonly number[])[], t: number, f16: (bits: number) => number,
        f16Bits: (value: number) => number, causalSum: (token: number, term: (key: number) => number) => number) => number;
  });
  return input.map((_, token) => functions.map((formula) => formula(input, token, f16, f16Bits, causalSum)));
}
