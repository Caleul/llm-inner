import { rewriteTextualCalls } from "./direct-numeric-stream.js";
import { mkdtemp, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { SafetensorsCatalogReader } from "./safetensors.js";
import { f16BitsToDyadic } from "./fixed-f16-projection.js";

/** Replace every reached F16 normalization coordinate directly at its call site. */
export async function rewriteDirectNormInputs(
  inputPath: string, outputPath: string, directory: string, weightName: string,
  epsilon: number, width: number, outputName: string, inputName: string,
  factorPosition?: string,
): Promise<{ replacements: number; bytes: number }> {
  if (!/^[a-z][A-Za-z0-9]*$/.test(outputName) || !/^[a-z][A-Za-z0-9]*$/.test(inputName)) {
    throw new Error("Nome de fronteira escalar inválido");
  }
  const reader = new SafetensorsCatalogReader(directory);
  try {
    const tensor = (await reader.inspect()).tensors.get(weightName);
    if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 1 ||
      tensor.logicalShape[0] !== width || !Number.isFinite(epsilon) || epsilon < 0) {
      throw new Error("RMSNorm descoberta incompatível com o checkpoint");
    }
    const weights = await reader.readTensorBytes(tensor);
    const pattern = new RegExp(`\\b${outputName}\\[([a-z][a-z0-9]*)\\]\\[(\\d+)\\]`, "g");
    const factors = new Map<string, string>();
    const factor = (position: string): string => {
      let expression = factors.get(position);
      if (expression) return expression;
      let sum = "0";
      for (let coordinate = 0; coordinate < width; coordinate++) {
        const value = `f16(${inputName}[${position}][${coordinate}])`;
        sum = `Math.fround(${sum}+${value}*${value})`;
      }
      expression = `Math.fround(1/Math.sqrt(Math.fround(Math.fround(${sum}/${width})+${Math.fround(epsilon)})))`;
      factors.set(position, expression);
      return expression;
    };
    return rewriteTextualCalls(inputPath, outputPath, pattern, (match) => {
      const position = match[1]!, coordinate = Number(match[2]);
      if (factorPosition !== undefined && position !== factorPosition) {
        throw new Error("Fator compartilhado usado em outra posição");
      }
      if (!Number.isInteger(coordinate) || coordinate < 0 || coordinate >= width) {
        throw new RangeError("Coordenada de normalização inválida");
      }
      const bits = weights.readUInt16LE(coordinate * 2);
      const dyadic = f16BitsToDyadic(bits);
      const weight = Number(dyadic.coefficient) * 2 ** dyadic.exponent;
      const normalized = `f16Bits(Math.fround(f16(${inputName}[${position}][${coordinate}])*rmsFactor))`;
      const value = `f16Bits(f16(${normalized})*${Object.is(weight, -0) ? "-0" : weight})`;
      return factorPosition === undefined ? `(()=>{const rmsFactor=${factor(position)};return ${value};})()` : value;
    }, 64 * 1024, factorPosition === undefined ? "" : `(()=>{const rmsFactor=${factor(factorPosition)};return `,
    factorPosition === undefined ? "" : "})()");
  } finally {
    await reader.close();
  }
}

/** Share the RMS factor at t and within each key/value position loop. */
export async function rewriteDirectAttentionNormInputs(
  inputPath: string, outputPath: string, directory: string, weightName: string,
  epsilon: number, width: number,
): Promise<{ replacements: number; bytes: number }> {
  const reader = new SafetensorsCatalogReader(directory);
  const temporary = await mkdtemp(join(dirname(outputPath), ".attention-norm-"));
  try {
    const tensor = (await reader.inspect()).tensors.get(weightName);
    if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 1 ||
      tensor.logicalShape[0] !== width || !Number.isFinite(epsilon) || epsilon < 0) {
      throw new Error("Normalização de entrada da atenção inválida");
    }
    const weights = await reader.readTensorBytes(tensor);
    const factor = (position: "t" | "j"): string => {
      let sum = "0";
      for (let coordinate = 0; coordinate < width; coordinate++) {
        const value = `f16(layerInput[${position}][${coordinate}])`;
        sum = `Math.fround(${sum}+${value}*${value})`;
      }
      return `Math.fround(1/Math.sqrt(Math.fround(Math.fround(${sum}/${width})+${Math.fround(epsilon)})))`;
    };
    const loops = join(temporary, "loops");
    const scoped = await rewriteTextualCalls(inputPath, loops, /for\(let j=0;j<n;j\+\+\)\{/g,
      (match) => `${match[0]}const rmsFactorJ=${factor("j")};`, 64 * 1024,
      `(()=>{const rmsFactorT=${factor("t")};return `, "})()");
    if (scoped.replacements === 0) throw new Error("Laços de atenção não reconhecidos");
    return rewriteTextualCalls(loops, outputPath, /\battnInput\[(t|j)\]\[(\d+)\]/g, (match) => {
      const position = match[1] as "t" | "j", coordinate = Number(match[2]);
      if (!Number.isInteger(coordinate) || coordinate < 0 || coordinate >= width) {
        throw new RangeError("Coordenada da atenção inválida");
      }
      const bits = weights.readUInt16LE(coordinate * 2);
      const dyadic = f16BitsToDyadic(bits);
      const weight = Number(dyadic.coefficient) * 2 ** dyadic.exponent;
      const normalized = `f16Bits(Math.fround(f16(layerInput[${position}][${coordinate}])*` +
        (position === "t" ? "rmsFactorT" : "rmsFactorJ") + "))";
      return `f16Bits(f16(${normalized})*${Object.is(weight, -0) ? "-0" : weight})`;
    }, 64 * 1024);
  } finally {
    await reader.close();
    await rm(temporary, { recursive: true, force: true });
  }
}
