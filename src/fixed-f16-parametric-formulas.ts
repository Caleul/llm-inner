import { compileFixedF16Projection, f16BitsToDyadic, f32BitsToDyadic, roundDyadicToF16IfElse, type FixedF16Projection } from "./fixed-f16-projection.js";
import { type SafetensorsCatalogReader } from "./safetensors.js";

/** Flat source formulas: F_d(x,t), valid for every t in an arbitrary-length input. */
export interface FixedF16ParametricFormulas {
  kind: "fixed-f16-parametric-formulas";
  inputSize: number;
  outputSize: number;
  formulas: string[];
  arithmetic: "f32-ascending-products-and-sum" | "f32-interleaved-four-lane-pairwise";
  rmsFactor?: { expression: string; compactFormulas: string[]; symbol: string };
}
export type FixedF16FormulaArithmetic = FixedF16ParametricFormulas["arithmetic"];

/** Source for one requested scalar F(t,d); declarations contain only scalar caches. */
export interface FixedF16CachedScalarSource {
  kind: "fixed-f16-cached-scalar-source";
  inputSize: number;
  outputSize: number;
  formulas: string[];
  declarations: string;
  nextCacheId: number;
}

export interface FixedF16ScalarMetrics {
  calls: Record<string, number>;
}

export function scalarSourceFromFormulas(program: FixedF16ParametricFormulas): FixedF16CachedScalarSource {
  return { kind: "fixed-f16-cached-scalar-source", inputSize: program.inputSize,
    outputSize: program.outputSize, formulas: program.formulas, declarations: "", nextCacheId: 0 };
}

function scalarCases(formulas: readonly string[]): string {
  return formulas.map((formula, dimension) => `case ${dimension}: return ${formula};`).join("\n");
}

/** Substitute a preceding scalar calculation once; repeated coordinates use its local cache. */
export function composeCachedScalarSource(
  consumer: FixedF16ParametricFormulas, producer: FixedF16CachedScalarSource,
): FixedF16CachedScalarSource {
  if (consumer.inputSize !== producer.outputSize) throw new Error("Dimensões incompatíveis na composição escalar.");
  const id = producer.nextCacheId;
  const name = `scalar_cache_${id}`;
  const map = `scalar_values_${id}`;
  const declaration = `const ${map} = new Map();\n` +
    `const ${name} = (p,d) => { const key = p * ${producer.outputSize} + d; ` +
    `if (${map}.has(key)) return ${map}.get(key); ` +
    `const value = (() => { const t = p; switch(d) { ${scalarCases(producer.formulas)} ` +
    `default: throw new RangeError("Dimensão escalar inválida"); } })(); ` +
    `${map}.set(key,value); return value; };\n`;
  const formulas = consumer.formulas.map((formula) => formula.replace(
    /x\[([a-z][a-z0-9]*)\]\[([a-z][a-z0-9]*|\d+)\]/g,
    (_match, position: string, dimension: string) => `${name}(${position},${dimension})`,
  ));
  return { kind: "fixed-f16-cached-scalar-source", inputSize: producer.inputSize,
    outputSize: consumer.outputSize, declarations: producer.declarations + declaration,
    formulas, nextCacheId: id + 1 };
}

/** Compose two already factored scalar functions, renaming cache bindings before substitution. */
export function composeCachedScalarSources(
  consumer: FixedF16CachedScalarSource, producer: FixedF16CachedScalarSource,
): FixedF16CachedScalarSource {
  if (consumer.inputSize !== producer.outputSize) throw new Error("Dimensões incompatíveis na composição escalar.");
  const shifted = (source: string) => source.replace(/\b(scalar_cache_|scalar_values_)(\d+)\b/g,
    (_match, prefix: string, id: string) => `${prefix}${Number(id) + producer.nextCacheId + 1}`);
  const shiftedFormulas = consumer.formulas.map(shifted);
  const shiftedDeclarations = shifted(consumer.declarations);
  const selector = composeCachedScalarSource({ kind: "fixed-f16-parametric-formulas", inputSize: producer.outputSize,
    outputSize: consumer.outputSize, formulas: shiftedFormulas,
    arithmetic: "f32-interleaved-four-lane-pairwise" }, producer);
  const selectorName = `scalar_cache_${producer.nextCacheId}`;
  const declarations = producer.declarations + selector.declarations.slice(producer.declarations.length) +
    shiftedDeclarations.replace(/x\[([a-z][a-z0-9]*)\]\[([a-z][a-z0-9]*|\d+)\]/g,
      (_match, position: string, dimension: string) => `${selectorName}(${position},${dimension})`);
  return { ...selector, declarations,
    nextCacheId: producer.nextCacheId + consumer.nextCacheId + 1 };
}

/** Prepare the generated scalar source once, retaining no input or checkpoint state. */
export function prepareFixedF16CachedScalarSource(
  program: FixedF16CachedScalarSource, metrics?: FixedF16ScalarMetrics,
): (input: readonly (readonly number[])[]) => number[][] {
  const count = <T extends (...args: never[]) => number>(name: string, functionValue: T): T =>
    (metrics ? ((...args: Parameters<T>) => {
      metrics.calls[name] = (metrics.calls[name] ?? 0) + 1;
      return functionValue(...args);
    }) as T : functionValue);
  const scalarF16 = count("f16", f16);
  const scalarF16Bits = count("f16Bits", f16Bits);
  const scalarAdd16 = count("add16", add16);
  const scalarMul16 = count("mul16", mul16);
  const scalarNeg16 = count("neg16", neg16);
  const scalarRopeBits = count("ropeBits", ropeBits);
  const scalarMath = metrics ? Object.assign(Object.create(Math) as Math,
    Object.fromEntries(["fround", "exp", "sqrt", "sin", "cos", "max"].map((name) =>
      [name, count(`Math.${name}`, (Math as unknown as Record<string, (...args: never[]) => number>)[name]!)]))) : Math;
  const validate = (input: readonly (readonly number[])[]) => {
    if (input.some((row) => row.length !== program.inputSize)) throw new Error("Dimensão de entrada incompatível.");
  };
  if (program.outputSize > 1024 && program.nextCacheId > 0) {
    if (program.formulas.every((formula) => formula === program.formulas[0])) {
      const factory = new Function("x", "f16", "f16Bits", "add16", "mul16", "neg16", "ropeBits", "Math",
        `${program.declarations} return (t,d) => ${program.formulas[0]};`) as (...args: any[]) =>
        (t: number, d: number) => number;
      return (input) => {
        validate(input);
        const scalar = factory(input, scalarF16, scalarF16Bits, scalarAdd16,
          scalarMul16, scalarNeg16, scalarRopeBits, scalarMath);
        return input.map((_, t) => Array.from({ length: program.outputSize }, (_unused, d) => scalar(t, d)));
      };
    }
    const name = `scalar_cache_${program.nextCacheId - 1}`;
    if (program.formulas.some((formula) => !formula.includes(`${name}(`))) {
      throw new Error("Saída larga requer uma fonte escalar final fatorada.");
    }
    const coreFactory = new Function("x", "f16", "f16Bits", "add16", "mul16", "neg16", "ropeBits", "Math",
      `${program.declarations} return ${name};`) as (...args: any[]) => (t: number, d: number) => number;
    const rows = program.formulas.map((formula) => new Function("t", name, "f16", "f16Bits", "add16", "mul16",
      "neg16", "ropeBits", "Math", `return ${formula};`) as (...args: any[]) => number);
    return (input) => {
      validate(input);
      const core = coreFactory(input, scalarF16, scalarF16Bits, scalarAdd16, scalarMul16, scalarNeg16, scalarRopeBits, scalarMath);
      return input.map((_, t) => rows.map((row) => row(t, core, scalarF16, scalarF16Bits,
        scalarAdd16, scalarMul16, scalarNeg16, scalarRopeBits, scalarMath)));
    };
  }
  const source = `${program.declarations} return (t,d) => { switch(d) { ${scalarCases(program.formulas)} ` +
    `default: throw new RangeError("Dimensão escalar inválida"); } };`;
  const factory = new Function("x", "f16", "f16Bits", "add16", "mul16", "neg16", "ropeBits", "Math", source) as
    (x: readonly (readonly number[])[], f16: (bits: number) => number, f16Bits: (value: number) => number,
      add16: (left: number, right: number) => number, mul16: (left: number, right: number) => number,
      neg16: (bits: number) => number,
      rope: (position: number, dimension: number, headDim: number, theta: number, sine: number) => number,
      math: Math)
      => (t: number, d: number) => number;
  return (input) => {
    validate(input);
    const scalar = factory(input, scalarF16, scalarF16Bits, scalarAdd16, scalarMul16, scalarNeg16, scalarRopeBits, scalarMath);
    return input.map((_, t) => Array.from({ length: program.outputSize }, (_unused, d) => scalar(t, d)));
  };
}

export function evaluateFixedF16CachedScalarSource(
  program: FixedF16CachedScalarSource, input: readonly (readonly number[])[],
): number[][] {
  return prepareFixedF16CachedScalarSource(program)(input);
}

function literal(bits: number): string {
  const value = f16BitsToDyadic(bits);
  const numeric = Number(value.coefficient) * 2 ** value.exponent;
  if (!Number.isFinite(numeric)) throw new Error("Peso F16 não finito.");
  return Object.is(numeric, -0) ? "-0" : String(numeric);
}

/** Decode learned weights at compile time; the emitted formulas contain only literals and x[t][i]. */
export function scalarizeFixedF16ProjectionForAnyLength(
  projection: FixedF16Projection, arithmetic: FixedF16FormulaArithmetic = projection.arithmetic,
): FixedF16ParametricFormulas {
  if (projection.rounding !== "binary16-nearest-ties-to-even-conditional" ||
    (arithmetic !== "f32-ascending-products-and-sum" && arithmetic !== "f32-interleaved-four-lane-pairwise")) {
    throw new Error("Política numérica incompatível com a fórmula F16 paramétrica.");
  }
  const formulas = projection.rows.map((row) => {
    let accumulator = "0";
    const lanes = ["0", "0", "0", "0"];
    for (const term of row.terms) {
      const weight = literal(term.weightBits);
      const product = weight === "1" ? `f16(x[t][${term.input}])` :
        weight === "-1" ? `(-f16(x[t][${term.input}]))` :
        `Math.fround(f16(x[t][${term.input}]) * ${weight})`;
      if (arithmetic === "f32-interleaved-four-lane-pairwise") {
        const lane = term.input & 3;
        lanes[lane] = `Math.fround(${lanes[lane]} + ${product})`;
      } else accumulator = `Math.fround(${accumulator} + ${product})`;
    }
    if (arithmetic === "f32-interleaved-four-lane-pairwise") {
      accumulator = `Math.fround(Math.fround(${lanes[0]} + ${lanes[1]}) + Math.fround(${lanes[2]} + ${lanes[3]}))`;
    }
    return `f16Bits(${accumulator})`;
  });
  return { kind: "fixed-f16-parametric-formulas", inputSize: projection.inputSize,
    outputSize: projection.outputSize, formulas, arithmetic };
}

export async function compileFixedF16ParametricFormulas(
  reader: SafetensorsCatalogReader, tensor: string, arithmetic?: FixedF16FormulaArithmetic,
): Promise<FixedF16ParametricFormulas> {
  const projection = await compileFixedF16Projection(reader, tensor);
  return scalarizeFixedF16ProjectionForAnyLength(projection, arithmetic ?? projection.arithmetic);
}

/** Literal row selection plus the declared four-lane F32 reduction for a wide output projection. */
export async function compileFixedF16CachedWideLinearSource(
  reader: SafetensorsCatalogReader, tensorName: string,
): Promise<FixedF16CachedScalarSource> {
  const tensor = (await reader.inspect()).tensors.get(tensorName);
  if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 2 ||
    !tensor.logicalShape[0] || !tensor.logicalShape[1]) throw new Error(`${tensorName}: matriz F16 incompatível.`);
  const [outputs, inputs] = tensor.logicalShape as [number, number];
  const bytes = await reader.readTensorBytes(tensor);
  const rows = Array.from({ length: outputs }, (_unused, output) => {
    const weights = Array.from({ length: inputs }, (_unusedInput, input) =>
      literal(bytes.readUInt16LE((output * inputs + input) * 2)));
    return `case ${output}: return [${weights.join(",")}];`;
  }).join("");
  const declarations = `const wideRow=(d)=>{switch(d){${rows}default:throw new RangeError("Dimensão inválida");}};\n` +
    `const wideDot=(p,d)=>{const row=wideRow(d),lanes=[0,0,0,0];` +
    `for(let i=0;i<${inputs};i++){const weight=row[i],input=f16(x[p][i]);` +
    `const product=weight===1?input:weight===-1?-input:Math.fround(input*weight);` +
    `const lane=i&3;lanes[lane]=Math.fround(lanes[lane]+product);}` +
    `return f16Bits(Math.fround(Math.fround(lanes[0]+lanes[1])+Math.fround(lanes[2]+lanes[3])));};\n`;
  return { kind: "fixed-f16-cached-scalar-source", inputSize: inputs, outputSize: outputs,
    declarations, nextCacheId: 0, formulas: Array.from({ length: outputs }, () => "wideDot(t,d)") };
}

/** Literal token selection for one embedding dimension, with arbitrary input token IDs. */
export async function compileFixedF16EmbeddingParametricFormulas(
  reader: SafetensorsCatalogReader, tensorName: string,
): Promise<FixedF16ParametricFormulas> {
  const tensor = (await reader.inspect()).tensors.get(tensorName);
  if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 2 ||
    !tensor.logicalShape[0] || !tensor.logicalShape[1]) throw new Error(`${tensorName}: embedding F16 incompatível.`);
  const [vocabulary, dimensions] = tensor.logicalShape as [number, number];
  const bytes = await reader.readTensorBytes(tensor);
  const formulas = Array.from({ length: dimensions }, (_, dimension) => {
    const cases = Array.from({ length: vocabulary }, (_unused, token) =>
      `case ${token}: return ${bytes.readUInt16LE((token * dimensions + dimension) * 2)};`).join("");
    return `(() => { switch(x[t][0]) { ${cases} default: throw new RangeError("Token fora do vocabulário"); } })()`;
  });
  return { kind: "fixed-f16-parametric-formulas", inputSize: 1, outputSize: dimensions,
    formulas, arithmetic: "f32-ascending-products-and-sum" };
}

/** One token switch containing literal rows, instead of repeating 32k cases for every dimension. */
export async function compileFixedF16CachedEmbeddingSource(
  reader: SafetensorsCatalogReader, tensorName: string,
): Promise<FixedF16CachedScalarSource> {
  const tensor = (await reader.inspect()).tensors.get(tensorName);
  if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 2 ||
    !tensor.logicalShape[0] || !tensor.logicalShape[1]) throw new Error(`${tensorName}: embedding F16 incompatível.`);
  const [vocabulary, dimensions] = tensor.logicalShape as [number, number];
  const bytes = await reader.readTensorBytes(tensor);
  const cases = Array.from({ length: vocabulary }, (_unused, token) => {
    const row = Array.from({ length: dimensions }, (_unusedDimension, dimension) =>
      bytes.readUInt16LE((token * dimensions + dimension) * 2));
    return `case ${token}: return [${row.join(",")}][d];`;
  }).join("");
  const declarations = `const embeddedScalar=(p,d)=>{ switch(x[p][0]){ ${cases} ` +
    `default: throw new RangeError("Token fora do vocabulário"); } };\n`;
  return { kind: "fixed-f16-cached-scalar-source", inputSize: 1, outputSize: dimensions,
    declarations, nextCacheId: 0,
    formulas: Array.from({ length: dimensions }, () => "embeddedScalar(t,d)") };
}

/** Per-dimension RMSNorm formula with a literal learned scale and an F32 variance reduction. */
export async function compileFixedF16RmsNormParametricFormulas(
  reader: SafetensorsCatalogReader, tensorName: string, epsilon: number,
): Promise<FixedF16ParametricFormulas> {
  const tensor = (await reader.inspect()).tensors.get(tensorName);
  if (!tensor || tensor.storageDtype !== "F16" || tensor.logicalShape.length !== 1 ||
    !tensor.logicalShape[0] || !Number.isFinite(epsilon) || epsilon < 0) {
    throw new Error(`${tensorName}: RMSNorm requer vetor F16 e epsilon finito.`);
  }
  const size = tensor.logicalShape[0]!;
  const weights = await reader.readTensorBytes(tensor);
  let sum = "0";
  for (let dimension = 0; dimension < size; dimension++) {
    const x = `f16(x[t][${dimension}])`;
    sum = `Math.fround(${sum} + Math.fround(${x} * ${x}))`;
  }
  const variance = `Math.fround(${sum} / ${size})`;
  const inverse = `Math.fround(1 / Math.sqrt(Math.fround(${variance} + Math.fround(${epsilon}))))`;
  const formulas = Array.from({ length: size }, (_, dimension) => {
    const normalized = `f16Bits(Math.fround(f16(x[t][${dimension}]) * ${inverse}))`;
    const weight = literal(weights.readUInt16LE(dimension * 2));
    return weight === "1" ? normalized : `f16Bits(f16(${normalized}) * ${weight})`;
  });
  const compactFormulas = Array.from({ length: size }, (_, dimension) => {
    const normalized = `f16Bits(Math.fround(f16(x[t][${dimension}]) * rms_factor))`;
    const weight = literal(weights.readUInt16LE(dimension * 2));
    return weight === "1" ? normalized : `f16Bits(f16(${normalized}) * ${weight})`;
  });
  return { kind: "fixed-f16-parametric-formulas", inputSize: size, outputSize: size, formulas,
    arithmetic: "f32-ascending-products-and-sum", rmsFactor: {
      expression: inverse, compactFormulas, symbol: `rms_factor_${tensorName.replace(/\W/g, "_")}` } };
}

/** Hoist only the repeated scalar RMS denominator into the final coordinate function. */
export function substituteFactoredRmsNorm(
  consumer: FixedF16ParametricFormulas, norm: FixedF16ParametricFormulas, maxCharacters = 50_000_000,
): FixedF16ParametricFormulas {
  if (!norm.rmsFactor) throw new Error("Produtor não é uma RMSNorm fatorável.");
  const compact: FixedF16ParametricFormulas = { kind: norm.kind, inputSize: norm.inputSize,
    outputSize: norm.outputSize, arithmetic: norm.arithmetic,
    formulas: norm.rmsFactor.compactFormulas.map((formula) =>
      formula.replace(/\brms_factor\b/g, `${norm.rmsFactor!.symbol}(t)`)) };
  const body = substituteFixedF16ParametricFormulas(consumer, compact, maxCharacters);
  const prefix = `(() => { const ${norm.rmsFactor.symbol} = (p) => ${norm.rmsFactor.expression.replace(/x\[t\]/g, "x[p]")}; return `;
  const formulas = body.formulas.map((formula) => `${prefix}${formula}; })()`);
  if (formulas.reduce((sum, formula) => sum + formula.length, 0) > maxCharacters) {
    throw new Error(`Fatoração RMSNorm excede ${maxCharacters} caracteres.`);
  }
  return { ...body, formulas };
}

/** One scalar inverse RMS per position, shared by every output coordinate. */
export function scalarSourceFromFactoredRmsNorm(norm: FixedF16ParametricFormulas): FixedF16CachedScalarSource {
  if (!norm.rmsFactor) throw new Error("RMSNorm fatorável necessária.");
  const { symbol, expression, compactFormulas } = norm.rmsFactor;
  const values = `${symbol}_values`;
  const declarations = `const ${values}=new Map(); const ${symbol}=(p)=>{ if(${values}.has(p)) return ${values}.get(p); ` +
    `const value=${expression.replace(/x\[t\]/g, "x[p]")}; ${values}.set(p,value); return value; };\n`;
  return { kind: "fixed-f16-cached-scalar-source", inputSize: norm.inputSize,
    outputSize: norm.outputSize, declarations, nextCacheId: 0,
    formulas: compactFormulas.map((formula) => formula.replace(/\brms_factor\b/g, `${symbol}(t)`)) };
}

/** Add the original token coordinate after an already substituted branch. */
export function addFixedF16Residual(branch: FixedF16ParametricFormulas): FixedF16ParametricFormulas {
  if (branch.inputSize !== branch.outputSize) throw new Error("Residual requer a mesma dimensão de entrada e saída.");
  return { ...branch, formulas: branch.formulas.map((formula, dimension) =>
    `add16(x[t][${dimension}],${formula})`) };
}

/** Fuse gate, SiLU, up and down by physical substitution into each output dimension. */
export async function compileFixedF16MlpParametricFormulas(
  reader: SafetensorsCatalogReader, tensors: { gate: string; up: string; down: string }, arithmetic: FixedF16FormulaArithmetic,
  maxCharacters = 50_000_000,
): Promise<FixedF16ParametricFormulas> {
  const gate = await compileFixedF16ParametricFormulas(reader, tensors.gate, arithmetic);
  const up = await compileFixedF16ParametricFormulas(reader, tensors.up, arithmetic);
  const down = await compileFixedF16ParametricFormulas(reader, tensors.down, arithmetic);
  if (gate.inputSize !== up.inputSize || gate.outputSize !== up.outputSize || down.inputSize !== gate.outputSize) {
    throw new Error("MLP: shapes incompatíveis.");
  }
  const hidden = gate.formulas.map((formula, dimension) => {
    const silu = "f16Bits(f16(gate_value) / (1 + Math.exp(-f16(gate_value))))";
    return `(() => { const gate_value = ${formula}; const up_value = ${up.formulas[dimension]!}; ` +
      `return f16Bits(f16(${silu}) * f16(up_value)); })()`;
  });
  return substituteFixedF16ParametricFormulas(down, {
    kind: "fixed-f16-parametric-formulas", inputSize: gate.inputSize,
    outputSize: gate.outputSize, formulas: hidden, arithmetic,
  }, maxCharacters);
}

let mlpSourceId = 0;

/** Factor gate/up activations before the down projection; all weights remain literal. */
export async function compileFixedF16CachedMlpSource(
  reader: SafetensorsCatalogReader, tensors: { gate: string; up: string; down: string },
  arithmetic: FixedF16FormulaArithmetic,
): Promise<FixedF16CachedScalarSource> {
  const gate = await compileFixedF16ParametricFormulas(reader, tensors.gate, arithmetic);
  const up = await compileFixedF16ParametricFormulas(reader, tensors.up, arithmetic);
  const down = await compileFixedF16ParametricFormulas(reader, tensors.down, arithmetic);
  if (gate.inputSize !== up.inputSize || gate.outputSize !== up.outputSize || down.inputSize !== gate.outputSize) {
    throw new Error("MLP: shapes incompatíveis.");
  }
  const suffix = `_${mlpSourceId++}`;
  const projection = (name: string, program: FixedF16ParametricFormulas) =>
    `const ${name}Values${suffix}=new Map(); const ${name}${suffix}=(p,d)=>{ ` +
    `const key=p*${program.outputSize}+d; if(${name}Values${suffix}.has(key)) return ${name}Values${suffix}.get(key); ` +
    `const value=(() => { const t=p; switch(d){ ${scalarCases(program.formulas)} default: throw new RangeError("Dimensão inválida"); } })(); ` +
    `${name}Values${suffix}.set(key,value); return value; };\n`;
  const declarations = projection("gate", gate) + projection("up", up) +
    `const hiddenValues${suffix}=new Map(); const hidden${suffix}=(p,d)=>{ ` +
    `const key=p*${gate.outputSize}+d; if(hiddenValues${suffix}.has(key)) return hiddenValues${suffix}.get(key); ` +
    `const g=gate${suffix}(p,d),u=up${suffix}(p,d); ` +
    `const activated=f16Bits(f16(g)/(1+Math.exp(-f16(g)))); ` +
    `const value=f16Bits(f16(activated)*f16(u)); hiddenValues${suffix}.set(key,value); return value; };\n`;
  const formulas = down.formulas.map((formula) => formula.replace(/x\[t\]\[(\d+)\]/g,
    (_match, dimension: string) => `hidden${suffix}(t,${dimension})`));
  return { kind: "fixed-f16-cached-scalar-source", inputSize: gate.inputSize,
    outputSize: down.outputSize, formulas, declarations, nextCacheId: 0 };
}

/** Expand eager causal attention into scalar formulas with a runtime key bound t. */
export async function compileFixedF16AttentionParametricFormulas(
  reader: SafetensorsCatalogReader, tensors: { q: string; k: string; v: string; o: string },
  heads: number, kvHeads: number, headDim: number, ropeTheta: number,
  arithmetic: FixedF16FormulaArithmetic, maxCharacters = 100_000_000,
): Promise<FixedF16ParametricFormulas> {
  if (!Number.isInteger(heads) || !Number.isInteger(kvHeads) || !Number.isInteger(headDim) ||
    heads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim % 2 !== 0 || heads % kvHeads !== 0 ||
    !Number.isFinite(ropeTheta) || ropeTheta <= 0) throw new Error("Configuração de atenção incompatível.");
  const q = await compileFixedF16ParametricFormulas(reader, tensors.q, arithmetic);
  const k = await compileFixedF16ParametricFormulas(reader, tensors.k, arithmetic);
  const v = await compileFixedF16ParametricFormulas(reader, tensors.v, arithmetic);
  const o = await compileFixedF16ParametricFormulas(reader, tensors.o, arithmetic);
  if (q.outputSize !== heads * headDim || k.outputSize !== kvHeads * headDim ||
    v.outputSize !== kvHeads * headDim || o.inputSize !== q.outputSize ||
    q.inputSize !== k.inputSize || q.inputSize !== v.inputSize) throw new Error("Shapes Q/K/V/O incompatíveis.");
  const at = (formula: string, position: string) => formula.replace(/x\[t\]/g, `x[${position}]`);
  const rotate = (formula: string, partner: string, position: string, dimension: number) => {
    const signedPartner = dimension < headDim / 2 ? `neg16(${at(partner, position)})` : at(partner, position);
    return `add16(mul16(${at(formula, position)},ropeBits(${position},${dimension},${headDim},${ropeTheta},0)),` +
      `mul16(${signedPartner},ropeBits(${position},${dimension},${headDim},${ropeTheta},1)))`;
  };
  const context = Array.from({ length: heads * headDim }, (_, output) => {
    const head = Math.floor(output / headDim), dimension = output % headDim;
    const kvHead = Math.floor(head / (heads / kvHeads));
    const query = Array.from({ length: headDim }, (_, dim) =>
      rotate(q.formulas[head * headDim + dim]!, q.formulas[head * headDim + (dim + headDim / 2) % headDim]!, "t", dim));
    const scoreTerms = Array.from({ length: headDim }, (_, dim) => {
      const key = rotate(k.formulas[kvHead * headDim + dim]!,
        k.formulas[kvHead * headDim + (dim + headDim / 2) % headDim]!, "j", dim);
      return `Math.fround(f16(${query[dim]!}) * f16(${key}))`;
    });
    let score = "0";
    for (const term of scoreTerms) score = `Math.fround(${score} + ${term})`;
    const scoreBody = `return mul16(f16Bits(${score}),f16Bits(${1 / Math.sqrt(headDim)}));`;
    const value = at(v.formulas[kvHead * headDim + dimension]!, "j");
    return `(() => { const score = (j) => { ${scoreBody} }; ` +
      `let maximum = -Infinity; for (let j = 0; j <= t; j++) maximum = Math.max(maximum, f16(score(j))); ` +
      `let denominator = Math.fround(0); for (let j = 0; j <= t; j++) denominator = Math.fround(denominator + Math.fround(Math.exp(Math.fround(f16(score(j)) - maximum)))); ` +
      `let acc = Math.fround(0); for (let j = 0; j <= t; j++) { ` +
      `const probability = f16Bits(Math.fround(Math.fround(Math.exp(Math.fround(f16(score(j)) - maximum))) / denominator)); ` +
      `acc = Math.fround(acc + Math.fround(f16(probability) * f16(${value}))); } return f16Bits(acc); })()`;
  });
  const result = substituteFixedF16ParametricFormulas(o, { kind: "fixed-f16-parametric-formulas",
    inputSize: q.inputSize, outputSize: context.length, formulas: context, arithmetic }, maxCharacters);
  return result;
}

let attentionSourceId = 0;

/** Same eager attention equation, with each Q/K/V, score and context scalar evaluated once per coordinate. */
export async function compileFixedF16CachedAttentionSource(
  reader: SafetensorsCatalogReader, tensors: { q: string; k: string; v: string; o: string },
  heads: number, kvHeads: number, headDim: number, ropeTheta: number,
  arithmetic: FixedF16FormulaArithmetic,
): Promise<FixedF16CachedScalarSource> {
  if (!Number.isInteger(heads) || !Number.isInteger(kvHeads) || !Number.isInteger(headDim) ||
    heads <= 0 || kvHeads <= 0 || headDim <= 0 || headDim % 2 !== 0 || heads % kvHeads !== 0 ||
    !Number.isFinite(ropeTheta) || ropeTheta <= 0) throw new Error("Configuração de atenção incompatível.");
  const q = await compileFixedF16ParametricFormulas(reader, tensors.q, arithmetic);
  const k = await compileFixedF16ParametricFormulas(reader, tensors.k, arithmetic);
  const v = await compileFixedF16ParametricFormulas(reader, tensors.v, arithmetic);
  const o = await compileFixedF16ParametricFormulas(reader, tensors.o, arithmetic);
  if (q.outputSize !== heads * headDim || k.outputSize !== kvHeads * headDim ||
    v.outputSize !== kvHeads * headDim || o.inputSize !== q.outputSize ||
    q.inputSize !== k.inputSize || q.inputSize !== v.inputSize) throw new Error("Shapes Q/K/V/O incompatíveis.");
  const id = attentionSourceId++;
  const suffix = `_${id}`;
  const projection = (name: string, program: FixedF16ParametricFormulas) =>
    `const ${name}Values${suffix} = new Map(); const ${name}${suffix} = (p,d) => { ` +
    `const key=p*${program.outputSize}+d; if (${name}Values${suffix}.has(key)) return ${name}Values${suffix}.get(key); ` +
    `const value=(() => { const t=p; switch(d){ ${scalarCases(program.formulas)} default: throw new RangeError("Dimensão inválida"); } })(); ` +
    `${name}Values${suffix}.set(key,value); return value; };\n`;
  const rotate = (project: string) => `const ${project}Rot${suffix} = (p,h,d) => { ` +
    `const partner=(d+${headDim / 2})%${headDim}; ` +
    `const signed=d<${headDim / 2}?neg16(${project}${suffix}(p,h*${headDim}+partner)):${project}${suffix}(p,h*${headDim}+partner); ` +
    `return add16(mul16(${project}${suffix}(p,h*${headDim}+d),ropeBits(p,d,${headDim},${ropeTheta},0)),` +
    `mul16(signed,ropeBits(p,d,${headDim},${ropeTheta},1))); };\n`;
  const declarations = projection("q", q) + projection("k", k) + projection("v", v) +
    rotate("q") + rotate("k") +
    `const scoreValues${suffix}=new Map(); const score${suffix}=(p,j,h)=>{ const key=(p*(p+1)/2+j)*${heads}+h; ` +
    `if(scoreValues${suffix}.has(key)) return scoreValues${suffix}.get(key); ` +
    `let total=Math.fround(0); for(let d=0;d<${headDim};d++){ ` +
    `total=Math.fround(total+Math.fround(f16(qRot${suffix}(p,h,d))*f16(kRot${suffix}(j,Math.floor(h/${heads / kvHeads}),d)))); } ` +
    `const value=mul16(f16Bits(total),f16Bits(${1 / Math.sqrt(headDim)})); ` +
    `scoreValues${suffix}.set(key,value); return value; };\n` +
    `const contextValues${suffix}=new Map(); const context${suffix}=(p,coordinate)=>{ ` +
    `const key=p*${heads * headDim}+coordinate; if(contextValues${suffix}.has(key)) return contextValues${suffix}.get(key); ` +
    `const h=Math.floor(coordinate/${headDim}),d=coordinate%${headDim},kvh=Math.floor(h/${heads / kvHeads}); ` +
    `let maximum=-Infinity; for(let j=0;j<=p;j++) maximum=Math.max(maximum,f16(score${suffix}(p,j,h))); ` +
    `let denominator=Math.fround(0); for(let j=0;j<=p;j++) ` +
    `denominator=Math.fround(denominator+Math.fround(Math.exp(Math.fround(f16(score${suffix}(p,j,h))-maximum)))); ` +
    `let acc=Math.fround(0); for(let j=0;j<=p;j++){ ` +
    `const probability=f16Bits(Math.fround(Math.fround(Math.exp(Math.fround(f16(score${suffix}(p,j,h))-maximum)))/denominator)); ` +
    `acc=Math.fround(acc+Math.fround(f16(probability)*f16(v${suffix}(j,kvh*${headDim}+d)))); } ` +
    `const value=f16Bits(acc); contextValues${suffix}.set(key,value); return value; };\n`;
  const formulas = o.formulas.map((formula) => formula.replace(/x\[t\]\[(\d+)\]/g,
    (_match, dimension: string) => `context${suffix}(t,${dimension})`));
  return { kind: "fixed-f16-cached-scalar-source", inputSize: q.inputSize,
    outputSize: o.outputSize, formulas, declarations, nextCacheId: 0 };
}

function neg16(bits: number): number { return bits ^ 0x8000; }
function mul16(left: number, right: number): number { return f16Bits(f16(left) * f16(right)); }
function add16(left: number, right: number): number { return f16Bits(f16(left) + f16(right)); }
function ropeBits(position: number, dimension: number, headDim: number, theta: number, sine: number): number {
  const frequency = 1 / theta ** (2 * (dimension % (headDim / 2)) / headDim);
  const angle = Math.fround(position * Math.fround(frequency));
  return f16Bits(sine ? Math.sin(angle) : Math.cos(angle));
}

/** Physically substitute every previous-dimension formula; no producer call or stage input remains. */
export function substituteFixedF16ParametricFormulas(
  consumer: FixedF16ParametricFormulas, producer: FixedF16ParametricFormulas, maxCharacters = 10_000_000,
): FixedF16ParametricFormulas {
  if (consumer.inputSize !== producer.outputSize) throw new Error("Dimensões incompatíveis na substituição.");
  const estimatedCharacters = estimateFixedF16ParametricSubstitutionCharacters(consumer, producer);
  if (estimatedCharacters > BigInt(maxCharacters)) {
    throw new Error(`Expansão literal requer ${estimatedCharacters} caracteres; limite ${maxCharacters}. Nenhuma fórmula parcial será retornada.`);
  }
  let size = 0;
  const formulas = consumer.formulas.map((formula) => {
    const expanded = formula.replace(/x\[([a-z][a-z0-9]*)\]\[(\d+)\]/g, (_match, position: string, rawIndex: string) => {
      const index = Number(rawIndex);
      const replacement = producer.formulas[index];
      if (replacement === undefined) throw new Error(`Dimensão ${index} ausente no produtor.`);
      return `(${replacement.replace(/x\[t\]/g, `x[${position}]`).replace(/(rms_factor_[a-zA-Z0-9_]+)\(t\)/g, `$1(${position})`)})`;
    });
    size += expanded.length;
    if (size > maxCharacters) throw new Error(`Expansão literal excede ${maxCharacters} caracteres; nenhuma fórmula parcial será retornada.`);
    return expanded;
  });
  return { kind: "fixed-f16-parametric-formulas", inputSize: producer.inputSize,
    outputSize: consumer.outputSize, formulas, arithmetic: consumer.arithmetic };
}

/** Count substituted source bytes before allocating any expanded formula. */
export function estimateFixedF16ParametricSubstitutionCharacters(
  consumer: FixedF16ParametricFormulas, producer: FixedF16ParametricFormulas,
): bigint {
  if (consumer.inputSize !== producer.outputSize) throw new Error("Dimensões incompatíveis na estimativa.");
  let total = 0n;
  for (const formula of consumer.formulas) {
    let length = BigInt(formula.length);
    for (const match of formula.matchAll(/x\[([a-z][a-z0-9]*)\]\[(\d+)\]/g)) {
      const replacement = producer.formulas[Number(match[2])];
      if (replacement === undefined) throw new Error(`Dimensão ${match[2]} ausente no produtor.`);
      const occurrences = [...replacement.matchAll(/x\[t\]/g)].length;
      const factorOccurrences = [...replacement.matchAll(/rms_factor_[a-zA-Z0-9_]+\(t\)/g)].length;
      length += BigInt(replacement.length + 2 - match[0].length +
        (occurrences + factorOccurrences) * (match[1]!.length - 1));
    }
    total += length;
  }
  return total;
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
    if (!(formula.startsWith("f16Bits(") || formula.startsWith("add16(") || formula.startsWith("(() => { const rms_factor_")) ||
      /\b(?:weight|projection|layer|eval|require|import)\b/.test(formula)) {
      throw new Error("Fórmula escalar inválida ou não substituída.");
    }
    return new Function("x", "t", "f16", "f16Bits", "causalSum", "add16", "mul16", "neg16", "ropeBits", `return ${formula};`) as
      (x: readonly (readonly number[])[], t: number, f16: (bits: number) => number,
        f16Bits: (value: number) => number, causalSum: (token: number, term: (key: number) => number) => number,
        add16: (left: number, right: number) => number, mul16: (left: number, right: number) => number,
        neg16: (value: number) => number, ropeBits: (position: number, dimension: number, headDim: number, theta: number, sine: number) => number) => number;
  });
  return input.map((_, token) => functions.map((formula) => formula(input, token, f16, f16Bits, causalSum, add16, mul16, neg16, ropeBits)));
}
