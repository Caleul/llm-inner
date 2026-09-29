import {
  addDyadic, f16BitsToDyadic, f32BitsToDyadic, multiplyDyadic,
  roundDyadicToF16IfElse, roundDyadicToF32IfElse,
  type Dyadic, type FixedF16Projection,
} from "./fixed-f16-projection.js";

export type FixedScalarExpression =
  | { kind: "input"; index: number }
  | { kind: "constant"; value: { coefficient: string; exponent: number } }
  | { kind: "add" | "multiply"; left: FixedScalarExpression; right: FixedScalarExpression }
  | { kind: "round-f32" | "round-f16"; value: FixedScalarExpression };

/** One closed calculation per output coordinate. No tensor or projection lookup remains. */
export interface FixedScalarFunctions {
  kind: "fixed-scalar-functions";
  inputSize: number;
  outputs: FixedScalarExpression[];
}

function constant(value: Dyadic): FixedScalarExpression {
  return { kind: "constant", value: { coefficient: value.coefficient.toString(), exponent: value.exponent } };
}
function numeric(expression: Extract<FixedScalarExpression, { kind: "constant" }>): Dyadic {
  return { coefficient: BigInt(expression.value.coefficient), exponent: expression.value.exponent };
}
function binary(kind: "add" | "multiply", left: FixedScalarExpression, right: FixedScalarExpression): FixedScalarExpression {
  if (left.kind === "constant" && right.kind === "constant") {
    return constant(kind === "add" ? addDyadic(numeric(left), numeric(right)) : multiplyDyadic(numeric(left), numeric(right)));
  }
  if (kind === "multiply" && (left.kind === "constant" && BigInt(left.value.coefficient) === 0n ||
    right.kind === "constant" && BigInt(right.value.coefficient) === 0n)) return constant({ coefficient: 0n, exponent: 0 });
  if (kind === "add" && left.kind === "constant" && BigInt(left.value.coefficient) === 0n) return right;
  if (kind === "add" && right.kind === "constant" && BigInt(right.value.coefficient) === 0n) return left;
  return { kind, left, right };
}
function round(kind: "round-f32" | "round-f16", value: FixedScalarExpression): FixedScalarExpression {
  if (value.kind === "constant") {
    const dyadic = numeric(value);
    const bits = kind === "round-f32" ? roundDyadicToF32IfElse(dyadic) : roundDyadicToF16IfElse(dyadic);
    return constant(kind === "round-f32" ? f32BitsToDyadic(bits) : f16BitsToDyadic(bits));
  }
  if (value.kind === kind) return value;
  return { kind, value };
}

export function scalarizeFixedF16Projection(projection: FixedF16Projection): FixedScalarFunctions {
  if (projection.arithmetic !== "f32-ascending-products-and-sum" || projection.rounding !== "binary16-nearest-ties-to-even-conditional") {
    throw new Error("Política da projeção escalar incompatível.");
  }
  return {
    kind: "fixed-scalar-functions", inputSize: projection.inputSize,
    outputs: projection.rows.map((row) => {
      let sum: FixedScalarExpression = constant({ coefficient: 0n, exponent: 0 });
      for (const term of row.terms) {
        const weight = constant(f16BitsToDyadic(term.weightBits));
        // Two finite F16 significands have at most 22 product bits; their
        // exponent range also fits F32. This F32 rounding is an identity.
        const product = binary("multiply", { kind: "input", index: term.input }, weight);
        sum = round("round-f32", binary("add", sum, product));
      }
      return round("round-f16", sum);
    }),
  };
}

/** Inline every producer expression into every consumer output; no graph sharing. */
export function substituteFixedScalarFunctions(
  consumer: FixedScalarFunctions, producer: FixedScalarFunctions,
): FixedScalarFunctions {
  if (consumer.inputSize !== producer.outputs.length) throw new Error("Dimensões incompatíveis na substituição escalar.");
  const replace = (expression: FixedScalarExpression): FixedScalarExpression => {
    switch (expression.kind) {
      case "input": {
        const output = producer.outputs[expression.index];
        if (!output) throw new Error(`Dimensão ${expression.index} ausente no produtor.`);
        return output;
      }
      case "constant": return expression;
      case "add": case "multiply": return binary(expression.kind, replace(expression.left), replace(expression.right));
      case "round-f32": case "round-f16": return round(expression.kind, replace(expression.value));
    }
  };
  return { kind: "fixed-scalar-functions", inputSize: producer.inputSize, outputs: consumer.outputs.map(replace) };
}

export function evaluateFixedScalarFunctions(program: FixedScalarFunctions, inputBits: readonly number[]): number[] {
  if (inputBits.length !== program.inputSize) throw new Error("Tamanho da entrada escalar inválido.");
  const inputs = inputBits.map(f16BitsToDyadic);
  const evaluate = (expression: FixedScalarExpression): Dyadic => {
    switch (expression.kind) {
      case "input": return inputs[expression.index]!;
      case "constant": return numeric(expression);
      case "add": return addDyadic(evaluate(expression.left), evaluate(expression.right));
      case "multiply": return multiplyDyadic(evaluate(expression.left), evaluate(expression.right));
      case "round-f32": return f32BitsToDyadic(roundDyadicToF32IfElse(evaluate(expression.value)));
      case "round-f16": return f16BitsToDyadic(roundDyadicToF16IfElse(evaluate(expression.value)));
    }
  };
  return program.outputs.map((output) => roundDyadicToF16IfElse(evaluate(output)));
}
