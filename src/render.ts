import type { ModelIR, Operation } from "./types.js";

export function renderEquations(ir: ModelIR): string {
  const lines: string[] = [];
  lines.push(`# ${ir.architecture.modelType}`);
  lines.push(`# MAX_FEATURES é apenas preview: ${ir.preview.outputRows} saídas × ${ir.preview.inputTerms} termos.`);
  for (const op of [...ir.prelude, ...ir.layers.flatMap((layer) => layer.operations), ...ir.epilogue]) {
    lines.push(renderOperation(op));
  }
  return `${lines.join("\n\n")}\n`;
}

function renderOperation(op: Operation): string {
  switch (op.op) {
    case "embedding":
      return `${op.output}[t,d] = ${op.weight.name}[${op.tokenInput}[t],d]${op.scale ? ` * ${op.scale}` : ""}`;
    case "per_layer_embedding":
      return `${op.output}[t,l,d] = ${op.weight.name}[${op.tokenInput}[t], l*${op.layerWidth}+d]${op.scale ? ` * ${op.scale}` : ""}`;
    case "rms_norm": {
      const weight = op.weightTransform === "none" ? "1" : op.weightTransform === "one_plus_weight" ? `(1 + ${op.weight!.name}[d])` : `${op.weight!.name}[d]`;
      return `${op.output}[...,d] = ${op.input}[...,d] * (${op.epsilon} + mean_k(${op.input}[...,k]^2))^(-1/2) * ${weight}`;
    }
    case "linear": {
      const bias = op.bias ? ` + ${op.bias.name}[o]` : "";
      const base = `${op.output}[...,o] = Σ_{i=0}^{${op.inFeatures - 1}} ${op.input}[...,i] * ${op.weight.name}[o,i]${bias}`;
      if (!op.preview) return base;
      const preview = op.preview.rows
        .map((row) => {
          const terms = row.terms.map((term) => `${op.input}[...,${term.inputIndex}] * ${format(term.weight)}`).join(" + ");
          return `  o=${row.outputIndex}: ${terms}${row.omittedInputTerms ? ` + … (${row.omittedInputTerms} termos)` : ""}`;
        })
        .join("\n");
      return `${base}\npreview:\n${preview}${op.preview.omittedOutputRows ? `\n  … (${op.preview.omittedOutputRows} saídas)` : ""}`;
    }
    case "reshape_heads":
      return `${op.output} = reshape_heads(${op.input}, heads=${op.numHeads}, head_dim=${op.headDim}, layout=${op.layout})`;
    case "reshape_per_layer":
      return `${op.output} = reshape(${op.input}, [batch, sequence, ${op.numLayers}, ${op.layerWidth}])`;
    case "select_per_layer":
      return `${op.output}[t,d] = ${op.input}[t,${op.layerIndex},d]`;
    case "tensor_scale":
      return `${op.output} = ${op.input} * ${op.scalar.name}[0]`;
    case "rotary_embedding":
      return `${op.output} = RoPE(${op.input}, positions=${op.positionInput}, theta=${op.theta}, dim=${op.rotaryDim}, type=${op.ropeType}, layout=${op.layout})`;
    case "scaled_dot_product_attention":
      return `${op.output} = softmax_fp32(softcap((QKᵀ) * ${op.scale}${op.scoreSoftcap ? `, ${op.scoreSoftcap}` : ""}) + ${op.maskInput})V`;
    case "activation":
      return `${op.output} = ${op.function}${op.approximation ? `[${op.approximation}]` : ""}(${op.input})`;
    case "elementwise":
      if (op.kind === "add") return `${op.output} = ${op.inputs.join(" + ")}`;
      if (op.kind === "multiply") return `${op.output} = ${op.inputs.join(" * ")}`;
      if (op.kind === "tanh_softcap") return `${op.output} = ${op.scalar} * tanh(${op.inputs[0]} / ${op.scalar})`;
      return `${op.output} = ${op.inputs[0]} * ${op.scalar}`;
  }
}

function format(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  return value.toPrecision(9).replace(/(?:\.0+|(?:(\.\d*?)0+))$/, "$1");
}
