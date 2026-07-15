import test from "node:test";
import assert from "node:assert/strict";
import { renderEquations } from "../src/render.js";
import type { ModelIR } from "../src/types.js";

test("MAX_FEATURES limita somente preview, não inFeatures", () => {
  const ir: ModelIR = {
    schemaVersion: 2,
    source: { path: "/tmp/model", format: "safetensors" },
    architecture: {
      modelType: "llama",
      hiddenSize: 4096,
      numLayers: 1,
      numAttentionHeads: 32,
      numKeyValueHeads: 8,
      headDim: 128,
    },
    config: {},
    preview: { outputRows: 1, inputTerms: 2, includeWeights: true },
    inputs: [],
    prelude: [],
    layers: [
      {
        index: 0,
        layerType: "full_attention",
        operations: [
          {
            id: "linear",
            layer: 0,
            op: "linear",
            input: "x",
            output: "y",
            weight: { name: "w", shape: [4096, 4096], storageDtype: "BF16" },
            inFeatures: 4096,
            outFeatures: 4096,
            transposeWeight: true,
            preview: {
              rows: [
                {
                  outputIndex: 0,
                  terms: [
                    { inputIndex: 0, weight: 1 },
                    { inputIndex: 1, weight: 2 },
                  ],
                  omittedInputTerms: 4094,
                },
              ],
              omittedOutputRows: 4095,
            },
            dtypePolicy: {},
          },
        ],
      },
    ],
    epilogue: [],
    fidelity: { exactByConstruction: true, assumptions: [], unsupported: [], warnings: [] },
  };
  const rendered = renderEquations(ir);
  assert.match(rendered, /Σ_\{i=0\}\^\{4095\}/);
  assert.match(rendered, /4094 termos/);
});
