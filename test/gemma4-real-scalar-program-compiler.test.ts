import assert from "node:assert/strict";
import test from "node:test";
import { Gemma4RealExpressionBuilder } from "../src/gemma4-real-expression.js";
import { compileGemma4ScalarCalculationToExactReal } from "../src/gemma4-real-scalar-program-compiler.js";
import { buildGemma4LiteralScalarStatementDataflow, type Gemma4LiteralScalarCalculation } from "../src/gemma4-literal-scalar-calculations.js";
import { buildGemma4LiteralScalarStatementEnvironment, buildGemma4LiteralScalarStatementPrograms } from "../src/gemma4-literal-scalar-statement-programs.js";

test("compila loops escalares locais em uma única raiz real", () => {
  const scalarAssignments = [
    "acc[-1]=F32(0)",
    "acc[d]=F32_FMA(acc[d-1],q[d],k[d]), d=0..3 ascending",
    "y[hidden]=BF16(acc[3])",
  ];
  const programs = buildGemma4LiteralScalarStatementPrograms(scalarAssignments, "y", "loop-real");
  const calculation: Gemma4LiteralScalarCalculation = {
    scope: "vision",
    definitionId: "scores",
    operation: "attention-score-matmul",
    orderedInputs: ["q", "k"],
    output: "y",
    outputCoordinates: ["hidden"],
    learnedOperandRoles: [],
    scalarAssignments,
    statementDataflow: buildGemma4LiteralScalarStatementDataflow(scalarAssignments, "y", "loop-real"),
    statementPrograms: programs,
    statementEnvironment: buildGemma4LiteralScalarStatementEnvironment(programs, {
      output: "y", outputCoordinates: ["hidden"], orderedInputs: ["q", "k"], learnedOperandRoles: [], reductions: [{ indices: ["d=0..3"], source: "assignment" }],
    }, "loop-real"),
    formula: scalarAssignments.at(-1)!,
    dtypePolicy: { inputDtype: "BF16", computeDtype: "pytorch-native-batched-matmul", accumulationDtype: "runtime-defined", outputDtype: "BF16" },
    reduction: { indices: ["d=0..3"], domains: [], order: "runtime-defined" },
    reproducibility: "fail-closed-runtime-reduction",
  };
  const builder = new Gemma4RealExpressionBuilder();
  const compiled = compileGemma4ScalarCalculationToExactReal(calculation, [0], {
    builder,
    resolveTensorElement: (name, coordinates) => builder.input(name, coordinates.map(String)),
    resolveLearnedElement: () => { throw new Error("BMM fixture não possui peso."); },
  });
  const graph = builder.build();
  assert.ok(graph.nodes.every((node) => !JSON.stringify(node).includes("runtime-defined")));
  assert.equal(graph.nodes.filter((node) => node.kind === "input").length, 8);
  assert.equal(builder.requiredNode(compiled.root).kind, "add");
});
