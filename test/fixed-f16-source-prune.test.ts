import assert from "node:assert/strict";
import test from "node:test";
import { pruneFixedF16ScalarDeclarations } from "../src/fixed-f16-source-prune.js";
import { evaluateFixedF16CachedScalarSource, type FixedF16CachedScalarSource } from "../src/fixed-f16-parametric-formulas.js";

test("poda declarações escalares não alcançadas após substituição", () => {
  const source: FixedF16CachedScalarSource = {
    kind: "fixed-f16-cached-scalar-source", inputSize: 1, outputSize: 1, nextCacheId: 0,
    declarations: "const base=(p)=>x[p][0]; const used=(p)=>base(p); const dead=(p)=>42;",
    formulas: ["used(t)"],
  };
  const pruned = pruneFixedF16ScalarDeclarations(source);
  assert.match(pruned.declarations, /const base=/);
  assert.match(pruned.declarations, /const used=/);
  assert.doesNotMatch(pruned.declarations, /const dead=/);
  assert.deepEqual(evaluateFixedF16CachedScalarSource(pruned, [[7], [9]]), [[7], [9]]);
});
