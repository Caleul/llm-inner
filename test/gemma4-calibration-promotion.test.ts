import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { evaluateGemma4CalibrationPromotion, parseGemma4CalibrationPromotionOptions } from "../src/gemma4-calibration-promotion.js";

const hash = "a".repeat(64);
const calibration = (tokensPerSecond: number, tokenIds = [7, 8], terminalHash = hash) => ({
  kind: "gemma4-three-way-calibration", schemaVersion: 1,
  source: "/bundle", configuration: { prompts: 1, tokensPerPrompt: 2, requestThreads: 1, precision: "f32", roundingPolicy: "none" },
  summary: { comparedTokenSteps: 2, directEqualTokenSteps: 2, directRootDivergences: 0, directTokensPerSecond: tokensPerSecond },
  cases: [{ prompt: "x", baseline: { tokenIds: [7, 8] }, direct: { tokenIds, terminalLogitsSha256: terminalHash } }],
});

test("gate promove somente ganho com identidade token/logit", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-promotion-"));
  const baseline = join(directory, "baseline.json"), candidate = join(directory, "candidate.json"), output = join(directory, "promotion.json");
  await writeFile(baseline, JSON.stringify(calibration(20))); await writeFile(candidate, JSON.stringify(calibration(22)));
  try {
    const report = await evaluateGemma4CalibrationPromotion({ baseline, candidate, output, minimumThroughputGainPercent: 5 });
    assert.equal(report.accepted, true); assert.ok(Math.abs(report.metrics.throughputGainPercent - 10) < 1e-9); assert.equal(report.checks.every((check) => check.passed), true);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("gate rejeita divergência e ganho abaixo do mínimo", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-promotion-reject-"));
  const baseline = join(directory, "baseline.json"), candidate = join(directory, "candidate.json");
  await writeFile(baseline, JSON.stringify(calibration(20))); await writeFile(candidate, JSON.stringify(calibration(20.1, [7, 9], "b".repeat(64))));
  try {
    const report = await evaluateGemma4CalibrationPromotion({ baseline, candidate, minimumThroughputGainPercent: 2 });
    assert.equal(report.accepted, false);
    assert.deepEqual(report.checks.filter((check) => !check.passed).map((check) => check.id), ["compiled-token-identity", "terminal-logit-hash-identity", "minimum-throughput-gain"]);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("gate falha fechado sem hash terminal e valida CLI", async () => {
  const directory = await mkdtemp(join(tmpdir(), "gemma4-promotion-invalid-")); const invalid = join(directory, "invalid.json");
  await writeFile(invalid, JSON.stringify({ ...calibration(20), cases: [{ prompt: "x", baseline: { tokenIds: [7, 8] }, direct: { tokenIds: [7, 8] } }] }));
  try {
    await assert.rejects(evaluateGemma4CalibrationPromotion({ baseline: invalid, candidate: invalid, minimumThroughputGainPercent: 0 }), /identidade terminal completa/);
    assert.equal(parseGemma4CalibrationPromotionOptions(["--baseline", "a", "--candidate", "b"]).minimumThroughputGainPercent, 2);
    assert.throws(() => parseGemma4CalibrationPromotionOptions(["--baseline", "a"]), /obrigatórios/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
