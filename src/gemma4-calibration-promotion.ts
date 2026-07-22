import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

interface CalibrationCase {
  prompt: string;
  baseline: { tokenIds: number[] };
  direct: { tokenIds: number[]; terminalLogitsSha256: string };
}

interface CalibrationReport {
  kind: "gemma4-three-way-calibration";
  schemaVersion: 1;
  source: string;
  configuration: { prompts: number; tokensPerPrompt: number; requestThreads: number; precision: string; roundingPolicy: string };
  summary: { comparedTokenSteps: number; directEqualTokenSteps: number; directRootDivergences: number; directTokensPerSecond: number };
  cases: CalibrationCase[];
}

export interface Gemma4CalibrationPromotionOptions {
  baseline: string;
  candidate: string;
  output?: string;
  minimumThroughputGainPercent: number;
}

export interface Gemma4CalibrationPromotionReport {
  kind: "gemma4-calibration-promotion";
  schemaVersion: 1;
  accepted: boolean;
  baseline: string;
  candidate: string;
  thresholds: { minimumThroughputGainPercent: number };
  metrics: { baselineTokensPerSecond: number; candidateTokensPerSecond: number; throughputGainPercent: number; comparedTokenSteps: number; comparedPrompts: number };
  checks: Array<{ id: string; passed: boolean; detail: string }>;
}

export async function evaluateGemma4CalibrationPromotion(options: Gemma4CalibrationPromotionOptions): Promise<Gemma4CalibrationPromotionReport> {
  const baseline = await readCalibration(options.baseline, "baseline");
  const candidate = await readCalibration(options.candidate, "candidato");
  const checks: Gemma4CalibrationPromotionReport["checks"] = [];
  const add = (id: string, passed: boolean, detail: string) => checks.push({ id, passed, detail });
  const sameShape = baseline.source === candidate.source
    && baseline.configuration.prompts === candidate.configuration.prompts
    && baseline.configuration.tokensPerPrompt === candidate.configuration.tokensPerPrompt
    && baseline.configuration.requestThreads === candidate.configuration.requestThreads
    && baseline.configuration.precision === candidate.configuration.precision
    && baseline.configuration.roundingPolicy === candidate.configuration.roundingPolicy
    && baseline.cases.length === candidate.cases.length;
  add("same-workload", sameShape, `${baseline.cases.length}/${candidate.cases.length} prompts; ${baseline.configuration.tokensPerPrompt}/${candidate.configuration.tokensPerPrompt} tokens; fonte ${baseline.source === candidate.source ? "igual" : "divergente"}`);
  add("baseline-token-parity", baseline.summary.directRootDivergences === 0 && baseline.summary.directEqualTokenSteps === baseline.summary.comparedTokenSteps,
    `${baseline.summary.directEqualTokenSteps}/${baseline.summary.comparedTokenSteps} tokens; ${baseline.summary.directRootDivergences} divergências raiz`);
  add("candidate-token-parity", candidate.summary.directRootDivergences === 0 && candidate.summary.directEqualTokenSteps === candidate.summary.comparedTokenSteps,
    `${candidate.summary.directEqualTokenSteps}/${candidate.summary.comparedTokenSteps} tokens; ${candidate.summary.directRootDivergences} divergências raiz`);
  let promptsEqual = sameShape, baselineTokensEqual = sameShape, directTokensEqual = sameShape, terminalHashesEqual = sameShape;
  if (sameShape) for (let index = 0; index < baseline.cases.length; index++) {
    const before = baseline.cases[index]!, after = candidate.cases[index]!;
    promptsEqual &&= before.prompt === after.prompt;
    baselineTokensEqual &&= equalNumbers(before.baseline.tokenIds, after.baseline.tokenIds);
    directTokensEqual &&= equalNumbers(before.direct.tokenIds, after.direct.tokenIds);
    terminalHashesEqual &&= before.direct.terminalLogitsSha256 === after.direct.terminalLogitsSha256;
  }
  add("prompt-identity", promptsEqual, promptsEqual ? "prompts idênticos e na mesma ordem" : "prompts divergentes");
  add("authoritative-token-identity", baselineTokensEqual, baselineTokensEqual ? "tokens originais idênticos" : "baseline autoritativo mudou");
  add("compiled-token-identity", directTokensEqual, directTokensEqual ? "tokens compilados idênticos" : "tokens compilados mudaram");
  add("terminal-logit-hash-identity", terminalHashesEqual, terminalHashesEqual ? "hash terminal idêntico por prompt" : "hash terminal ausente ou divergente");
  const baselineTps = baseline.summary.directTokensPerSecond, candidateTps = candidate.summary.directTokensPerSecond;
  const throughputGainPercent = (candidateTps / baselineTps - 1) * 100;
  add("minimum-throughput-gain", throughputGainPercent >= options.minimumThroughputGainPercent,
    `${throughputGainPercent.toFixed(3)}% observado; mínimo ${options.minimumThroughputGainPercent.toFixed(3)}%`);
  const report: Gemma4CalibrationPromotionReport = {
    kind: "gemma4-calibration-promotion", schemaVersion: 1, accepted: checks.every((check) => check.passed),
    baseline: resolve(options.baseline), candidate: resolve(options.candidate), thresholds: { minimumThroughputGainPercent: options.minimumThroughputGainPercent },
    metrics: { baselineTokensPerSecond: baselineTps, candidateTokensPerSecond: candidateTps, throughputGainPercent, comparedTokenSteps: candidate.summary.comparedTokenSteps, comparedPrompts: candidate.cases.length }, checks,
  };
  if (options.output) await writeFile(resolve(options.output), `${JSON.stringify(report, null, 2)}\n`);
  return report;
}

export function parseGemma4CalibrationPromotionOptions(argv: readonly string[]): Gemma4CalibrationPromotionOptions {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index], value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
    values.set(flag, value);
  }
  const known = new Set(["--baseline", "--candidate", "--output", "--min-throughput-gain-percent"]);
  for (const flag of values.keys()) if (!known.has(flag)) throw new Error(`Flag desconhecida: ${flag}.`);
  const baseline = values.get("--baseline"), candidate = values.get("--candidate");
  if (!baseline || !candidate) throw new Error("--baseline e --candidate são obrigatórios.");
  const minimumThroughputGainPercent = Number(values.get("--min-throughput-gain-percent") ?? "2");
  if (!Number.isFinite(minimumThroughputGainPercent) || minimumThroughputGainPercent < 0) throw new Error("--min-throughput-gain-percent deve ser finito e não negativo.");
  const output = values.get("--output");
  return { baseline: resolve(baseline), candidate: resolve(candidate), ...(output ? { output: resolve(output) } : {}), minimumThroughputGainPercent };
}

async function readCalibration(path: string, label: string): Promise<CalibrationReport> {
  let value: unknown;
  try { value = JSON.parse(await readFile(resolve(path), "utf8")) as unknown; }
  catch (error) { throw new Error(`Relatório ${label} inválido: ${(error as Error).message}`); }
  const report = value as Partial<CalibrationReport>;
  if (report.kind !== "gemma4-three-way-calibration" || report.schemaVersion !== 1 || typeof report.source !== "string" || !report.configuration || !report.summary || !Array.isArray(report.cases)) throw new Error(`Relatório ${label} não é uma calibração Gemma 4 v1.`);
  const numbers = [report.configuration.prompts, report.configuration.tokensPerPrompt, report.configuration.requestThreads, report.summary.comparedTokenSteps, report.summary.directEqualTokenSteps, report.summary.directRootDivergences, report.summary.directTokensPerSecond];
  if (numbers.some((number) => typeof number !== "number" || !Number.isFinite(number)) || report.summary.directTokensPerSecond! <= 0) throw new Error(`Relatório ${label} contém métricas inválidas.`);
  if (typeof report.configuration.precision !== "string" || typeof report.configuration.roundingPolicy !== "string" || report.configuration.prompts !== report.cases.length) throw new Error(`Relatório ${label} contém configuração de carga inválida.`);
  for (const entry of report.cases) {
    if (typeof entry?.prompt !== "string" || !Array.isArray(entry.baseline?.tokenIds) || !Array.isArray(entry.direct?.tokenIds) || !/^[0-9a-f]{64}$/.test(entry.direct?.terminalLogitsSha256 ?? "")) throw new Error(`Relatório ${label} não contém identidade terminal completa por prompt.`);
  }
  return report as CalibrationReport;
}

function equalNumbers(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}
