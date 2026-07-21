import { readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { createGemma4RealComparisonServer, parseGemma4RealServerOptions, type Gemma4RealComparisonRunnerOptions } from "./gemma4-real-compare-server.js";

export const GEMMA4_CALIBRATION_PROMPTS = [
  "The capital of France is",
  "Two plus two equals",
  "Write one short sentence about the Moon:",
  "Translate to Portuguese: Good morning",
  "Complete the Python expression: sum([1, 2, 3]) ==",
  "Water freezes at",
  "Era uma vez um pequeno robô que",
  "Answer yes or no: Is the Earth round?",
] as const;

export interface Gemma4CalibrationCliOptions {
  prompts: string[];
  output: string;
  maxNewTokens: number;
  requestThreads: number;
  precision: "f32" | "f64";
  roundingPolicy: "none" | "layer-bf16" | "operation-bf16";
  runner: Gemma4RealComparisonRunnerOptions & { host: string; port: number };
}

interface ThreeWayCase {
  prompt: string;
  inputIds: number[][];
  baselineGeneratedTokenIds: number[];
  baselineGeneratedText: string;
  candidateGeneratedTokenIds: number[];
  candidateGeneratedText: string;
  generatedTokensEqual: boolean;
  firstDivergentStep: number | null;
  performance: { baselineSeconds: number; candidateSeconds: number; baselineTokensPerSecond: number; candidateTokensPerSecond: number; candidateSpeedup: number };
  steps: Array<{ step: number; contextsEqualBeforeStep: boolean; baselineToken: number; candidateToken: number; baselineTopLogits: unknown; candidateTopLogits: unknown }>;
  direct: {
    generatedTokenIds: number[];
    generatedText: string;
    elapsedSeconds: number;
    tokensPerSecond: number;
    linearThreads: number;
    maxReadMiB?: number;
    finalHeadReadMiB?: number;
    linearReferenceDispatches?: number;
    wholeNativeBf16Dispatches?: number;
    processRssBytes?: number;
    processMaxRssKiB?: number;
    linearBatchDispatches?: number;
    linearBatchedProjectionTiles?: number;
    fusedMlpRounding?: "off" | "bf16" | "real" | "native-bf16";
    fusedMlpDispatches?: number;
    fusedFfnRounding?: "off" | "native-bf16";
    fusedFfnDispatches?: number;
    fusedDecoderLayerRounding?: "off" | "native-bf16";
    fusedDecoderLayerDispatches?: number;
    fusedPleRounding?: "off" | "bf16" | "real";
    fusedPlePreludeRounding?: "off" | "bf16" | "real";
    fusedPleDispatches?: number;
    fusedPlePreludeDispatches?: number;
    finalHeadCompute?: "f32" | "native-bf16" | "native-bf16-whole";
    nativeAttentionRounding?: "off" | "bf16" | "real";
    nativeAttentionDispatches?: number;
    fusedAttentionRounding?: "off" | "bf16" | "real" | "native-bf16";
    fusedAttentionDispatches?: number;
    referenceSeconds?: number;
    batchSeconds?: number;
    fusedMlpSeconds?: number;
    fusedFfnSeconds?: number;
    fusedDecoderLayerSeconds?: number;
    fusedPleSeconds?: number;
    fusedPlePreludeSeconds?: number;
    nativeAttentionSeconds?: number;
    fusedAttentionSeconds?: number;
    tokensEqualBaseline: boolean;
    firstDivergentStep: number | null;
    steps: Array<{ step: number; tokenId: number; forwardSeconds: number; topLogits: unknown }>;
  };
}

export interface Gemma4ThreeWayCalibrationReport {
  kind: "gemma4-three-way-calibration";
  schemaVersion: 1;
  source: string;
  configuration: Record<string, unknown>;
  initialization: Record<string, unknown>;
  summary: Record<string, number>;
  cases: Array<Record<string, unknown>>;
}

export async function runGemma4ThreeWayCalibration(options: Gemma4CalibrationCliOptions): Promise<Gemma4ThreeWayCalibrationReport> {
  const server = createGemma4RealComparisonServer(options.runner);
  await new Promise<void>((accept, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => accept());
  });
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("Servidor de calibração não expôs uma porta TCP.");
    const endpoint = `http://127.0.0.1:${address.port}`;
    const status = await waitUntilReady(endpoint);
    const cases: ThreeWayCase[] = [];
    for (const prompt of options.prompts) {
      const response = await fetch(`${endpoint}/api/compare`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt, maxNewTokens: options.maxNewTokens, threads: options.requestThreads, precision: options.precision, roundingPolicy: options.roundingPolicy }),
      });
      const body = await response.json() as ThreeWayCase | { error?: string };
      if (!response.ok) throw new Error(`Calibração falhou para ${JSON.stringify(prompt)}: ${"error" in body ? body.error : response.statusText}`);
      assertThreeWayCase(body, prompt);
      cases.push(body);
    }
    const report = summarizeGemma4ThreeWayCalibration(cases, options, status);
    await writeFile(options.output, `${JSON.stringify(report, null, 2)}\n`);
    return report;
  } finally {
    await new Promise<void>((accept, reject) => server.close((error) => error ? reject(error) : accept()));
  }
}

export function summarizeGemma4ThreeWayCalibration(cases: ThreeWayCase[], options: { maxNewTokens: number; requestThreads: number; precision: "f32" | "f64"; roundingPolicy: "none" | "layer-bf16" | "operation-bf16"; runner: Gemma4RealComparisonRunnerOptions }, status: Record<string, unknown>): Gemma4ThreeWayCalibrationReport {
  if (cases.length === 0) throw new Error("Calibração requer ao menos um caso.");
  const totalSteps = cases.reduce((total, entry) => total + entry.baselineGeneratedTokenIds.length, 0);
  const compatibilityEqualSteps = cases.reduce((total, entry) => total + equalTokenSteps(entry.baselineGeneratedTokenIds, entry.candidateGeneratedTokenIds), 0);
  const directEqualSteps = cases.reduce((total, entry) => total + equalTokenSteps(entry.baselineGeneratedTokenIds, entry.direct.generatedTokenIds), 0);
  const baselineSeconds = cases.reduce((total, entry) => total + entry.performance.baselineSeconds, 0);
  const compatibilitySeconds = cases.reduce((total, entry) => total + entry.performance.candidateSeconds, 0);
  const directSeconds = cases.reduce((total, entry) => total + entry.direct.elapsedSeconds, 0);
  const compatibilityPromptsEqual = cases.filter((entry) => entry.generatedTokensEqual).length;
  const directPromptsEqual = cases.filter((entry) => entry.direct.tokensEqualBaseline).length;
  const promptsThreeWayEqual = cases.filter((entry) => entry.generatedTokensEqual && entry.direct.tokensEqualBaseline).length;
  return {
    kind: "gemma4-three-way-calibration",
    schemaVersion: 1,
    source: options.runner.source,
    configuration: {
      prompts: cases.length, tokensPerPrompt: options.maxNewTokens, requestThreads: options.requestThreads,
      directThreads: options.runner.directThreads, directMaxReadMiB: options.runner.directMaxReadMiB ?? 16, directFinalHeadReadMiB: options.runner.directFinalHeadReadMiB ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? 32 : options.runner.directMaxReadMiB ?? 16), directLinearBackend: options.runner.directLinearBackend ?? "pytorch", directFusedMlp: options.runner.directFusedMlp ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "native-bf16" : "real"), directFusedFfn: options.runner.directFusedFfn ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "native-bf16" : "off"), directFusedDecoderLayer: options.runner.directFusedDecoderLayer ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "native-bf16" : "off"), directFusedPle: options.runner.directFusedPle ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "bf16" : "off"), directFusedPlePrelude: options.runner.directFusedPlePrelude ?? "off", directFinalHead: options.runner.directFinalHead ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "native-bf16" : "f32"), directNativeAttention: options.runner.directNativeAttention ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "real" : "off"), directFusedAttention: options.runner.directFusedAttention ?? ((options.runner.directLinearBackend ?? "pytorch") === "pytorch" ? "native-bf16" : "off"), precision: options.precision, roundingPolicy: options.roundingPolicy,
    },
    initialization: status,
    summary: {
      promptsThreeWayEqual,
      promptThreeWayAgreementRate: promptsThreeWayEqual / cases.length,
      compatibilityPromptsEqual,
      compatibilityPromptAgreementRate: compatibilityPromptsEqual / cases.length,
      directPromptsEqual,
      directPromptAgreementRate: directPromptsEqual / cases.length,
      comparedTokenSteps: totalSteps,
      compatibilityEqualTokenSteps: compatibilityEqualSteps,
      compatibilityTokenAgreementRate: compatibilityEqualSteps / totalSteps,
      directEqualTokenSteps: directEqualSteps,
      directTokenAgreementRate: directEqualSteps / totalSteps,
      baselineSeconds,
      compatibilitySeconds,
      directSeconds,
      baselineTokensPerSecond: totalSteps / baselineSeconds,
      compatibilityTokensPerSecond: totalSteps / compatibilitySeconds,
      directTokensPerSecond: totalSteps / directSeconds,
      compatibilityVsBaselineThroughputRatio: baselineSeconds / compatibilitySeconds,
      directVsBaselineThroughputRatio: baselineSeconds / directSeconds,
    },
    cases: cases.map((entry) => ({
      prompt: entry.prompt,
      inputIds: entry.inputIds,
      threeWayTokensEqual: entry.generatedTokensEqual && entry.direct.tokensEqualBaseline,
      baseline: { tokenIds: entry.baselineGeneratedTokenIds, text: entry.baselineGeneratedText, seconds: entry.performance.baselineSeconds, tokensPerSecond: entry.performance.baselineTokensPerSecond },
      compatibility: { tokenIds: entry.candidateGeneratedTokenIds, text: entry.candidateGeneratedText, tokensEqualBaseline: entry.generatedTokensEqual, firstDivergentStep: entry.firstDivergentStep, seconds: entry.performance.candidateSeconds, tokensPerSecond: entry.performance.candidateTokensPerSecond },
      direct: { tokenIds: entry.direct.generatedTokenIds, text: entry.direct.generatedText, tokensEqualBaseline: entry.direct.tokensEqualBaseline, firstDivergentStep: entry.direct.firstDivergentStep, seconds: entry.direct.elapsedSeconds, tokensPerSecond: entry.direct.tokensPerSecond, linearThreads: entry.direct.linearThreads,
        ...(entry.direct.maxReadMiB === undefined ? {} : { maxReadMiB: entry.direct.maxReadMiB }),
        ...(entry.direct.finalHeadReadMiB === undefined ? {} : { finalHeadReadMiB: entry.direct.finalHeadReadMiB }),
        ...(entry.direct.linearReferenceDispatches === undefined ? {} : { linearReferenceDispatches: entry.direct.linearReferenceDispatches }),
        ...(entry.direct.wholeNativeBf16Dispatches === undefined ? {} : { wholeNativeBf16Dispatches: entry.direct.wholeNativeBf16Dispatches }),
        ...(entry.direct.processRssBytes === undefined ? {} : { processRssBytes: entry.direct.processRssBytes }),
        ...(entry.direct.processMaxRssKiB === undefined ? {} : { processMaxRssKiB: entry.direct.processMaxRssKiB }),
        ...(entry.direct.linearBatchDispatches === undefined ? {} : { linearBatchDispatches: entry.direct.linearBatchDispatches }),
        ...(entry.direct.linearBatchedProjectionTiles === undefined ? {} : { linearBatchedProjectionTiles: entry.direct.linearBatchedProjectionTiles }),
        ...(entry.direct.fusedMlpRounding === undefined ? {} : { fusedMlpRounding: entry.direct.fusedMlpRounding }),
        ...(entry.direct.fusedMlpDispatches === undefined ? {} : { fusedMlpDispatches: entry.direct.fusedMlpDispatches }),
        ...(entry.direct.fusedFfnRounding === undefined ? {} : { fusedFfnRounding: entry.direct.fusedFfnRounding }),
        ...(entry.direct.fusedFfnDispatches === undefined ? {} : { fusedFfnDispatches: entry.direct.fusedFfnDispatches }),
        ...(entry.direct.fusedDecoderLayerRounding === undefined ? {} : { fusedDecoderLayerRounding: entry.direct.fusedDecoderLayerRounding }),
        ...(entry.direct.fusedDecoderLayerDispatches === undefined ? {} : { fusedDecoderLayerDispatches: entry.direct.fusedDecoderLayerDispatches }),
        ...(entry.direct.fusedPleRounding === undefined ? {} : { fusedPleRounding: entry.direct.fusedPleRounding }),
        ...(entry.direct.fusedPlePreludeRounding === undefined ? {} : { fusedPlePreludeRounding: entry.direct.fusedPlePreludeRounding }),
        ...(entry.direct.fusedPleDispatches === undefined ? {} : { fusedPleDispatches: entry.direct.fusedPleDispatches }),
        ...(entry.direct.fusedPlePreludeDispatches === undefined ? {} : { fusedPlePreludeDispatches: entry.direct.fusedPlePreludeDispatches }),
        ...(entry.direct.finalHeadCompute === undefined ? {} : { finalHeadCompute: entry.direct.finalHeadCompute }),
        ...(entry.direct.nativeAttentionRounding === undefined ? {} : { nativeAttentionRounding: entry.direct.nativeAttentionRounding }),
        ...(entry.direct.nativeAttentionDispatches === undefined ? {} : { nativeAttentionDispatches: entry.direct.nativeAttentionDispatches }),
        ...(entry.direct.fusedAttentionRounding === undefined ? {} : { fusedAttentionRounding: entry.direct.fusedAttentionRounding }),
        ...(entry.direct.fusedAttentionDispatches === undefined ? {} : { fusedAttentionDispatches: entry.direct.fusedAttentionDispatches }),
        ...(entry.direct.referenceSeconds === undefined ? {} : { referenceSeconds: entry.direct.referenceSeconds }),
        ...(entry.direct.batchSeconds === undefined ? {} : { batchSeconds: entry.direct.batchSeconds }),
        ...(entry.direct.fusedMlpSeconds === undefined ? {} : { fusedMlpSeconds: entry.direct.fusedMlpSeconds }),
        ...(entry.direct.fusedFfnSeconds === undefined ? {} : { fusedFfnSeconds: entry.direct.fusedFfnSeconds }),
        ...(entry.direct.fusedDecoderLayerSeconds === undefined ? {} : { fusedDecoderLayerSeconds: entry.direct.fusedDecoderLayerSeconds }),
        ...(entry.direct.fusedPleSeconds === undefined ? {} : { fusedPleSeconds: entry.direct.fusedPleSeconds }),
        ...(entry.direct.fusedPlePreludeSeconds === undefined ? {} : { fusedPlePreludeSeconds: entry.direct.fusedPlePreludeSeconds }),
        ...(entry.direct.nativeAttentionSeconds === undefined ? {} : { nativeAttentionSeconds: entry.direct.nativeAttentionSeconds }),
        ...(entry.direct.fusedAttentionSeconds === undefined ? {} : { fusedAttentionSeconds: entry.direct.fusedAttentionSeconds }),
        forwardSeconds: entry.direct.steps.map((step) => step.forwardSeconds) },
      steps: entry.steps.map((step, index) => ({
        step: step.step, contextsEqualBeforeStep: step.contextsEqualBeforeStep,
        baselineToken: step.baselineToken, compatibilityToken: step.candidateToken, directToken: entry.direct.steps[index]?.tokenId,
        baselineTopLogits: step.baselineTopLogits, compatibilityTopLogits: step.candidateTopLogits, directTopLogits: entry.direct.steps[index]?.topLogits,
      })),
    })),
  };
}

export async function parseGemma4CalibrationCliOptions(argv: readonly string[]): Promise<Gemma4CalibrationCliOptions> {
  const pairs: Array<[string, string]> = [];
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index], value = argv[index + 1];
    if (!flag?.startsWith("--") || value === undefined || pairs.some(([existing]) => existing === flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
    pairs.push([flag, value]);
  }
  const value = (flag: string): string | undefined => pairs.find(([candidate]) => candidate === flag)?.[1];
  const output = value("--output"); if (!output) throw new Error("--output é obrigatório.");
  const maxNewTokens = Number(value("--tokens") ?? "2"), requestThreads = Number(value("--request-threads") ?? "1");
  if (!Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 64) throw new Error("--tokens deve estar entre 1 e 64.");
  if (!Number.isSafeInteger(requestThreads) || requestThreads < 0 || requestThreads > 256) throw new Error("--request-threads deve estar entre 0 e 256.");
  const precision = value("--precision") ?? "f32"; if (precision !== "f32" && precision !== "f64") throw new Error("--precision deve ser f32 ou f64.");
  const roundingPolicy = value("--rounding-policy") ?? "none"; if (roundingPolicy !== "none" && roundingPolicy !== "layer-bf16" && roundingPolicy !== "operation-bf16") throw new Error("--rounding-policy inválido.");
  const promptsPath = value("--prompts-json");
  const prompts = promptsPath ? JSON.parse(await readFile(resolve(promptsPath), "utf8")) as unknown : [...GEMMA4_CALIBRATION_PROMPTS];
  if (!Array.isArray(prompts) || prompts.length === 0 || prompts.some((prompt) => typeof prompt !== "string" || prompt.length === 0 || prompt.length > 16_384)) throw new Error("--prompts-json deve conter um array não vazio de prompts válidos.");
  const calibrationFlags = new Set(["--output", "--tokens", "--request-threads", "--precision", "--rounding-policy", "--prompts-json"]);
  const runnerArguments = pairs.filter(([flag]) => !calibrationFlags.has(flag)).flat();
  return { prompts: prompts as string[], output: resolve(output), maxNewTokens, requestThreads, precision, roundingPolicy, runner: parseGemma4RealServerOptions(runnerArguments) };
}

async function waitUntilReady(endpoint: string): Promise<Record<string, unknown>> {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    const response = await fetch(`${endpoint}/api/status`);
    const status = await response.json() as Record<string, unknown>;
    if (status.ready === true) {
      const direct = status.direct as Record<string, unknown> | undefined;
      if (direct?.enabled !== true) throw new Error("Calibração três-vias requer o backend compilado direto habilitado.");
      return status;
    }
    await new Promise((accept) => setTimeout(accept, 100));
  }
  throw new Error("Workers persistentes não ficaram prontos em 120 segundos.");
}

function assertThreeWayCase(value: ThreeWayCase | { error?: string }, prompt: string): asserts value is ThreeWayCase {
  const candidate = value as Partial<ThreeWayCase>;
  if (!Array.isArray(candidate.baselineGeneratedTokenIds) || !Array.isArray(candidate.candidateGeneratedTokenIds) || !candidate.direct || !Array.isArray(candidate.direct.generatedTokenIds)) throw new Error(`Resposta três-vias inválida para ${JSON.stringify(prompt)}.`);
}

function equalTokenSteps(reference: readonly number[], candidate: readonly number[]): number {
  return reference.reduce((equal, token, index) => equal + Number(token === candidate[index]), 0);
}
