import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gemma4RealCompareHtml } from "./gemma4-real-compare-ui.js";

export interface Gemma4RealComparisonRequest { prompt: string; maxNewTokens: number; threads?: number; precision?: "f32" | "f64"; roundingPolicy?: "none" | "layer-bf16" | "operation-bf16"; sessionId?: number; continueSession?: boolean; conversationMode?: "raw" | "chat" }
export interface Gemma4RealComparisonRunnerOptions {
  source: string; python: string; helper: string;
  literalArtifact?: string; binaryPool?: string; directWorker?: string; directLinearHelper?: string; directMlxHelper?: string; directLinearBackend?: "pytorch" | "mlx"; directFusedMlp?: "off" | "bf16" | "real" | "native-bf16"; directFusedFfn?: "off" | "native-bf16"; directFusedDecoderLayer?: "off" | "native-bf16"; directFusedDecoderStack?: "off" | "native-bf16" | "native-bf16-ple"; directFusedPle?: "off" | "bf16" | "real"; directFusedPlePrelude?: "off" | "bf16" | "real"; directFusedTokenForward?: "off" | "bf16"; directResidentGeneration?: "off" | "on"; directFinalHead?: "f32" | "native-bf16" | "native-bf16-stream" | "native-bf16-whole"; directNativeAttention?: "off" | "bf16" | "real"; directFusedAttention?: "off" | "bf16" | "real" | "native-bf16"; directThreads?: number; directMaxReadMiB?: number; directFinalHeadReadMiB?: number; directVerificationMargin?: number;
}

const GEMMA4_CHAT_TEMPLATE = "llm-inner-gemma4-it-text-turn-v1";
interface SessionInput { tokenIds: number[]; mode: "raw" | "chat" }

export async function runGemma4RealComparison(request: Gemma4RealComparisonRequest, options: Gemma4RealComparisonRunnerOptions): Promise<unknown> {
  validateRequest(request);
  const directory = await mkdtemp(join(tmpdir(), "gemma4-real-compare-"));
  const output = join(directory, "report.json");
  try {
    await runProcess(options.python, [options.helper, "--source", options.source, "--prompt", request.prompt, "--max-new-tokens", String(request.maxNewTokens), "--threads", String(request.threads ?? 0), "--precision", request.precision ?? "f64", "--rounding-policy", request.roundingPolicy ?? "none", "--output", output]);
    return JSON.parse(await readFile(output, "utf8")) as unknown;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

export function createGemma4RealComparisonServer(options: Gemma4RealComparisonRunnerOptions) {
  const directBackend = options.directLinearBackend ?? "mlx";
  if (options.directVerificationMargin !== undefined && (!Number.isFinite(options.directVerificationMargin) || options.directVerificationMargin < 0)) throw new Error("directVerificationMargin deve ser finita e não negativa.");
  if (options.directVerificationMargin !== undefined && directBackend !== "mlx") throw new Error("directVerificationMargin requer backend MLX.");
  const checkpointChatTemplateDeclared = declaresChatTemplate(options.source);
  const directTokenForward = options.directFusedTokenForward ?? (directBackend === "mlx" && options.directFusedDecoderStack !== "off" ? "bf16" : "off");
  const directResidentGeneration = options.directResidentGeneration ?? (directTokenForward === "bf16" ? "on" : "off");
  const direct = options.literalArtifact && options.binaryPool
    ? new PersistentJsonlWorker(process.execPath, directWorkerArguments(options, directBackend), "Gemma 4 literal direto")
    : undefined;
  const verificationEnabled = direct !== undefined && directBackend === "mlx" && options.directVerificationMargin !== undefined;
  let verification: PersistentJsonlWorker | undefined;
  let worker: Gemma4PersistentComparisonWorker | undefined, initializationError: Error | undefined, closed = false;
  const sessionInputs = new Map<number, SessionInput>();
  let directWarmupSeconds: number | undefined;
  const initialize = (async () => {
    try {
      if (direct && directBackend === "mlx") directWarmupSeconds = await warmupDirectWorker(direct, directResidentGeneration === "on" ? 2 : 1);
      if (verificationEnabled) {
        verification = new PersistentJsonlWorker(process.execPath, directWorkerArguments(options, "pytorch"), "Gemma 4 literal verificador PyTorch");
      }
      if (!closed) worker = new Gemma4PersistentComparisonWorker(options);
    } catch (error) {
      initializationError = error instanceof Error ? error : new Error(String(error));
      throw initializationError;
    }
  })();
  void initialize.catch(() => undefined);
  const server = createServer(async (request, response) => {
    try {
      if (request.method === "GET" && request.url === "/") return send(response, 200, "text/html; charset=utf-8", gemma4RealCompareHtml);
      if (request.method === "GET" && request.url === "/api/status") return json(response, 200, { ready: worker?.ready === true && (!direct || direct.ready), source: options.source, runtime: "persistent-jsonl", chatTemplate: GEMMA4_CHAT_TEMPLATE, checkpointChatTemplateDeclared, initializationSeconds: worker?.initializationSeconds, ...(initializationError ? { error: initializationError.message } : {}), direct: direct ? { enabled: true, ...direct.readyMetadata, ready: direct.ready, initializationSeconds: direct.initializationSeconds, warmupSeconds: directWarmupSeconds, warmupComplete: directWarmupSeconds !== undefined, verification: verification ? { enabled: true, backend: "pytorch", marginThreshold: options.directVerificationMargin, ...verification.readyMetadata, ready: verification.ready, initializationSeconds: verification.initializationSeconds, warmupComplete: false } : { enabled: false } } : { enabled: false } });
      if (request.method === "POST" && request.url === "/api/compare") {
        await initialize;
        if (!worker) throw new Error("Comparador original não foi inicializado.");
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        const inputIds = await resolveComparisonInput(worker, body, sessionInputs);
        const report = await worker.compareTokens(body, inputIds) as ComparisonReport;
        if (!direct) return json(response, 200, { ...report, prompt: body.prompt, conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null });
        const directReport = await executeSelectedDirect(direct, verification, options.directVerificationMargin, { inputIds, maxNewTokens: body.maxNewTokens, ...(body.sessionId === undefined ? {} : { sessionId: body.sessionId }) });
        attachDirectLogitAgreement(report, directReport);
        const [generated, full] = await Promise.all([worker.decode(directReport.generatedTokenIds), worker.decode(directReport.fullTokenIds)]);
        directReport.generatedText = generated.text; directReport.fullText = full.text;
        directReport.tokensEqualBaseline = arraysEqual(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
        directReport.firstDivergentStep = firstDivergence(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
        if (body.sessionId !== undefined) rememberSession(sessionInputs, body.sessionId, directReport.fullTokenIds, body.conversationMode ?? "raw");
        return json(response, 200, { ...report, prompt: body.prompt, conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null, direct: directReport });
      }
      if (request.method === "POST" && request.url === "/api/compare-stream") {
        await initialize;
        if (!worker || !direct) throw new Error("Streaming comparativo requer os executores original e direto.");
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        const inputIds = await resolveComparisonInput(worker, body, sessionInputs);
        let clientDisconnected = false; response.once("close", () => { if (!response.writableEnded) clientDisconnected = true; });
        response.writeHead(200, { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store", "x-content-type-options": "nosniff" });
        try {
          const referencePromise = worker.compareTokens(body, inputIds) as Promise<ComparisonReport>;
          const directPromise = executeSelectedDirect(direct, verification, options.directVerificationMargin, { inputIds, maxNewTokens: body.maxNewTokens, stream: true, ...(body.sessionId === undefined ? {} : { sessionId: body.sessionId }) }, (event) => writeNdjson(response, { type: "direct-token", provisional: verification !== undefined, event }), (assessment) => writeNdjson(response, { type: "direct-fallback", ...assessment }), () => !clientDisconnected);
          const [report, directReport] = await Promise.all([referencePromise, directPromise]);
          if (clientDisconnected || response.destroyed) return;
          attachDirectLogitAgreement(report, directReport);
          const [generated, full] = await Promise.all([worker.decode(directReport.generatedTokenIds), worker.decode(directReport.fullTokenIds)]);
          directReport.generatedText = generated.text; directReport.fullText = full.text;
          directReport.tokensEqualBaseline = arraysEqual(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
          directReport.firstDivergentStep = firstDivergence(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
          if (body.sessionId !== undefined) rememberSession(sessionInputs, body.sessionId, directReport.fullTokenIds, body.conversationMode ?? "raw");
          writeNdjson(response, { type: "result", data: { ...report, prompt: body.prompt, conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null, direct: directReport } }); response.end(); return;
        } catch (error) {
          writeNdjson(response, { type: "error", error: error instanceof Error ? error.message : String(error) }); response.end(); return;
        }
      }
      return json(response, 404, { error: "Rota não encontrada." });
    } catch (error) {
      return json(response, 400, { error: error instanceof Error ? error.message : String(error) });
    }
  });
  server.once("close", () => { closed = true; worker?.close(); direct?.close(); verification?.close(); void initialize.catch(() => undefined); });
  return server;
}

function directWorkerArguments(options: Gemma4RealComparisonRunnerOptions, backend: "pytorch" | "mlx"): string[] {
  if (!options.literalArtifact || !options.binaryPool) throw new Error("Worker direto requer artefato literal e pool binário.");
  const primary = backend === (options.directLinearBackend ?? "mlx");
  const decoderStack = primary ? options.directFusedDecoderStack ?? "native-bf16" : "native-bf16";
  const tokenForward = backend === "mlx" ? (primary ? options.directFusedTokenForward ?? (decoderStack !== "off" ? "bf16" : "off") : "bf16") : "off";
  const residentGeneration = backend === "mlx" ? (primary ? options.directResidentGeneration ?? (tokenForward === "bf16" ? "on" : "off") : "on") : "off";
  return [options.directWorker ?? resolve("dist/src/gemma4-paged-runtime-worker-cli.js"), "--artifact", options.literalArtifact, "--binary-pool", options.binaryPool, "--python", options.python, "--linear-helper", options.directLinearHelper ?? resolve("scripts/gemma4-paged-linear-worker.py"), "--mlx-helper", options.directMlxHelper ?? resolve("scripts/gemma4-mlx-linear-worker.py"), "--linear-backend", backend, "--fused-mlp", primary ? options.directFusedMlp ?? (backend === "pytorch" ? "native-bf16" : "real") : "native-bf16", "--fused-ffn", primary ? options.directFusedFfn ?? (backend === "pytorch" ? "native-bf16" : "off") : "native-bf16", "--fused-decoder-layer", primary ? options.directFusedDecoderLayer ?? (backend === "pytorch" ? "native-bf16" : "off") : "native-bf16", "--fused-decoder-stack", decoderStack, "--fused-ple", primary ? options.directFusedPle ?? (backend === "pytorch" ? "bf16" : "off") : "bf16", "--fused-ple-prelude", primary ? options.directFusedPlePrelude ?? (backend === "mlx" ? "bf16" : "off") : "off", "--fused-token-forward", tokenForward, "--resident-generation", residentGeneration, "--final-head", primary ? options.directFinalHead ?? (backend === "pytorch" ? "native-bf16-stream" : "native-bf16-whole") : "native-bf16-stream", "--native-attention", primary ? options.directNativeAttention ?? (backend === "pytorch" ? "real" : "off") : "real", "--fused-attention", primary ? options.directFusedAttention ?? (backend === "pytorch" ? "native-bf16" : "off") : "native-bf16", "--threads", String(options.directThreads ?? 10), "--max-read-mib", String(options.directMaxReadMiB ?? 16), "--final-head-read-mib", String(primary ? options.directFinalHeadReadMiB ?? (backend === "pytorch" ? 32 : options.directMaxReadMiB ?? 16) : 32)];
}

async function warmupDirectWorker(worker: PersistentJsonlWorker, maxNewTokens: number): Promise<number> {
  const started = performance.now();
  await worker.send({ inputIds: [2], maxNewTokens });
  return (performance.now() - started) / 1000;
}

interface DirectMarginAssessment { trigger: boolean; reason: "margin-at-or-below-threshold" | "margin-unavailable"; minimumMargin: number | null; marginThreshold: number; sensitiveSteps: number[] }

export function assessDirectVerification(report: DirectReport, marginThreshold: number): DirectMarginAssessment {
  if (!Array.isArray(report.steps) || report.steps.length !== report.generatedTokenIds.length) return { trigger: true, reason: "margin-unavailable", minimumMargin: null, marginThreshold, sensitiveSteps: [] };
  const margins = report.steps.map((step) => {
    if (!step || typeof step !== "object") return undefined;
    const top = normalizeTopLogits((step as { topLogits?: unknown }).topLogits);
    return top.length >= 2 ? top[0]!.value - top[1]!.value : undefined;
  });
  if (margins.some((margin) => margin === undefined || !Number.isFinite(margin) || margin! < 0)) return { trigger: true, reason: "margin-unavailable", minimumMargin: null, marginThreshold, sensitiveSteps: [] };
  const minimumMargin = Math.min(...margins as number[]);
  const sensitiveSteps = (margins as number[]).flatMap((margin, step) => margin <= marginThreshold ? [step] : []);
  return { trigger: sensitiveSteps.length > 0, reason: "margin-at-or-below-threshold", minimumMargin, marginThreshold, sensitiveSteps };
}

async function executeSelectedDirect(primary: PersistentJsonlWorker, verification: PersistentJsonlWorker | undefined, marginThreshold: number | undefined, payload: { inputIds: number[]; maxNewTokens: number; sessionId?: number; stream?: boolean }, onEvent?: (event: unknown) => void, onFallback?: (assessment: DirectMarginAssessment) => void, shouldVerify: () => boolean = () => true): Promise<DirectReport> {
  const started = performance.now();
  const fast = await primary.send(payload, onEvent) as DirectReport;
  if (!verification || marginThreshold === undefined) return fast;
  const assessment = assessDirectVerification(fast, marginThreshold);
  const fastPathSeconds = (performance.now() - started) / 1000;
  if (!assessment.trigger) return Object.assign(fast, { selectionPolicy: "margin-verified-pytorch-v1", selectedBackend: "mlx", fallbackTriggered: false, fastPathMinimumMargin: assessment.minimumMargin, verificationMarginThreshold: marginThreshold, fastPathSeconds });
  if (!shouldVerify()) return fast;
  onFallback?.(assessment);
  const selectiveRequest = assessment.reason === "margin-at-or-below-threshold" && assessment.sensitiveSteps.length > 0 && typeof fast.terminalLogitsSha256 === "string" && /^[0-9a-f]{64}$/.test(fast.terminalLogitsSha256)
    ? { verificationFastPath: { generatedTokenIds: fast.generatedTokenIds, sensitiveSteps: assessment.sensitiveSteps, terminalLogitsSha256: fast.terminalLogitsSha256 } }
    : {};
  const verified = await verification.send({ inputIds: payload.inputIds, maxNewTokens: payload.maxNewTokens, ...selectiveRequest, ...(payload.sessionId === undefined ? {} : { sessionId: payload.sessionId }) }) as DirectReport;
  if (verified.selectiveVerification === true && Array.isArray(verified.steps) && Array.isArray(fast.steps)) {
    verified.steps = verified.steps.map((step, index) => step?.verificationSkipped === true && fast.steps![index]
      ? { ...fast.steps![index], ...(step.forwardSeconds === undefined ? {} : { forwardSeconds: step.forwardSeconds }), verificationSkipped: true }
      : step);
  }
  const hybridSeconds = (performance.now() - started) / 1000;
  return Object.assign(verified, {
    selectionPolicy: "margin-verified-pytorch-v1", selectedBackend: "pytorch", fallbackTriggered: true, fallbackReason: assessment.reason,
    fastPathMinimumMargin: assessment.minimumMargin, verificationMarginThreshold: marginThreshold, fastPathSeconds,
    verificationSeconds: Math.max(0, hybridSeconds - fastPathSeconds), workerVerificationSeconds: verified.elapsedSeconds,
    hybridSeconds, elapsedSeconds: hybridSeconds, tokensPerSecond: verified.generatedTokenIds.length / hybridSeconds,
    fastPath: fast,
  });
}

class Gemma4PersistentComparisonWorker {
  readonly transport: PersistentJsonlWorker;
  constructor(options: Gemma4RealComparisonRunnerOptions) { this.transport = new PersistentJsonlWorker(options.python, [options.helper, "--source", options.source, "--serve-jsonl"], "Gemma 4 Transformers"); }
  get ready(): boolean { return this.transport.ready; }
  get initializationSeconds(): number | undefined { return this.transport.initializationSeconds; }
  compare(request: Gemma4RealComparisonRequest): Promise<unknown> { return this.transport.send(request); }
  compareTokens(request: Gemma4RealComparisonRequest, inputIds: readonly number[]): Promise<unknown> { return this.transport.send({ ...request, prompt: undefined, inputIds: inputIds.join(",") }); }
  encode(text: string, addSpecialTokens: boolean): Promise<{ tokenIds: number[] }> { return this.transport.send({ mode: "encode", text, addSpecialTokens }) as Promise<{ tokenIds: number[] }>; }
  decode(tokenIds: number[]): Promise<{ text: string }> { return this.transport.send({ mode: "decode", tokenIds }) as Promise<{ text: string }>; }
  close(): void { this.transport.close(); }
}

class PersistentJsonlWorker {
  readonly child: ChildProcessWithoutNullStreams;
  readonly pending = new Map<number, { accept(value: unknown): void; reject(error: Error): void; onEvent?: (event: unknown) => void }>();
  ready = false; initializationSeconds?: number; readyMetadata: Record<string, unknown> = {};
  #nextId = 1; #stdout = ""; #readyAccept!: () => void; #readyReject!: (error: Error) => void; readonly #readyPromise: Promise<void>;
  constructor(command: string, arguments_: string[], readonly label: string) {
    this.#readyPromise = new Promise<void>((accept, reject) => { this.#readyAccept = accept; this.#readyReject = reject; });
    void this.#readyPromise.catch(() => undefined);
    this.child = spawn(command, arguments_, { stdio: ["pipe", "pipe", "pipe"] });
    this.child.stdout.setEncoding("utf8"); this.child.stdout.on("data", (chunk: string) => this.#consume(chunk));
    const errors: Buffer[] = []; let errorBytes = 0; this.child.stderr.on("data", (chunk: Buffer) => { if (errorBytes < 1024 * 1024) { errors.push(chunk); errorBytes += chunk.length; } });
    this.child.once("error", (error) => this.#fail(error));
    this.child.once("close", (code) => this.#fail(new Error(`${label} encerrou com código ${code}: ${Buffer.concat(errors).toString("utf8").trim()}`)));
  }
  async send(payload: object, onEvent?: (event: unknown) => void): Promise<unknown> { await this.#readyPromise; const id = this.#nextId++; const result = new Promise<unknown>((accept, reject) => this.pending.set(id, { accept, reject, ...(onEvent ? { onEvent } : {}) })); this.child.stdin.write(`${JSON.stringify({ id, ...payload })}\n`); return result; }
  close(): void { if (!this.child.killed) this.child.kill("SIGTERM"); }
  #consume(chunk: string): void {
    this.#stdout += chunk; if (this.#stdout.length > 64 * 1024 * 1024) return this.#fail(new Error(`${this.label} excedeu 64 MiB sem delimitar resposta.`));
    while (true) { const newline = this.#stdout.indexOf("\n"); if (newline < 0) break; const line = this.#stdout.slice(0, newline); this.#stdout = this.#stdout.slice(newline + 1); if (!line) continue;
      let message: { ready?: boolean; initializationSeconds?: number; id?: number; report?: unknown; event?: unknown; error?: string }; try { message = JSON.parse(line) as typeof message; } catch (error) { return this.#fail(new Error(`${this.label} retornou JSON inválido: ${error instanceof Error ? error.message : String(error)}`)); }
      if (message.ready) { this.ready = true; this.readyMetadata = { ...message }; if (message.initializationSeconds !== undefined) this.initializationSeconds = message.initializationSeconds; this.#readyAccept(); continue; }
      if (!Number.isSafeInteger(message.id)) return this.#fail(new Error(`${this.label} respondeu sem id válido.`)); const pending = this.pending.get(message.id!); if (!pending) return this.#fail(new Error(`${this.label} respondeu id desconhecido ${message.id}.`)); if (message.event !== undefined) { pending.onEvent?.(message.event); continue; } this.pending.delete(message.id!); if (message.error) pending.reject(new Error(message.error)); else pending.accept(message.report);
    }
  }
  #fail(error: Error): void { this.ready = false; this.#readyReject(error); for (const pending of this.pending.values()) pending.reject(error); this.pending.clear(); }
}

interface ComparisonReport { inputIds?: number[][]; baselineGeneratedTokenIds: number[]; steps?: Array<{ baselineToken?: number; baselineTopLogits?: unknown }> }
interface DirectReport { generatedTokenIds: number[]; fullTokenIds: number[]; generatedText?: string; fullText?: string; tokensEqualBaseline?: boolean; firstDivergentStep?: number | null; terminalLogitsSha256?: string; selectiveVerification?: boolean; steps?: Array<{ forwardSeconds?: number; verificationSkipped?: boolean; [key: string]: unknown }>; [key: string]: unknown }
export interface DirectLogitStepAgreement { step: number; contextsEqualBeforeStep: boolean; baselineArgmaxToken: number | null; directArgmaxToken: number | null; baselineArgmaxLogitAbsError: number | null; greedyMarginAbsError: number | null; topK: number; topKOverlapCount: number; topKOverlapRate: number | null; topKCommonLogitMaxAbsError: number | null }
export interface DirectLogitAgreement { reportedSteps: number; measuredSteps: number; rootDivergences: number; postDivergenceSteps: number; selectedLogitMeasuredSteps: number; meanBaselineArgmaxLogitAbsError: number | null; maxBaselineArgmaxLogitAbsError: number | null; marginMeasuredSteps: number; meanGreedyMarginAbsError: number | null; maxGreedyMarginAbsError: number | null; meanTopKOverlapRate: number | null; maxTopKCommonLogitAbsError: number | null; steps: DirectLogitStepAgreement[] }
function arraysEqual(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function firstDivergence(left: readonly number[], right: readonly number[]): number | null { const length = Math.max(left.length, right.length); for (let index = 0; index < length; index += 1) if (left[index] !== right[index]) return index; return null; }
function declaresChatTemplate(source: string): boolean { try { const parsed = JSON.parse(readFileSync(join(source, "tokenizer_config.json"), "utf8")) as { chat_template?: unknown }; return typeof parsed.chat_template === "string" && parsed.chat_template.length > 0; } catch { return false; } }

export function computeDirectLogitAgreement(baselineSteps: ReadonlyArray<{ baselineToken?: number; baselineTopLogits?: unknown }>, directSteps: ReadonlyArray<{ tokenId?: number; topLogits?: unknown }>): DirectLogitAgreement {
  const steps: DirectLogitStepAgreement[] = [];
  let contextsEqual = true;
  for (let step = 0; step < Math.min(baselineSteps.length, directSteps.length); step += 1) {
    const baseline = normalizeTopLogits(baselineSteps[step]!.baselineTopLogits), direct = normalizeTopLogits(directSteps[step]!.topLogits);
    if (baseline.length === 0 || direct.length === 0) continue;
    const baselineMap = new Map(baseline.map((entry) => [entry.tokenId, entry.value])), directMap = new Map(direct.map((entry) => [entry.tokenId, entry.value]));
    const topK = Math.min(baseline.length, direct.length), baselineTopK = baseline.slice(0, topK), directTopK = direct.slice(0, topK);
    const directTopIds = new Set(directTopK.map((entry) => entry.tokenId)), common = baselineTopK.filter((entry) => directTopIds.has(entry.tokenId));
    const commonErrors = common.map((entry) => Math.abs(entry.value - directMap.get(entry.tokenId)!));
    const baselineArgmaxToken = Number.isSafeInteger(baselineSteps[step]!.baselineToken) ? baselineSteps[step]!.baselineToken! : baseline[0]!.tokenId;
    const directArgmaxToken = Number.isSafeInteger(directSteps[step]!.tokenId) ? directSteps[step]!.tokenId! : direct[0]!.tokenId;
    const baselineArgmaxLogit = baselineMap.get(baselineArgmaxToken), directArgmaxLogit = directMap.get(baselineArgmaxToken);
    const baselineMargin = baseline.length >= 2 ? baseline[0]!.value - baseline[1]!.value : undefined, directMargin = direct.length >= 2 ? direct[0]!.value - direct[1]!.value : undefined;
    steps.push({
      step, contextsEqualBeforeStep: contextsEqual, baselineArgmaxToken, directArgmaxToken,
      baselineArgmaxLogitAbsError: !contextsEqual || baselineArgmaxLogit === undefined || directArgmaxLogit === undefined ? null : Math.abs(baselineArgmaxLogit - directArgmaxLogit),
      greedyMarginAbsError: !contextsEqual || baselineMargin === undefined || directMargin === undefined ? null : Math.abs(baselineMargin - directMargin),
      topK, topKOverlapCount: contextsEqual ? common.length : 0, topKOverlapRate: !contextsEqual || topK === 0 ? null : common.length / topK,
      topKCommonLogitMaxAbsError: !contextsEqual || commonErrors.length === 0 ? null : Math.max(...commonErrors),
    });
    if (baselineArgmaxToken !== directArgmaxToken) contextsEqual = false;
  }
  const comparableSteps = steps.filter((step) => step.contextsEqualBeforeStep);
  const selectedErrors = comparableSteps.flatMap((step) => step.baselineArgmaxLogitAbsError === null ? [] : [step.baselineArgmaxLogitAbsError]);
  const marginErrors = comparableSteps.flatMap((step) => step.greedyMarginAbsError === null ? [] : [step.greedyMarginAbsError]);
  const overlapRates = comparableSteps.flatMap((step) => step.topKOverlapRate === null ? [] : [step.topKOverlapRate]);
  const commonErrors = comparableSteps.flatMap((step) => step.topKCommonLogitMaxAbsError === null ? [] : [step.topKCommonLogitMaxAbsError]);
  return {
    reportedSteps: steps.length, measuredSteps: comparableSteps.length, rootDivergences: comparableSteps.filter((step) => step.baselineArgmaxToken !== step.directArgmaxToken).length, postDivergenceSteps: steps.length - comparableSteps.length, selectedLogitMeasuredSteps: selectedErrors.length,
    meanBaselineArgmaxLogitAbsError: meanOrNull(selectedErrors), maxBaselineArgmaxLogitAbsError: maxOrNull(selectedErrors),
    marginMeasuredSteps: marginErrors.length, meanGreedyMarginAbsError: meanOrNull(marginErrors), maxGreedyMarginAbsError: maxOrNull(marginErrors),
    meanTopKOverlapRate: meanOrNull(overlapRates), maxTopKCommonLogitAbsError: maxOrNull(commonErrors), steps,
  };
}

function attachDirectLogitAgreement(report: ComparisonReport, direct: DirectReport): void {
  if (!Array.isArray(report.steps) || !Array.isArray(direct.steps)) return;
  direct.logitAgreement = computeDirectLogitAgreement(report.steps, direct.steps as Array<{ tokenId?: number; topLogits?: unknown }>);
}
function normalizeTopLogits(value: unknown): Array<{ tokenId: number; value: number }> {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const candidate = entry as { tokenId?: unknown; logit?: unknown; value?: unknown }, numeric = typeof candidate.logit === "number" ? candidate.logit : candidate.value;
    return Number.isSafeInteger(candidate.tokenId) && typeof numeric === "number" && Number.isFinite(numeric) ? [{ tokenId: candidate.tokenId as number, value: numeric }] : [];
  });
}
function meanOrNull(values: readonly number[]): number | null { return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length; }
function maxOrNull(values: readonly number[]): number | null { return values.length === 0 ? null : Math.max(...values); }

async function resolveComparisonInput(worker: Gemma4PersistentComparisonWorker, request: Gemma4RealComparisonRequest, sessions: Map<number, SessionInput>): Promise<number[]> {
  const mode = request.conversationMode ?? "raw";
  if (request.continueSession) {
    const session = sessions.get(request.sessionId!); if (!session) throw new Error("Sessão de chat inexistente ou expirada.");
    if (session.mode !== mode) throw new Error("conversationMode não pode mudar durante uma sessão.");
    const suffix = await worker.encode(mode === "chat" ? formatGemma4ChatTurn(request.prompt, false) : request.prompt, false); validateEncodedIds(suffix.tokenIds);
    return [...session.tokenIds, ...suffix.tokenIds];
  }
  if (request.sessionId !== undefined) sessions.delete(request.sessionId);
  const encoded = await worker.encode(mode === "chat" ? formatGemma4ChatTurn(request.prompt, true) : request.prompt, true); validateEncodedIds(encoded.tokenIds); return [...encoded.tokenIds];
}
function formatGemma4ChatTurn(prompt: string, first: boolean): string { return `${first ? "" : "<turn|>\n"}<|turn>user\n${prompt.trim()}<turn|>\n<|turn>model\n`; }
function validateEncodedIds(value: unknown): asserts value is number[] { if (!Array.isArray(value) || value.length < 1 || value.some((token) => !Number.isSafeInteger(token) || token < 0)) throw new Error("Tokenizer retornou inputIds inválidos."); }
function rememberSession(sessions: Map<number, SessionInput>, sessionId: number, fullTokenIds: readonly number[], mode: "raw" | "chat" = "raw"): void {
  validateEncodedIds(fullTokenIds); sessions.delete(sessionId); sessions.set(sessionId, { tokenIds: [...fullTokenIds], mode });
  while (sessions.size > 32) sessions.delete(sessions.keys().next().value!);
}

function validateRequest(request: Gemma4RealComparisonRequest): void {
  if (typeof request.prompt !== "string" || request.prompt.length === 0 || request.prompt.length > 16_384) throw new Error("prompt deve conter entre 1 e 16.384 caracteres.");
  if (!Number.isSafeInteger(request.maxNewTokens) || request.maxNewTokens < 1 || request.maxNewTokens > 64) throw new Error("maxNewTokens deve estar entre 1 e 64.");
  if (request.threads !== undefined && (!Number.isSafeInteger(request.threads) || request.threads < 0 || request.threads > 256)) throw new Error("threads deve estar entre 0 e 256.");
  if (request.precision !== undefined && request.precision !== "f32" && request.precision !== "f64") throw new Error("precision deve ser f32 ou f64.");
  if (request.roundingPolicy !== undefined && request.roundingPolicy !== "none" && request.roundingPolicy !== "layer-bf16" && request.roundingPolicy !== "operation-bf16") throw new Error("roundingPolicy deve ser none, layer-bf16 ou operation-bf16.");
  if (request.sessionId !== undefined && (!Number.isSafeInteger(request.sessionId) || request.sessionId < 1 || request.sessionId > 0xffff_ffff)) throw new Error("sessionId deve estar entre 1 e 4.294.967.295.");
  if (request.continueSession !== undefined && typeof request.continueSession !== "boolean") throw new Error("continueSession deve ser booleano.");
  if (request.continueSession && request.sessionId === undefined) throw new Error("continueSession requer sessionId.");
  if (request.conversationMode !== undefined && request.conversationMode !== "raw" && request.conversationMode !== "chat") throw new Error("conversationMode deve ser raw ou chat.");
}

async function readBody(request: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = []; let bytes = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk); bytes += buffer.length;
    if (bytes > 1024 * 1024) throw new Error("Corpo HTTP excede 1 MiB.");
    chunks.push(buffer);
  }
  return Buffer.concat(chunks).toString("utf8");
}

async function runProcess(command: string, arguments_: string[]): Promise<void> {
  await new Promise<void>((accept, reject) => {
    const child = spawn(command, arguments_, { stdio: ["ignore", "ignore", "pipe"] });
    const errors: Buffer[] = []; let errorBytes = 0;
    child.stderr.on("data", (chunk: Buffer) => { if (errorBytes < 1024 * 1024) { errors.push(chunk); errorBytes += chunk.length; } });
    child.once("error", reject);
    child.once("close", (code) => code === 0 ? accept() : reject(new Error(`Comparador Gemma 4 terminou com código ${code}: ${Buffer.concat(errors).toString("utf8").trim()}`)));
  });
}

function send(response: ServerResponse, status: number, type: string, body: string): void {
  response.writeHead(status, { "content-type": type, "content-length": Buffer.byteLength(body), "cache-control": "no-store" }); response.end(body);
}
function json(response: ServerResponse, status: number, value: unknown): void { send(response, status, "application/json; charset=utf-8", `${JSON.stringify(value)}\n`); }
function writeNdjson(response: ServerResponse, value: unknown): void { if (!response.destroyed && !response.writableEnded) response.write(`${JSON.stringify(value)}\n`); }

export function parseGemma4RealServerOptions(arguments_: readonly string[]): Gemma4RealComparisonRunnerOptions & { port: number; host: string } {
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index], value = arguments_[index + 1];
    if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
    values.set(flag, value);
  }
  const known = new Set(["--source", "--python", "--helper", "--port", "--host", "--literal-artifact", "--binary-pool", "--direct-worker", "--direct-linear-helper", "--direct-mlx-helper", "--direct-linear-backend", "--direct-fused-mlp", "--direct-fused-ffn", "--direct-fused-decoder-layer", "--direct-fused-decoder-stack", "--direct-fused-ple", "--direct-fused-ple-prelude", "--direct-fused-token-forward", "--direct-resident-generation", "--direct-final-head", "--direct-final-head-read-mib", "--direct-native-attention", "--direct-fused-attention", "--direct-threads", "--direct-max-read-mib", "--direct-verification-margin"]); for (const key of values.keys()) if (!known.has(key)) throw new Error(`Flag desconhecida: ${key}.`);
  const port = Number(values.get("--port") ?? "8787"); if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) throw new Error("--port inválido.");
  const source = resolve(values.get("--source") ?? "gemma-4-E4B-dense"), inferredLiteral = join(source, "constants.literal.json"), bundledLiteral = resolve("artifacts/gemma4-compiled-global-runtime-bundle/constants.literal.json");
  const literalArtifact = values.get("--literal-artifact") ? resolve(values.get("--literal-artifact")!) : existsSync(inferredLiteral) ? inferredLiteral : !values.has("--source") && existsSync(bundledLiteral) ? bundledLiteral : undefined;
  const binaryPool = values.get("--binary-pool") ? resolve(values.get("--binary-pool")!) : literalArtifact && existsSync(join(source, "model.safetensors")) ? source : undefined;
  if ((literalArtifact === undefined) !== (binaryPool === undefined)) throw new Error("Backend direto requer --literal-artifact e --binary-pool juntos.");
  const directLinearBackend = values.get("--direct-linear-backend") ?? "mlx";
  if (directLinearBackend !== "pytorch" && directLinearBackend !== "mlx") throw new Error("--direct-linear-backend deve ser pytorch ou mlx.");
  const verificationMarginValue = values.get("--direct-verification-margin") ?? (directLinearBackend === "mlx" ? "0" : "off");
  const directVerificationMargin = verificationMarginValue === "off" ? undefined : Number(verificationMarginValue);
  if (directVerificationMargin !== undefined && (!Number.isFinite(directVerificationMargin) || directVerificationMargin < 0)) throw new Error("--direct-verification-margin deve ser off ou número finito não negativo.");
  if (directLinearBackend !== "mlx" && directVerificationMargin !== undefined) throw new Error("--direct-verification-margin requer backend mlx.");
  const directThreads = Number(values.get("--direct-threads") ?? "10"), directMaxReadMiB = Number(values.get("--direct-max-read-mib") ?? "16"), directFinalHeadReadMiB = Number(values.get("--direct-final-head-read-mib") ?? (directLinearBackend === "pytorch" ? "32" : String(directMaxReadMiB)));
  if (!Number.isSafeInteger(directThreads) || directThreads < 1 || directThreads > 256 || !Number.isSafeInteger(directMaxReadMiB) || directMaxReadMiB < 1 || directMaxReadMiB > 1024 || !Number.isSafeInteger(directFinalHeadReadMiB) || directFinalHeadReadMiB < 1 || directFinalHeadReadMiB > 1024) throw new Error("Configuração direta inválida.");
  const directFusedMlp = values.get("--direct-fused-mlp") ?? (directLinearBackend === "pytorch" ? "native-bf16" : "real");
  if (directFusedMlp !== "off" && directFusedMlp !== "bf16" && directFusedMlp !== "real" && directFusedMlp !== "native-bf16") throw new Error("--direct-fused-mlp deve ser off, bf16, real ou native-bf16.");
  if (directLinearBackend === "mlx" && directFusedMlp === "native-bf16") throw new Error("--direct-fused-mlp native-bf16 requer backend pytorch.");
  const directFusedFfn = values.get("--direct-fused-ffn") ?? (directLinearBackend === "pytorch" ? "native-bf16" : "off");
  if (directFusedFfn !== "off" && directFusedFfn !== "native-bf16") throw new Error("--direct-fused-ffn deve ser off ou native-bf16.");
  if (directLinearBackend === "mlx" && directFusedFfn !== "off") throw new Error("--direct-fused-ffn requer backend pytorch.");
  const directFusedDecoderLayer = values.get("--direct-fused-decoder-layer") ?? (directLinearBackend === "pytorch" ? "native-bf16" : "off");
  if (directFusedDecoderLayer !== "off" && directFusedDecoderLayer !== "native-bf16") throw new Error("--direct-fused-decoder-layer deve ser off ou native-bf16.");
  if (directLinearBackend === "mlx" && directFusedDecoderLayer !== "off") throw new Error("--direct-fused-decoder-layer requer backend pytorch.");
  const directFusedDecoderStack = values.get("--direct-fused-decoder-stack") ?? "native-bf16";
  if (directFusedDecoderStack !== "off" && directFusedDecoderStack !== "native-bf16" && directFusedDecoderStack !== "native-bf16-ple") throw new Error("--direct-fused-decoder-stack deve ser off, native-bf16 ou native-bf16-ple.");
  if (directLinearBackend === "mlx" && directFusedDecoderStack === "native-bf16-ple") throw new Error("--direct-fused-decoder-stack native-bf16-ple requer backend pytorch.");
  const directFusedPle = values.get("--direct-fused-ple") ?? (directLinearBackend === "pytorch" ? "bf16" : "off");
  if (directFusedPle !== "off" && directFusedPle !== "bf16" && directFusedPle !== "real") throw new Error("--direct-fused-ple deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directFusedPle !== "off") throw new Error("--direct-fused-ple requer backend pytorch.");
  const directFusedPlePrelude = values.get("--direct-fused-ple-prelude") ?? (directLinearBackend === "mlx" ? "bf16" : "off");
  if (directFusedPlePrelude !== "off" && directFusedPlePrelude !== "bf16" && directFusedPlePrelude !== "real") throw new Error("--direct-fused-ple-prelude deve ser off, bf16 ou real.");
  const directFusedTokenForward = values.get("--direct-fused-token-forward") ?? (directLinearBackend === "mlx" && directFusedDecoderStack !== "off" ? "bf16" : "off");
  if (directFusedTokenForward !== "off" && directFusedTokenForward !== "bf16") throw new Error("--direct-fused-token-forward deve ser off ou bf16.");
  if (directLinearBackend !== "mlx" && directFusedTokenForward !== "off") throw new Error("--direct-fused-token-forward requer backend mlx.");
  if (directFusedDecoderStack === "off" && directFusedTokenForward !== "off") throw new Error("--direct-fused-token-forward requer decoder stack habilitada.");
  const directResidentGeneration = values.get("--direct-resident-generation") ?? (directFusedTokenForward === "bf16" ? "on" : "off");
  if (directResidentGeneration !== "off" && directResidentGeneration !== "on") throw new Error("--direct-resident-generation deve ser off ou on.");
  if (directResidentGeneration === "on" && (directLinearBackend !== "mlx" || directFusedTokenForward !== "bf16")) throw new Error("--direct-resident-generation on requer token forward MLX bf16.");
  const directFinalHead = values.get("--direct-final-head") ?? (directLinearBackend === "pytorch" ? "native-bf16-stream" : "native-bf16-whole");
  if (directFinalHead !== "f32" && directFinalHead !== "native-bf16" && directFinalHead !== "native-bf16-stream" && directFinalHead !== "native-bf16-whole") throw new Error("--direct-final-head deve ser f32, native-bf16, native-bf16-stream ou native-bf16-whole.");
  if (directLinearBackend === "mlx" && directFinalHead !== "f32" && directFinalHead !== "native-bf16-whole") throw new Error("--direct-final-head no backend MLX deve ser f32 ou native-bf16-whole.");
  const directNativeAttention = values.get("--direct-native-attention") ?? (directLinearBackend === "pytorch" ? "real" : "off");
  if (directNativeAttention !== "off" && directNativeAttention !== "bf16" && directNativeAttention !== "real") throw new Error("--direct-native-attention deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directNativeAttention !== "off") throw new Error("--direct-native-attention requer backend pytorch.");
  const directFusedAttention = values.get("--direct-fused-attention") ?? (directLinearBackend === "pytorch" ? "native-bf16" : "off");
  if (directFusedAttention !== "off" && directFusedAttention !== "bf16" && directFusedAttention !== "real" && directFusedAttention !== "native-bf16") throw new Error("--direct-fused-attention deve ser off, bf16, real ou native-bf16.");
  if (directLinearBackend === "mlx" && directFusedAttention !== "off") throw new Error("--direct-fused-attention requer backend pytorch.");
  return {
    source, python: values.get("--python") ?? (existsSync("venv/bin/python") ? resolve("venv/bin/python") : "python3"),
    helper: resolve(values.get("--helper") ?? "scripts/gemma4-real-differential.py"), port, host: values.get("--host") ?? "127.0.0.1",
    ...(literalArtifact && binaryPool ? { literalArtifact, binaryPool, directWorker: resolve(values.get("--direct-worker") ?? "dist/src/gemma4-paged-runtime-worker-cli.js"), directLinearHelper: resolve(values.get("--direct-linear-helper") ?? "scripts/gemma4-paged-linear-worker.py"), directMlxHelper: resolve(values.get("--direct-mlx-helper") ?? "scripts/gemma4-mlx-linear-worker.py"), directLinearBackend, directFusedMlp, directFusedFfn, directFusedDecoderLayer, directFusedDecoderStack, directFusedPle, directFusedPlePrelude, directFusedTokenForward, directResidentGeneration, directFinalHead, directNativeAttention, directFusedAttention, directThreads, directMaxReadMiB, directFinalHeadReadMiB, ...(directVerificationMargin === undefined ? {} : { directVerificationMargin }) } : {}),
  };
}
