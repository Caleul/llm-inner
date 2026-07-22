import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { gemma4RealCompareHtml } from "./gemma4-real-compare-ui.js";
import { readGemma4GlobalSsaOutput, resolveGemma4GlobalSsaRoot, type Gemma4GlobalSsaOutput, type Gemma4GlobalSsaRootResolution } from "./gemma4-global-ssa-output-reader.js";
import { validateGemma4FinalFormulaRuntime } from "./gemma4-final-formula-runtime.js";
import { normalizeGemma4DecoderQuantizationLayers, type Gemma4MlxDecoderQuantization } from "./gemma4-paged-native-linear.js";
import { validateGemma4VectorizedRealLoweringPlan, type Gemma4VectorizedRealLoweringPlan } from "./gemma4-vectorized-real-lowering.js";

export interface Gemma4RealComparisonRequest { prompt: string; maxNewTokens: number; threads?: number; precision?: "f32" | "f64"; roundingPolicy?: "none" | "layer-bf16" | "operation-bf16"; sessionId?: number; continueSession?: boolean; conversationMode?: "raw" | "chat"; measurementSchedule?: "isolated" | "parallel"; captureLayerHidden?: boolean; captureLayerStages?: number }
export interface Gemma4RealComparisonRunnerOptions {
  source: string; python: string; helper: string; tokenizerHelper?: string;
  compiledProgram?: Gemma4CompiledProgramStatus;
  directMlxDecoderQuantization?: Gemma4MlxDecoderQuantization;
  directMlxDecoderQuantizationLayers?: string;
  literalArtifact?: string; binaryPool?: string; directArtifactIndex?: string; directArtifactIndexSha256?: string; directArtifactSha256?: string; directWorker?: string; directLinearHelper?: string; directMlxHelper?: string; directLinearBackend?: "pytorch" | "mlx"; directMlxHeadQuantization?: "off" | "q8" | "q8-shortlist" | "q4"; directFusedMlp?: "off" | "bf16" | "real" | "native-bf16"; directFusedFfn?: "off" | "native-bf16"; directFusedDecoderLayer?: "off" | "native-bf16"; directFusedDecoderStack?: "off" | "real" | "native-bf16" | "native-bf16-ple"; directFusedPle?: "off" | "bf16" | "real"; directFusedPlePrelude?: "off" | "bf16" | "real"; directFusedTokenForward?: "off" | "bf16"; directResidentGeneration?: "off" | "on"; directFinalHead?: "f32" | "native-bf16" | "native-bf16-stream" | "native-bf16-whole"; directNativeAttention?: "off" | "bf16" | "real"; directFusedAttention?: "off" | "bf16" | "real" | "native-bf16"; directThreads?: number; directMaxReadMiB?: number; directFinalHeadReadMiB?: number; directVerificationMargin?: number; directVerificationBackend?: "pytorch" | "mlx-control" | "mlx-shared-control";
}

export interface Gemma4CompiledProgramStatus {
  bundle: string;
  execution: "compiled-parametric-output-program-runtime";
  formulaSemantics: "gemma4-exact-real-simplified-v1";
  globalFormula: { file: string; sha256: string; terminalLogits: number; role: "compiled-executable-shared-dag" };
  finalFormulaMap: { file: string; sha256: string; functions: number; inputTensor: "x"; evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))"; orderedRootsSha256: string };
  finalFormulaRuntime: { file: "final-formulas.runtime.json"; sha256: string; schemaVersion: 1; evaluator: "EVAL_EXACT_DAG" };
  constantPool: { file: string; sha256: string };
  exampleFormula: { key: string; family: string; dimension: number; root: string; expressionNodes: number; inputTensor: "x"; inputLength: number; file: string; sha256: string };
  terminalLogitOutputs: { offset: number; dimensions: number; operationId: string; finalQuantization: "BF16-round-to-nearest-ties-to-even" };
  runtimeIndex?: { file: string; schemaVersion: 1 | 2; sha256: string; constantPoolSha256: string; integrityRootSha256: string };
  directRuntime: { engine: "mlx-f32-real-decoder-stack-v1"; directlyExecutesGlobalFormula: true; executesPersistedLoweringPlan: true; compiledOutputProgram: Gemma4VectorizedRealLoweringPlan["contract"]["compiledOutputProgram"]; plan: { file: string; sha256: string; functionBindingsSha256: string; outputBindingsSha256: string; standaloneSsaOutputsSha256: string; outputFunctions: number; realSimplifiedProgramSha256: string } };
}

const GEMMA4_CHAT_TEMPLATE = "google-gemma4-it-text-turn-v1";
const GEMMA4_CHAT_SOT_TOKEN_ID = 105;
const GEMMA4_CHAT_EOT_TOKEN_ID = 106;
const GEMMA4_CHAT_CONTRACT_SOURCE = "https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4";
interface SessionInput { tokenIds: number[]; mode: "raw" | "chat"; assistantTurnClosed: boolean }

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
  options = {
    ...options,
    directMlxHeadQuantization: options.directMlxHeadQuantization ?? "off",
    directMlxDecoderQuantization: options.directMlxDecoderQuantization ?? (directBackend === "mlx" ? "q8-ffn-gate-up" : "off"),
  };
  options.directVerificationBackend ??= options.directMlxDecoderQuantization === "q8-ffn-gate-up-down" ? "mlx-shared-control" : "pytorch";
  if (options.directVerificationMargin !== undefined && (!Number.isFinite(options.directVerificationMargin) || options.directVerificationMargin < 0)) throw new Error("directVerificationMargin deve ser finita e não negativa.");
  if (options.directVerificationMargin !== undefined && directBackend !== "mlx") throw new Error("directVerificationMargin requer backend MLX.");
  if (options.directMlxDecoderQuantizationLayers !== undefined) {
    options.directMlxDecoderQuantizationLayers = normalizeGemma4DecoderQuantizationLayers(options.directMlxDecoderQuantizationLayers);
    if (options.directMlxDecoderQuantization !== "q8-ffn-gate-up" && options.directMlxDecoderQuantization !== "q8-ffn-gate-up-down" && options.directMlxDecoderQuantization !== "q4-ffn-gate-up") throw new Error("directMlxDecoderQuantizationLayers requer q8-ffn-gate-up, q8-ffn-gate-up-down ou q4-ffn-gate-up.");
  }
  const checkpointChatTemplateDeclared = declaresChatTemplate(options.source);
  const directTokenForward = options.directFusedTokenForward ?? (directBackend === "mlx" && options.directFusedDecoderStack !== "off" ? "bf16" : "off");
  const directResidentGeneration = options.directResidentGeneration ?? (directTokenForward === "bf16" ? "on" : "off");
  const direct = options.literalArtifact && options.binaryPool
    ? new PersistentJsonlWorker(process.execPath, directWorkerArguments(options, directBackend, true), "Gemma 4 literal direto")
    : undefined;
  const tokenizer = new Gemma4PersistentTokenizerWorker(options);
  const verificationEnabled = direct !== undefined && directBackend === "mlx" && options.directVerificationMargin !== undefined;
  let verification: PersistentJsonlWorker | undefined, verificationInitialization: Promise<PersistentJsonlWorker> | undefined, verificationWarmupSeconds: number | undefined;
  let worker: Gemma4PersistentComparisonWorker | undefined, referenceInitialization: Promise<Gemma4PersistentComparisonWorker> | undefined, initializationError: Error | undefined, referenceError: Error | undefined, closed = false, referenceLeases = 0, referenceReleaseRequested = false;
  const sessionInputs = new Map<number, SessionInput>();
  const finalFormulaCache = new Map<number, Promise<Gemma4GlobalSsaOutput>>();
  const finalFormulaResolutionCache = new Map<string, Promise<Gemma4GlobalSsaRootResolution>>();
  let directWarmupSeconds: number | undefined;
  let directBackgroundRecovery: Promise<number> | undefined, directBackgroundRecoveryError: Error | undefined, lastDirectBackgroundRecoverySeconds: number | undefined;
  const waitForDirectBackgroundRecovery = async (): Promise<{ waitedSeconds: number; recoverySeconds?: number }> => {
    if (directBackgroundRecoveryError) throw directBackgroundRecoveryError;
    const pending = directBackgroundRecovery;
    if (!pending) return { waitedSeconds: 0 };
    const started = performance.now(), recoverySeconds = await pending;
    if (directBackgroundRecoveryError) throw directBackgroundRecoveryError;
    return { waitedSeconds: elapsedSeconds(started), recoverySeconds };
  };
  const scheduleDirectBackgroundRecovery = (): boolean => {
    if (!direct || closed || directBackgroundRecovery || directBackgroundRecoveryError) return false;
    const recovery = recoverDirectWorker(direct); directBackgroundRecovery = recovery;
    void recovery.then((seconds) => {
      lastDirectBackgroundRecoverySeconds = seconds;
      if (directBackgroundRecovery === recovery) directBackgroundRecovery = undefined;
    }, (error) => {
      directBackgroundRecoveryError = error instanceof Error ? error : new Error(String(error));
      if (directBackgroundRecovery === recovery) directBackgroundRecovery = undefined;
    });
    return true;
  };
  const directBackgroundRecoveryStatus = (): Record<string, unknown> => ({
    state: directBackgroundRecoveryError ? "failed" : directBackgroundRecovery ? "running" : "idle",
    ...(lastDirectBackgroundRecoverySeconds === undefined ? {} : { lastSeconds: lastDirectBackgroundRecoverySeconds }),
    ...(directBackgroundRecoveryError ? { error: directBackgroundRecoveryError.message } : {}),
  });
  const getReferenceWorker = (): Promise<Gemma4PersistentComparisonWorker> => {
    if (referenceInitialization) return referenceInitialization;
    referenceInitialization = (async () => {
      try {
        if (closed) throw new Error("Servidor encerrado antes de inicializar a referência.");
        const candidate = new Gemma4PersistentComparisonWorker(options); worker = candidate;
        await candidate.whenReady(); return candidate;
      } catch (error) {
        referenceError = error instanceof Error ? error : new Error(String(error)); throw referenceError;
      }
    })();
    void referenceInitialization.catch(() => undefined);
    return referenceInitialization;
  };
  const acquireReferenceWorker = async (): Promise<{ worker: Gemma4PersistentComparisonWorker; coldStart: boolean; startupSeconds: number; release(dispose: boolean): void }> => {
    const coldStart = !worker?.ready, started = performance.now(); referenceLeases += 1;
    try {
      const acquired = await getReferenceWorker(); let released = false;
      return { worker: acquired, coldStart, startupSeconds: elapsedSeconds(started), release(dispose: boolean) {
        if (released) return; released = true; if (dispose) referenceReleaseRequested = true; referenceLeases -= 1;
        if (referenceLeases === 0 && referenceReleaseRequested) {
          worker?.close(); worker = undefined; referenceInitialization = undefined; referenceError = undefined; referenceReleaseRequested = false;
        }
      } };
    } catch (error) { referenceLeases -= 1; throw error; }
  };
  const getVerificationWorker = (): Promise<PersistentJsonlWorker> => {
    if (!verificationEnabled) return Promise.reject(new Error("Verificação direta não está habilitada."));
    if (verificationInitialization) return verificationInitialization;
    verificationInitialization = (async () => {
      const backend = options.directVerificationBackend === "mlx-control" ? "mlx" : "pytorch";
      const label = backend === "mlx" ? "Gemma 4 literal verificador MLX calibrado" : "Gemma 4 literal verificador PyTorch";
      const candidate = new PersistentJsonlWorker(process.execPath, directWorkerArguments(options, backend, false), label); verification = candidate;
      await candidate.whenReady();
      verificationWarmupSeconds = await warmupDirectWorker(candidate, backend === "mlx" ? 2 : 1);
      return candidate;
    })();
    void verificationInitialization.catch(() => undefined);
    return verificationInitialization;
  };
  const releaseVerificationWorker = async (): Promise<void> => {
    if (!verification?.ready) return;
    // The exact PyTorch kernel stays resident after its one-time warm-up. Its
    // mmap-backed weights are immutable, while session KV is independently
    // guarded by exact token-prefix validation inside the worker.
  };
  const verificationProvider = verificationEnabled ? { ...(options.directVerificationBackend === "mlx-shared-control" ? {} : { get: getVerificationWorker, release: releaseVerificationWorker }), backend: options.directVerificationBackend!, selectiveHeads: options.directVerificationBackend === "pytorch" } : undefined;
  const directVerificationStatus = (): Record<string, unknown> => {
    if (!verificationEnabled) return { enabled: false };
    if (options.directVerificationBackend === "mlx-shared-control") return {
      enabled: true, backend: "mlx-shared-control", executionBackend: direct?.readyMetadata.backend, lifecycle: "shared-worker-compiled-control-v1", marginThreshold: options.directVerificationMargin,
      state: direct?.ready ? "ready" : "initializing", ready: direct?.ready ?? false, initializationSeconds: direct?.initializationSeconds, warmupSeconds: directWarmupSeconds, warmupComplete: directWarmupSeconds !== undefined,
    };
    return {
      enabled: true, ...verification?.readyMetadata, backend: options.directVerificationBackend, executionBackend: verification?.readyMetadata.backend,
      lifecycle: options.directVerificationBackend === "mlx-control" ? "persistent-compiled-control-v1" : "warmed-persistent-exact-kernel-v1", marginThreshold: options.directVerificationMargin,
      state: verification?.ready ? "ready" : verificationInitialization ? "initializing" : "unloaded", ready: verification?.ready ?? false, initializationSeconds: verification?.initializationSeconds,
      warmupSeconds: verificationWarmupSeconds, warmupComplete: verificationWarmupSeconds !== undefined,
    };
  };
  const initialize = (async () => {
    try {
      const tokenizerReady = tokenizer.whenReady();
      if (direct && directBackend === "mlx") {
        await direct.whenReady();
        if (verificationEnabled && options.directVerificationBackend !== "mlx-shared-control") await getVerificationWorker();
        directWarmupSeconds = await warmupDirectWorker(direct, directResidentGeneration === "on" ? 2 : 1);
        if (options.compiledProgram && (options.directFusedDecoderStack ?? "real") === "real") assertDirectFinalFormulaProgram(direct.readyMetadata, options.compiledProgram.finalFormulaMap, options.compiledProgram.finalFormulaRuntime);
        direct.readyMetadata.mlxDecoderQuantization = options.directMlxDecoderQuantization ?? "q8-ffn-gate-up";
        if (options.directMlxDecoderQuantizationLayers !== undefined) direct.readyMetadata.mlxDecoderQuantizationLayers = options.directMlxDecoderQuantizationLayers;
      }
      await tokenizerReady;
      if (!direct) await getReferenceWorker();
    } catch (error) {
      initializationError = error instanceof Error ? error : new Error(String(error));
      throw initializationError;
    }
  })();
  void initialize.catch(() => undefined);
  const server = createServer(async (request, response) => {
    try {
      if (request.method === "GET" && request.url === "/") return send(response, 200, "text/html; charset=utf-8", gemma4RealCompareHtml);
      if (request.method === "GET" && request.url?.split("?", 1)[0] === "/api/final-formula") {
        if (!options.compiledProgram) return json(response, 404, { error: "Bundle compilado não está disponível." });
        const url = new URL(request.url, "http://127.0.0.1"), rawDimension = url.searchParams.get("dimension"), dimension = Number(rawDimension);
        const outputs = options.compiledProgram.terminalLogitOutputs;
        if (rawDimension === null || !/^\d+$/.test(rawDimension) || !Number.isSafeInteger(dimension) || dimension < 0 || dimension >= outputs.dimensions) throw new Error(`dimension deve estar entre 0 e ${outputs.dimensions - 1}.`);
        let pending = finalFormulaCache.get(dimension);
        if (!pending) {
          const path = join(options.compiledProgram.bundle, options.compiledProgram.globalFormula.file);
          pending = readGemma4GlobalSsaOutput(path, outputs.offset + dimension); finalFormulaCache.set(dimension, pending);
          void pending.catch(() => finalFormulaCache.delete(dimension));
        }
        const binding = await pending, expectedAssignment = `calc_terminal_logit_${dimension}`;
        if (binding.assignment !== expectedAssignment || binding.finalQuantization !== outputs.finalQuantization) throw new Error(`Binding final ${dimension} diverge do contrato terminal_logit.`);
        const globalPath = join(options.compiledProgram.bundle, options.compiledProgram.globalFormula.file);
        let resolution = finalFormulaResolutionCache.get(binding.value);
        if (!resolution) {
          resolution = resolveGemma4GlobalSsaRoot(globalPath, binding.value); finalFormulaResolutionCache.set(binding.value, resolution);
          void resolution.catch(() => finalFormulaResolutionCache.delete(binding.value));
        }
        const resolved = await resolution;
        await tokenizer.whenReady(); const decoded = await tokenizer.decode([dimension]);
        return json(response, 200, {
          key: `calc_final_${dimension}`, family: "terminal_logit", dimension, token: { id: dimension, text: decoded.text },
          expression: `BF16_RNE(EVAL_EXACT_DAG(\"${binding.value}\", x))`, root: binding.value, parameters: binding.parameters, coordinate: binding.coordinate,
          resolvedMath: {
            expression: resolved.root.expression, node: resolved.root.node, dependencies: resolved.dependencies, fragment: resolved.fragment,
            ...(resolved.calledFunction ? { operation: {
              functionId: resolved.calledFunction.functionId, operationId: resolved.calledFunction.operationId, output: resolved.calledFunction.output,
              parameters: resolved.calledFunction.parameters, argumentRoots: resolved.dependencies, bodyRoot: resolved.calledFunction.root,
              bodyExpression: resolved.calledFunction.body.expression, bodyNode: resolved.calledFunction.body.node, bodyDependencies: resolved.calledFunction.bodyDependencies,
              bodyFragment: resolved.calledFunction.bodyFragment,
              predecessorFunctions: resolved.calledFunction.predecessorFunctions, operandBoundaries: resolved.calledFunction.operandBoundaries,
            } } : {}),
          },
          formulaMap: { file: options.compiledProgram.finalFormulaMap.file, key: `calc_final_${dimension}`, sha256: options.compiledProgram.finalFormulaMap.sha256 },
          formulaRuntime: options.compiledProgram.finalFormulaRuntime,
          finalQuantization: binding.finalQuantization, onlyFreeInput: "x", globalFormulaSha256: options.compiledProgram.globalFormula.sha256,
          outputBindingsSha256: options.compiledProgram.directRuntime.plan.outputBindingsSha256,
        });
      }
      if (request.method === "GET" && request.url === "/api/status") return json(response, 200, { ready: tokenizer.ready && (!direct ? worker?.ready === true : direct.ready && directWarmupSeconds !== undefined && !directBackgroundRecoveryError && (!verificationEnabled || options.directVerificationBackend === "mlx-shared-control" || verificationWarmupSeconds !== undefined)), source: options.source, runtime: "persistent-jsonl", chatTemplate: GEMMA4_CHAT_TEMPLATE, chatContract: { source: GEMMA4_CHAT_CONTRACT_SOURCE, startOfTurnToken: { text: "<|turn>", id: GEMMA4_CHAT_SOT_TOKEN_ID }, endOfTurnToken: { text: "<turn|>", id: GEMMA4_CHAT_EOT_TOKEN_ID }, generationStopsAtEndOfTurn: true }, checkpointChatTemplateDeclared, tokenizer: { ready: tokenizer.ready, initializationSeconds: tokenizer.initializationSeconds, helper: options.tokenizerHelper }, reference: { state: worker?.ready ? "ready" : referenceInitialization ? "initializing" : "unloaded", initializationSeconds: worker?.initializationSeconds, ...(referenceError ? { error: referenceError.message } : {}) }, ...(options.compiledProgram ? { compiledProgram: options.compiledProgram } : {}), ...(initializationError ? { error: initializationError.message } : {}), direct: direct ? { enabled: true, artifact: options.literalArtifact, binaryPool: options.binaryPool, sourceIndependentBundle: options.literalArtifact !== undefined && options.binaryPool === dirname(options.literalArtifact), ...direct.readyMetadata, ready: direct.ready, initializationSeconds: direct.initializationSeconds, warmupSeconds: directWarmupSeconds, warmupComplete: directWarmupSeconds !== undefined, backgroundRecovery: directBackgroundRecoveryStatus(), verification: directVerificationStatus() } : { enabled: false } });
      if (request.method === "POST" && request.url === "/api/diagnose-layers") {
        await initialize;
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        if (!direct || !verificationEnabled || options.directVerificationBackend !== "pytorch") throw new Error("Diagnóstico por camada requer execução compilada MLX com verificador PyTorch habilitado.");
        await waitForDirectBackgroundRecovery();
        if (body.sessionId !== undefined || body.continueSession || body.conversationMode === "chat") throw new Error("Diagnóstico por camada aceita somente prompts raw sem sessão.");
        const inputIds = await resolveComparisonInput(tokenizer, body, sessionInputs);
        if (inputIds.length > 16 || body.maxNewTokens > 8) throw new Error("Diagnóstico tensorial aceita no máximo 16 tokens de entrada e 8 tokens novos.");
        const fast = await direct.send({ inputIds, maxNewTokens: body.maxNewTokens }) as DirectReport;
        if (typeof fast.terminalLogitsSha256 !== "string" || !Array.isArray(fast.generatedTokenIds)) throw new Error("Execução rápida não retornou contrato verificável para diagnóstico.");
        const verificationWorker = await getVerificationWorker(), lease = await acquireReferenceWorker();
        let reference: ComparisonReport, verifier: DirectReport;
        try {
          [reference, verifier] = await Promise.all([
            lease.worker.compareTokens({ ...body, captureLayerHidden: true }, inputIds) as Promise<ComparisonReport>,
            verificationWorker.send({ inputIds, maxNewTokens: body.maxNewTokens, captureLayerHidden: true, ...(body.captureLayerStages === undefined ? {} : { captureLayerStages: body.captureLayerStages }), verificationFastPath: { generatedTokenIds: fast.generatedTokenIds, sensitiveSteps: fast.generatedTokenIds.map((_token, step) => step), terminalLogitsSha256: fast.terminalLogitsSha256 } }) as Promise<DirectReport>,
          ]);
        } finally {
          lease.release(true);
          await releaseVerificationWorker();
        }
        return json(response, 200, { kind: "gemma4-decoder-layer-differential", schemaVersion: 1, prompt: body.prompt, inputIds, fastGeneratedTokenIds: fast.generatedTokenIds, reference, verifier });
      }
      if (request.method === "POST" && request.url === "/api/compare") {
        await initialize;
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        const inputIds = await resolveComparisonInput(tokenizer, body, sessionInputs);
        const directRecoveryWait = direct ? await waitForDirectBackgroundRecovery() : { waitedSeconds: 0 };
        const comparisonStarted = performance.now();
        const schedule = body.measurementSchedule ?? "isolated";
        if (!direct) {
          const lease = await acquireReferenceWorker(); let report: ComparisonReport;
          try { report = await lease.worker.compareTokens(body, inputIds, chatStopToken(body)) as ComparisonReport; } finally { lease.release(false); }
          return json(response, 200, { ...report, prompt: body.prompt, conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null, comparisonTiming: { schedule: "reference-only", totalWallSeconds: elapsedSeconds(comparisonStarted) } });
        }
        const executeDirect = () => executeSelectedDirect(direct, verificationProvider, options.directVerificationMargin, { inputIds, maxNewTokens: body.maxNewTokens, ...(body.conversationMode === "chat" ? { eosTokenId: GEMMA4_CHAT_EOT_TOKEN_ID } : {}), ...(body.sessionId === undefined ? {} : { sessionId: body.sessionId }) });
        let report: ComparisonReport, directReport: DirectReport, directPhaseSeconds: number, referencePhaseSeconds: number, referenceStartupSeconds: number, referenceComputeSeconds: number, referenceColdStart: boolean, directRecoverySeconds = 0;
        if (schedule === "isolated") {
          const directStarted = performance.now(); directReport = await executeDirect(); directPhaseSeconds = elapsedSeconds(directStarted);
          const referenceStarted = performance.now(), lease = await acquireReferenceWorker(); referenceStartupSeconds = lease.startupSeconds; referenceColdStart = lease.coldStart;
          const computeStarted = performance.now(); try { report = await lease.worker.compareTokens(body, inputIds, chatStopToken(body)) as ComparisonReport; } finally { lease.release(true); } referenceComputeSeconds = elapsedSeconds(computeStarted); referencePhaseSeconds = elapsedSeconds(referenceStarted);
        } else {
          const lease = await acquireReferenceWorker(); referenceStartupSeconds = lease.startupSeconds; referenceColdStart = lease.coldStart;
          const directStarted = performance.now(), referenceStarted = performance.now();
          try { [report, directReport] = await Promise.all([lease.worker.compareTokens(body, inputIds, chatStopToken(body)) as Promise<ComparisonReport>, executeDirect()]); } finally { lease.release(true); }
          directPhaseSeconds = elapsedSeconds(directStarted); referenceComputeSeconds = elapsedSeconds(referenceStarted); referencePhaseSeconds = referenceStartupSeconds + referenceComputeSeconds;
        }
        directRecoverySeconds = await recoverDirectWorker(direct);
        attachDirectLogitAgreement(report, directReport);
        const [generated, full] = await Promise.all([tokenizer.decode(directReport.generatedTokenIds), tokenizer.decode(directReport.fullTokenIds)]);
        directReport.generatedText = generated.text; directReport.fullText = full.text;
        directReport.tokensEqualBaseline = arraysEqual(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
        directReport.firstDivergentStep = firstDivergence(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
        attachDirectFallbackOutcome(directReport, report.baselineGeneratedTokenIds);
        if (body.sessionId !== undefined) rememberSession(sessionInputs, body.sessionId, directReport.fullTokenIds, body.conversationMode ?? "raw");
        return json(response, 200, { ...report, prompt: body.prompt, conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null, ...(options.compiledProgram ? { compiledProgram: options.compiledProgram } : {}), direct: directReport, directExecutionMetrics: computeDirectExecutionMetrics(directReport, inputIds.length, report.performance?.baselineSeconds), comparisonTiming: { schedule, directBackgroundRecoveryWaitSeconds: directRecoveryWait.waitedSeconds, ...(directRecoveryWait.recoverySeconds === undefined ? {} : { awaitedBackgroundRecoverySeconds: directRecoveryWait.recoverySeconds }), directPhaseSeconds, referenceColdStart, referenceStartupSeconds, referenceComputeSeconds, referencePhaseSeconds, referenceReleased: true, directRecoverySeconds, totalWallSeconds: elapsedSeconds(comparisonStarted) } });
      }
      if (request.method === "POST" && request.url === "/api/compare-stream") {
        await initialize;
        if (!direct) throw new Error("Streaming comparativo requer o executor direto.");
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        const inputIds = await resolveComparisonInput(tokenizer, body, sessionInputs);
        const directRecoveryWait = await waitForDirectBackgroundRecovery();
        let clientDisconnected = false; response.once("close", () => { if (!response.writableEnded) clientDisconnected = true; });
        response.writeHead(200, { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store", "x-content-type-options": "nosniff" });
        try {
          const comparisonStarted = performance.now(), schedule = body.measurementSchedule ?? "isolated";
          let directExecutionStarted = comparisonStarted, firstTokenWallSeconds: number | undefined;
          const decodedStream = new DecodedTokenNdjsonStream(tokenizer, response);
          const executeDirect = () => executeSelectedDirect(direct, verificationProvider, options.directVerificationMargin, { inputIds, maxNewTokens: body.maxNewTokens, stream: true, ...(body.conversationMode === "chat" ? { eosTokenId: GEMMA4_CHAT_EOT_TOKEN_ID } : {}), ...(body.sessionId === undefined ? {} : { sessionId: body.sessionId }) }, (event) => {
            firstTokenWallSeconds ??= elapsedSeconds(directExecutionStarted);
            decodedStream.push(event, { type: "direct-token", provisional: verificationEnabled, serverElapsedSeconds: elapsedSeconds(directExecutionStarted) });
          }, (assessment) => decodedStream.write({ type: "direct-fallback", ...assessment }), () => !clientDisconnected, (prefill) => decodedStream.write({ type: "direct-verification-prefill", ...prefill }));
          let report: ComparisonReport, directReport: DirectReport, directPhaseSeconds: number, referencePhaseSeconds: number, referenceStartupSeconds: number, referenceComputeSeconds: number, referenceColdStart: boolean;
          if (schedule === "isolated") {
            const directStarted = performance.now(); directExecutionStarted = directStarted; directReport = await executeDirect(); directPhaseSeconds = elapsedSeconds(directStarted);
            if (clientDisconnected || response.destroyed) return;
            await decodedStream.flush();
            const [generated, full] = await Promise.all([tokenizer.decode(directReport.generatedTokenIds), tokenizer.decode(directReport.fullTokenIds)]);
            directReport.generatedText = generated.text; directReport.fullText = full.text;
            writeNdjson(response, { type: "direct-complete", data: { generatedText: generated.text, generatedTokenIds: directReport.generatedTokenIds, directPhaseSeconds } });
            writeNdjson(response, { type: "reference-loading", coldStart: !worker?.ready });
            const referenceStarted = performance.now(), lease = await acquireReferenceWorker(); referenceStartupSeconds = lease.startupSeconds; referenceColdStart = lease.coldStart;
            const computeStarted = performance.now(); try { report = await lease.worker.compareTokens(body, inputIds, chatStopToken(body)) as ComparisonReport; } finally { lease.release(true); } referenceComputeSeconds = elapsedSeconds(computeStarted); referencePhaseSeconds = elapsedSeconds(referenceStarted);
          } else {
            writeNdjson(response, { type: "reference-loading", coldStart: !worker?.ready });
            const lease = await acquireReferenceWorker(); referenceStartupSeconds = lease.startupSeconds; referenceColdStart = lease.coldStart;
            const directStarted = performance.now(), referenceStarted = performance.now(); directExecutionStarted = directStarted;
            try { [report, directReport] = await Promise.all([lease.worker.compareTokens(body, inputIds, chatStopToken(body)) as Promise<ComparisonReport>, executeDirect()]); } finally { lease.release(true); }
            directPhaseSeconds = elapsedSeconds(directStarted); referenceComputeSeconds = elapsedSeconds(referenceStarted); referencePhaseSeconds = referenceStartupSeconds + referenceComputeSeconds;
          }
          await decodedStream.flush();
          if (clientDisconnected || response.destroyed) return;
          attachDirectLogitAgreement(report, directReport);
          if (directReport.generatedText === undefined || directReport.fullText === undefined) {
            const [generated, full] = await Promise.all([tokenizer.decode(directReport.generatedTokenIds), tokenizer.decode(directReport.fullTokenIds)]);
            directReport.generatedText = generated.text; directReport.fullText = full.text;
          }
          directReport.tokensEqualBaseline = arraysEqual(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
          directReport.firstDivergentStep = firstDivergence(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
          attachDirectFallbackOutcome(directReport, report.baselineGeneratedTokenIds);
          if (body.sessionId !== undefined) rememberSession(sessionInputs, body.sessionId, directReport.fullTokenIds, body.conversationMode ?? "raw");
          writeNdjson(response, { type: "direct-rewarming", background: true });
          const backgroundRecoveryScheduled = scheduleDirectBackgroundRecovery();
          writeNdjson(response, { type: "result", data: { ...report, prompt: body.prompt, conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null, ...(options.compiledProgram ? { compiledProgram: options.compiledProgram } : {}), direct: directReport, directExecutionMetrics: computeDirectExecutionMetrics(directReport, inputIds.length, report.performance?.baselineSeconds, firstTokenWallSeconds), comparisonTiming: { schedule, directBackgroundRecoveryWaitSeconds: directRecoveryWait.waitedSeconds, ...(directRecoveryWait.recoverySeconds === undefined ? {} : { awaitedBackgroundRecoverySeconds: directRecoveryWait.recoverySeconds }), backgroundRecoveryScheduled, directPhaseSeconds, referenceColdStart, referenceStartupSeconds, referenceComputeSeconds, referencePhaseSeconds, referenceReleased: true, totalWallSeconds: elapsedSeconds(comparisonStarted) } } }); response.end(); return;
        } catch (error) {
          writeNdjson(response, { type: "error", error: error instanceof Error ? error.message : String(error) }); response.end(); return;
        }
      }
      if (request.method === "POST" && request.url === "/api/generate-stream") {
        await initialize;
        if (!direct) throw new Error("Streaming compilado requer o executor direto.");
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        const inputIds = await resolveComparisonInput(tokenizer, body, sessionInputs);
        const directRecoveryWait = await waitForDirectBackgroundRecovery();
        let clientDisconnected = false; response.once("close", () => { if (!response.writableEnded) clientDisconnected = true; });
        response.writeHead(200, { "content-type": "application/x-ndjson; charset=utf-8", "cache-control": "no-store", "x-content-type-options": "nosniff" });
        try {
          const generationStarted = performance.now(); let firstTokenWallSeconds: number | undefined;
          const decodedStream = new DecodedTokenNdjsonStream(tokenizer, response);
          const directReport = await executeSelectedDirect(direct, verificationProvider, options.directVerificationMargin, {
            inputIds, maxNewTokens: body.maxNewTokens, stream: true,
            ...(body.conversationMode === "chat" ? { eosTokenId: GEMMA4_CHAT_EOT_TOKEN_ID } : {}),
            ...(body.sessionId === undefined ? {} : { sessionId: body.sessionId }),
          }, (event) => {
            firstTokenWallSeconds ??= elapsedSeconds(generationStarted);
            decodedStream.push(event, { type: "direct-token", provisional: verificationEnabled, serverElapsedSeconds: elapsedSeconds(generationStarted) });
          }, (assessment) => decodedStream.write({ type: "direct-fallback", ...assessment }), () => !clientDisconnected, (prefill) => decodedStream.write({ type: "direct-verification-prefill", ...prefill }));
          const directPhaseSeconds = elapsedSeconds(generationStarted);
          if (clientDisconnected || response.destroyed) return;
          await decodedStream.flush();
          const [generated, full] = await Promise.all([tokenizer.decode(directReport.generatedTokenIds), tokenizer.decode(directReport.fullTokenIds)]);
          directReport.generatedText = generated.text; directReport.fullText = full.text;
          if (body.sessionId !== undefined) rememberSession(sessionInputs, body.sessionId, directReport.fullTokenIds, body.conversationMode ?? "raw");
          writeNdjson(response, { type: "direct-complete", data: { generatedText: generated.text, generatedTokenIds: directReport.generatedTokenIds, directPhaseSeconds } });
          const backgroundRecoveryScheduled = directReport.fallbackTriggered === true && scheduleDirectBackgroundRecovery();
          writeNdjson(response, { type: "result", data: {
            executionMode: "compiled-only", referenceExecuted: false, prompt: body.prompt,
            conversationMode: body.conversationMode ?? "raw", chatTemplate: body.conversationMode === "chat" ? GEMMA4_CHAT_TEMPLATE : null,
            ...(options.compiledProgram ? { compiledProgram: options.compiledProgram } : {}), direct: directReport,
            directExecutionMetrics: computeDirectExecutionMetrics(directReport, inputIds.length, undefined, firstTokenWallSeconds, false),
            comparisonTiming: { schedule: "compiled-only", directBackgroundRecoveryWaitSeconds: directRecoveryWait.waitedSeconds, ...(directRecoveryWait.recoverySeconds === undefined ? {} : { awaitedBackgroundRecoverySeconds: directRecoveryWait.recoverySeconds }), backgroundRecoveryScheduled, directPhaseSeconds, totalWallSeconds: elapsedSeconds(generationStarted) },
          } }); response.end(); return;
        } catch (error) {
          writeNdjson(response, { type: "error", error: error instanceof Error ? error.message : String(error) }); response.end(); return;
        }
      }
      return json(response, 404, { error: "Rota não encontrada." });
    } catch (error) {
      return json(response, 400, { error: error instanceof Error ? error.message : String(error) });
    }
  });
  server.once("close", () => { closed = true; tokenizer.close(); worker?.close(); direct?.close(); verification?.close(); void initialize.catch(() => undefined); });
  return server;
}

function directWorkerArguments(options: Gemma4RealComparisonRunnerOptions, backend: "pytorch" | "mlx", primary: boolean): string[] {
  if (!options.literalArtifact || !options.binaryPool) throw new Error("Worker direto requer artefato literal e pool binário.");
  const compiledControl = !primary && backend === "mlx";
  const accelerated = primary || compiledControl;
  const decoderStack = accelerated ? options.directFusedDecoderStack ?? (backend === "mlx" ? "real" : "native-bf16") : "native-bf16-ple";
  const tokenForward = backend === "mlx" ? options.directFusedTokenForward ?? (decoderStack !== "off" ? "bf16" : "off") : "off";
  const residentGeneration = backend === "mlx" ? options.directResidentGeneration ?? (tokenForward === "bf16" ? "on" : "off") : "off";
  const runtimeIndex = [
    ...(options.directArtifactIndex && options.directArtifactIndexSha256 && options.directArtifactSha256 ? ["--artifact-index", options.directArtifactIndex, "--artifact-index-sha256", options.directArtifactIndexSha256, "--artifact-sha256", options.directArtifactSha256] : []),
    "--mlx-decoder-quantization", primary ? options.directMlxDecoderQuantization ?? (backend === "mlx" ? "q8-ffn-gate-up" : "off") : compiledControl ? "q8-ffn-gate-up" : "off",
    ...(primary && options.directMlxDecoderQuantizationLayers !== undefined ? ["--mlx-decoder-quantization-layers", options.directMlxDecoderQuantizationLayers] : []),
  ];
  const finalFormulaProgram = accelerated && decoderStack === "real" && options.compiledProgram ? ["--final-formula-map", join(options.compiledProgram.bundle, options.compiledProgram.finalFormulaMap.file), "--final-formula-map-sha256", options.compiledProgram.finalFormulaMap.sha256, "--final-formula-functions", String(options.compiledProgram.finalFormulaMap.functions), "--final-formula-roots-sha256", options.compiledProgram.finalFormulaMap.orderedRootsSha256, "--final-formula-runtime", join(options.compiledProgram.bundle, options.compiledProgram.finalFormulaRuntime.file), "--final-formula-runtime-sha256", options.compiledProgram.finalFormulaRuntime.sha256] : [];
  return [options.directWorker ?? resolve("dist/src/gemma4-paged-runtime-worker-cli.js"), "--artifact", options.literalArtifact, ...runtimeIndex, ...finalFormulaProgram, "--binary-pool", options.binaryPool, "--python", options.python, "--linear-helper", options.directLinearHelper ?? resolve("scripts/gemma4-paged-linear-worker.py"), "--mlx-helper", options.directMlxHelper ?? resolve("scripts/gemma4-mlx-linear-worker.py"), "--linear-backend", backend, "--mlx-head-quantization", accelerated ? options.directMlxHeadQuantization ?? "off" : "off", "--fused-mlp", accelerated ? options.directFusedMlp ?? (backend === "mlx" ? "real" : "native-bf16") : "native-bf16", "--fused-ffn", accelerated ? options.directFusedFfn ?? (backend === "mlx" ? "off" : "native-bf16") : "native-bf16", "--fused-decoder-layer", accelerated ? options.directFusedDecoderLayer ?? (backend === "mlx" ? "off" : "native-bf16") : "native-bf16", "--fused-decoder-stack", decoderStack, "--fused-ple", accelerated ? options.directFusedPle ?? (backend === "mlx" ? "off" : "bf16") : "bf16", "--fused-ple-prelude", accelerated ? options.directFusedPlePrelude ?? (backend === "mlx" ? "bf16" : "off") : "bf16", "--fused-token-forward", tokenForward, "--resident-generation", residentGeneration, "--final-head", accelerated ? options.directFinalHead ?? (backend === "mlx" ? "native-bf16-whole" : "native-bf16-stream") : "native-bf16-stream", "--native-attention", accelerated ? options.directNativeAttention ?? (backend === "mlx" ? "off" : "real") : "real", "--fused-attention", accelerated ? options.directFusedAttention ?? (backend === "mlx" ? "off" : "native-bf16") : "native-bf16", "--threads", String(options.directThreads ?? 8), "--max-read-mib", String(options.directMaxReadMiB ?? 16), "--final-head-read-mib", String(accelerated ? options.directFinalHeadReadMiB ?? (backend === "mlx" ? options.directMaxReadMiB ?? 16 : 32) : 32)];
}

async function warmupDirectWorker(worker: PersistentJsonlWorker, maxNewTokens: number): Promise<number> {
  const started = performance.now();
  await worker.send({ inputIds: [2], maxNewTokens });
  if (maxNewTokens > 1) await worker.send({ inputIds: [2, 2, 2, 2, 2, 2, 2, 2], maxNewTokens });
  return (performance.now() - started) / 1000;
}

async function recoverDirectWorker(worker: PersistentJsonlWorker): Promise<number> {
  return warmupDirectWorker(worker, 2);
}

interface DirectMarginAssessment { trigger: boolean; reason: "margin-at-or-below-threshold" | "margin-unavailable"; minimumMargin: number | null; marginThreshold: number; sensitiveSteps: number[] }

interface VerificationPrefillReport {
  verificationPrefill?: boolean;
  elapsedSeconds?: number;
  verificationSessionCacheHit?: boolean;
  verificationPrefixTokensReused?: number;
  verificationPrefillTokensComputed?: number;
}

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

async function executeSelectedDirect(primary: PersistentJsonlWorker, verificationProvider: { get?(): Promise<PersistentJsonlWorker>; release?(): Promise<void>; backend: "pytorch" | "mlx-control" | "mlx-shared-control"; selectiveHeads: boolean } | undefined, marginThreshold: number | undefined, payload: { inputIds: number[]; maxNewTokens: number; eosTokenId?: number; sessionId?: number; stream?: boolean }, onEvent?: (event: unknown) => void, onFallback?: (assessment: DirectMarginAssessment & { verificationBackend: "pytorch" | "mlx-control" | "mlx-shared-control" }) => void, shouldVerify: () => boolean = () => true, onPrefillAhead?: (event: { step: number; margin: number; marginThreshold: number }) => void): Promise<DirectReport> {
  const started = performance.now();
  const canPrefillAhead = verificationProvider?.backend === "pytorch" && verificationProvider.selectiveHeads && marginThreshold !== undefined && payload.sessionId !== undefined;
  let prefillAhead: { startedAt: number; completedAt?: number; promise: Promise<VerificationPrefillReport> } | undefined;
  let terminalPrefillSkipped = false;
  const fast = await primary.send(canPrefillAhead ? { ...payload, stream: true } : payload, (event) => {
    onEvent?.(event);
    const sensitive = canPrefillAhead ? streamingEventSensitiveMargin(event, marginThreshold!) : undefined;
    if (!prefillAhead && sensitive) {
      if (!shouldStartDirectVerificationPrefill(sensitive, payload.maxNewTokens, payload.eosTokenId)) { terminalPrefillSkipped = true; return; }
      onPrefillAhead?.({ ...sensitive, marginThreshold: marginThreshold! });
      const startedAt = performance.now();
      const promise = verificationProvider!.get!()
        .then((verification) => verification.send({ inputIds: payload.inputIds, maxNewTokens: 1, sessionId: payload.sessionId, verificationPrefill: true }) as Promise<VerificationPrefillReport>)
        .then((report) => { if (prefillAhead) prefillAhead.completedAt = performance.now(); return report; });
      void promise.catch(() => undefined);
      prefillAhead = { startedAt, promise };
    }
  }) as DirectReport;
  const fastCompletedAt = performance.now();
  if (!verificationProvider || marginThreshold === undefined) return fast;
  const assessment = assessDirectVerification(fast, marginThreshold);
  const fastPathSeconds = (performance.now() - started) / 1000;
  if (!assessment.trigger) return Object.assign(fast, { selectionPolicy: "margin-verified-pytorch-v1", selectedBackend: "mlx", fallbackTriggered: false, fastPathMinimumMargin: assessment.minimumMargin, verificationMarginThreshold: marginThreshold, fastPathSeconds });
  if (!shouldVerify()) return fast;
  onFallback?.({ ...assessment, verificationBackend: verificationProvider.backend });
  const selectiveRequest = verificationProvider.selectiveHeads && assessment.reason === "margin-at-or-below-threshold" && assessment.sensitiveSteps.length > 0 && typeof fast.terminalLogitsSha256 === "string" && /^[0-9a-f]{64}$/.test(fast.terminalLogitsSha256)
    ? { verificationFastPath: { generatedTokenIds: fast.generatedTokenIds, sensitiveSteps: assessment.sensitiveSteps, terminalLogitsSha256: fast.terminalLogitsSha256, stopAfterDivergence: verificationProvider.backend === "pytorch" } }
    : {};
  let verified: DirectReport;
  if (verificationProvider.backend === "mlx-shared-control") {
    const verifiedTokenSteps = assessment.sensitiveSteps.at(-1)! + 1;
    const prefix = await primary.send({ inputIds: payload.inputIds, maxNewTokens: verifiedTokenSteps, verificationControl: true, ...(payload.eosTokenId === undefined ? {} : { eosTokenId: payload.eosTokenId }) }) as DirectReport;
    const prefixMatches = arraysEqual(prefix.generatedTokenIds, fast.generatedTokenIds.slice(0, prefix.generatedTokenIds.length)) && prefix.generatedTokenIds.length === Math.min(verifiedTokenSteps, fast.generatedTokenIds.length);
    if (prefixMatches) verified = { ...fast, ...(fast.steps ? { steps: fast.steps.map((step) => ({ ...step })) } : {}), controlVerificationTokenSteps: prefix.generatedTokenIds.length, controlCorrectionTriggered: false };
    else {
      verified = verifiedTokenSteps >= payload.maxNewTokens ? prefix : await primary.send({ inputIds: payload.inputIds, maxNewTokens: payload.maxNewTokens, verificationControl: true, ...(payload.eosTokenId === undefined ? {} : { eosTokenId: payload.eosTokenId }) }) as DirectReport;
      Object.assign(verified, { controlVerificationTokenSteps: verifiedTokenSteps, controlCorrectionTriggered: true });
    }
  } else {
    const verification = await verificationProvider.get!();
    let prefillReport: VerificationPrefillReport | undefined;
    try {
      if (prefillAhead) prefillReport = await prefillAhead.promise;
      const verificationRequest = { inputIds: payload.inputIds, maxNewTokens: payload.maxNewTokens, ...selectiveRequest, ...(payload.eosTokenId === undefined ? {} : { eosTokenId: payload.eosTokenId }), ...(payload.sessionId === undefined ? {} : { sessionId: payload.sessionId }) };
      verified = await verification.send(verificationRequest) as DirectReport;
      if (verified.verificationStoppedAfterDivergence === true && verified.generatedTokenIds.length < payload.maxNewTokens && verified.generatedTokenIds.at(-1) !== payload.eosTokenId) {
        const remaining = payload.maxNewTokens - verified.generatedTokenIds.length;
        const continuationInputIds = [...payload.inputIds, ...verified.generatedTokenIds];
        let continuation = await primary.send({ inputIds: continuationInputIds, maxNewTokens: remaining, ...(payload.eosTokenId === undefined ? {} : { eosTokenId: payload.eosTokenId }), ...(payload.sessionId === undefined ? {} : { sessionId: payload.sessionId }) }) as DirectReport;
        let continuationAssessment = assessDirectVerification(continuation, marginThreshold);
        let prefixRetry: { seconds: number; minimumMargin: number | null; tokenSteps: number } | undefined;
        if (continuationAssessment.trigger && continuation.sessionCacheHit === true) {
          prefixRetry = { seconds: finiteNonNegative(continuation.elapsedSeconds), minimumMargin: continuationAssessment.minimumMargin, tokenSteps: continuation.generatedTokenIds.length };
          continuation = await primary.send({ inputIds: continuationInputIds, maxNewTokens: remaining, ...(payload.eosTokenId === undefined ? {} : { eosTokenId: payload.eosTokenId }) }) as DirectReport;
          continuationAssessment = assessDirectVerification(continuation, marginThreshold);
        }
        if (!continuationAssessment.trigger) {
          verified = combineVerifiedPrefixWithCompiledContinuation(payload.inputIds, verified, continuation, continuationAssessment);
          if (prefixRetry) Object.assign(verified, {
            compiledContinuationPrefixRetry: true,
            compiledContinuationPrefixAttemptSeconds: prefixRetry.seconds,
            compiledContinuationPrefixAttemptMinimumMargin: prefixRetry.minimumMargin,
            compiledContinuationPrefixAttemptTokenSteps: prefixRetry.tokenSteps,
            compiledContinuationSeconds: prefixRetry.seconds + finiteNonNegative(continuation.elapsedSeconds),
          });
        }
        else {
          const fastPath = "verificationFastPath" in selectiveRequest ? selectiveRequest.verificationFastPath : undefined;
          verified = await verification.send({ ...verificationRequest, ...(fastPath ? { verificationFastPath: { ...fastPath, stopAfterDivergence: false } } : {}) }) as DirectReport;
          Object.assign(verified, {
            compiledContinuationAccepted: false,
            compiledContinuationRejectedReason: continuationAssessment.reason,
            compiledContinuationMinimumMargin: continuationAssessment.minimumMargin,
            compiledContinuationTokenSteps: continuation.generatedTokenIds.length,
            compiledContinuationSeconds: finiteNonNegative(continuation.elapsedSeconds) + (prefixRetry?.seconds ?? 0),
            ...(prefixRetry ? { compiledContinuationPrefixRetry: true, compiledContinuationPrefixAttemptSeconds: prefixRetry.seconds, compiledContinuationPrefixAttemptMinimumMargin: prefixRetry.minimumMargin, compiledContinuationPrefixAttemptTokenSteps: prefixRetry.tokenSteps } : {}),
          });
        }
      }
      if (prefillAhead && prefillReport) {
        const completedAt = prefillAhead.completedAt ?? performance.now();
        Object.assign(verified, {
          verificationPrefillAhead: true,
          verificationPrefillAheadSeconds: Math.max(0, (completedAt - prefillAhead.startedAt) / 1000),
          verificationPrefillAheadWorkerSeconds: typeof prefillReport.elapsedSeconds === "number" ? prefillReport.elapsedSeconds : undefined,
          verificationPrefillOverlapSeconds: Math.max(0, (Math.min(completedAt, fastCompletedAt) - prefillAhead.startedAt) / 1000),
          verificationPrefillWaitSeconds: Math.max(0, (completedAt - fastCompletedAt) / 1000),
          verificationPrefillAheadCacheHit: prefillReport.verificationSessionCacheHit === true,
          verificationPrefillAheadPrefixTokensReused: Number.isSafeInteger(prefillReport.verificationPrefixTokensReused) ? prefillReport.verificationPrefixTokensReused : 0,
          verificationPrefillAheadTokensComputed: Number.isSafeInteger(prefillReport.verificationPrefillTokensComputed) ? prefillReport.verificationPrefillTokensComputed : 0,
        });
      }
    }
    finally { await verificationProvider.release!(); }
  }
  if (verified.selectiveVerification === true && Array.isArray(verified.steps) && Array.isArray(fast.steps)) {
    verified.steps = verified.steps.map((step, index) => step?.verificationSkipped === true && fast.steps![index]
      ? { ...fast.steps![index], ...(step.forwardSeconds === undefined ? {} : { forwardSeconds: step.forwardSeconds }), verificationSkipped: true }
      : step);
  }
  const hybridSeconds = (performance.now() - started) / 1000;
  const compiledContinuationAccepted = verified.compiledContinuationAccepted === true;
  return Object.assign(verified, {
    selectionPolicy: compiledContinuationAccepted ? "margin-verified-pytorch-root-mlx-continuation-v1" : verificationProvider.backend === "mlx-shared-control" ? "margin-verified-mlx-shared-control-v1" : verificationProvider.backend === "mlx-control" ? "margin-verified-mlx-control-v1" : "margin-verified-pytorch-v1", selectedBackend: compiledContinuationAccepted ? "pytorch-root+mlx-continuation" : verificationProvider.backend, fallbackTriggered: true, fallbackReason: assessment.reason,
    fastPathMinimumMargin: assessment.minimumMargin, verificationMarginThreshold: marginThreshold, fastPathSeconds,
    ...(terminalPrefillSkipped ? { verificationPrefillSkippedTerminal: true } : {}),
    verificationSeconds: Math.max(0, hybridSeconds - fastPathSeconds), workerVerificationSeconds: verified.elapsedSeconds,
    hybridSeconds, elapsedSeconds: hybridSeconds, tokensPerSecond: verified.generatedTokenIds.length / hybridSeconds,
    fastPath: fast,
  });
}

function combineVerifiedPrefixWithCompiledContinuation(inputIds: readonly number[], verified: DirectReport, continuation: DirectReport, assessment: DirectMarginAssessment): DirectReport {
  const prefixLength = verified.generatedTokenIds.length;
  const continuationSteps = Array.isArray(continuation.steps) ? continuation.steps.map((step, index) => ({ ...step, step: prefixLength + index, contextLength: inputIds.length + prefixLength + index })) : [];
  const generatedTokenIds = [...verified.generatedTokenIds, ...continuation.generatedTokenIds];
  return {
    ...verified,
    generatedTokenIds,
    fullTokenIds: [...inputIds, ...generatedTokenIds],
    steps: [...(verified.steps ?? []), ...continuationSteps],
    ...(typeof continuation.terminalLogitsSha256 === "string" ? { terminalLogitsSha256: continuation.terminalLogitsSha256 } : {}),
    compiledContinuationAccepted: true,
    compiledContinuationMinimumMargin: assessment.minimumMargin,
    compiledContinuationTokenSteps: continuation.generatedTokenIds.length,
    compiledContinuationSeconds: continuation.elapsedSeconds,
    compiledContinuationSessionCacheHit: continuation.sessionCacheHit === true,
    compiledContinuationPrefixTokensReused: nonNegativeInteger(continuation.prefixTokensReused),
    compiledContinuationPrefillTokensComputed: nonNegativeInteger(continuation.prefillTokensComputed),
    trustedFastPathSteps: nonNegativeInteger(verified.trustedFastPathSteps) + continuation.generatedTokenIds.length,
    verificationDecoderStepsAvoided: nonNegativeInteger(verified.verificationDecoderStepsAvoided) + continuation.generatedTokenIds.length,
    verificationHeadPositionsAvoided: nonNegativeInteger(verified.verificationHeadPositionsAvoided) + continuation.generatedTokenIds.length,
  };
}

function streamingEventSensitiveMargin(value: unknown, marginThreshold: number): { step: number; tokenId?: number; margin: number } | undefined {
  if (typeof value !== "object" || value === null || !Number.isFinite(marginThreshold) || marginThreshold < 0) return undefined;
  const step = (value as { step?: unknown }).step;
  const tokenId = (value as { tokenId?: unknown }).tokenId;
  const top = normalizeTopLogits((value as { topLogits?: unknown }).topLogits);
  const margin = top.length >= 2 ? top[0]!.value - top[1]!.value : Number.NaN;
  return Number.isSafeInteger(step) && (step as number) >= 0 && Number.isFinite(margin) && margin <= marginThreshold ? { step: step as number, ...(Number.isSafeInteger(tokenId) && (tokenId as number) >= 0 ? { tokenId: tokenId as number } : {}), margin } : undefined;
}

export function shouldStartDirectVerificationPrefill(event: { step: number; tokenId?: number }, maxNewTokens: number, eosTokenId?: number): boolean {
  if (!Number.isSafeInteger(event.step) || event.step < 0 || !Number.isSafeInteger(maxNewTokens) || maxNewTokens < 1) throw new Error("Evento de prefill seletivo é inválido.");
  return event.step + 1 < maxNewTokens && (eosTokenId === undefined || event.tokenId !== eosTokenId);
}

class Gemma4PersistentComparisonWorker {
  readonly transport: PersistentJsonlWorker;
  constructor(options: Gemma4RealComparisonRunnerOptions) { this.transport = new PersistentJsonlWorker(options.python, [options.helper, "--source", options.source, "--serve-jsonl"], "Gemma 4 Transformers"); }
  get ready(): boolean { return this.transport.ready; }
  get initializationSeconds(): number | undefined { return this.transport.initializationSeconds; }
  whenReady(): Promise<void> { return this.transport.whenReady(); }
  compare(request: Gemma4RealComparisonRequest): Promise<unknown> { return this.transport.send(request); }
  compareTokens(request: Gemma4RealComparisonRequest, inputIds: readonly number[], eosTokenId?: number): Promise<unknown> { return this.transport.send({ ...request, prompt: undefined, inputIds: inputIds.join(","), ...(eosTokenId === undefined ? {} : { eosTokenId }) }); }
  encode(text: string, addSpecialTokens: boolean): Promise<{ tokenIds: number[] }> { return this.transport.send({ mode: "encode", text, addSpecialTokens }) as Promise<{ tokenIds: number[] }>; }
  decode(tokenIds: number[]): Promise<{ text: string }> { return this.transport.send({ mode: "decode", tokenIds }) as Promise<{ text: string }>; }
  close(): void { this.transport.close(); }
}

interface Gemma4TokenizerTransport {
  encode(text: string, addSpecialTokens: boolean): Promise<{ tokenIds: number[] }>;
  decode(tokenIds: number[]): Promise<{ text: string }>;
}

class Gemma4PersistentTokenizerWorker implements Gemma4TokenizerTransport {
  readonly transport: PersistentJsonlWorker;
  constructor(options: Gemma4RealComparisonRunnerOptions) { this.transport = new PersistentJsonlWorker(options.python, [options.tokenizerHelper ?? resolve("scripts/gemma4-tokenizer-jsonl.py"), "--source", options.source], "Gemma 4 tokenizer"); }
  get ready(): boolean { return this.transport.ready; }
  get initializationSeconds(): number | undefined { return this.transport.initializationSeconds; }
  whenReady(): Promise<void> { return this.transport.whenReady(); }
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
  whenReady(): Promise<void> { return this.#readyPromise; }
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

interface ComparisonReport { inputIds?: number[][]; baselineGeneratedTokenIds: number[]; steps?: Array<{ baselineToken?: number; baselineTopLogits?: unknown }>; performance?: { baselineSeconds?: number } }
interface DirectReport { generatedTokenIds: number[]; fullTokenIds: number[]; generatedText?: string; fullText?: string; tokensEqualBaseline?: boolean; firstDivergentStep?: number | null; terminalLogitsSha256?: string; selectiveVerification?: boolean; steps?: Array<{ forwardSeconds?: number; verificationSkipped?: boolean; [key: string]: unknown }>; [key: string]: unknown }
export interface DirectExecutionMetrics { inputTokens: number; generatedTokens: number; firstTokenForwardSeconds: number | null; firstTokenWallSeconds: number | null; decodeForwardSeconds: number; totalForwardSeconds: number; sessionCacheHit: boolean; prefixTokensReused: number; prefillTokensComputed: number; prefillAvoidedRate: number; baselineSpeedup: number | null; tokensEqualBaseline: boolean | null }
export interface DirectLogitStepAgreement { step: number; contextsEqualBeforeStep: boolean; baselineArgmaxToken: number | null; directArgmaxToken: number | null; baselineArgmaxLogitAbsError: number | null; greedyMarginAbsError: number | null; topK: number; topKOverlapCount: number; topKOverlapRate: number | null; topKCommonLogitMaxAbsError: number | null }
export interface DirectLogitAgreement { reportedSteps: number; measuredSteps: number; rootDivergences: number; postDivergenceSteps: number; selectedLogitMeasuredSteps: number; meanBaselineArgmaxLogitAbsError: number | null; maxBaselineArgmaxLogitAbsError: number | null; marginMeasuredSteps: number; meanGreedyMarginAbsError: number | null; maxGreedyMarginAbsError: number | null; meanTopKOverlapRate: number | null; maxTopKCommonLogitAbsError: number | null; steps: DirectLogitStepAgreement[] }
function arraysEqual(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function firstDivergence(left: readonly number[], right: readonly number[]): number | null { const length = Math.max(left.length, right.length); for (let index = 0; index < length; index += 1) if (left[index] !== right[index]) return index; return null; }
function elapsedSeconds(started: number): number { return (performance.now() - started) / 1000; }

export function attachDirectFallbackOutcome(direct: DirectReport, baselineTokenIds: readonly number[]): void {
  if (direct.fallbackTriggered !== true || !direct.fastPath || typeof direct.fastPath !== "object" || Array.isArray(direct.fastPath)) return;
  const fastTokenIds = (direct.fastPath as { generatedTokenIds?: unknown }).generatedTokenIds;
  if (!Array.isArray(fastTokenIds) || !fastTokenIds.every(Number.isSafeInteger)) return;
  const fastEqual = arraysEqual(fastTokenIds as number[], baselineTokenIds), selectedEqual = arraysEqual(direct.generatedTokenIds, baselineTokenIds), changed = !arraysEqual(fastTokenIds as number[], direct.generatedTokenIds);
  direct.fastPathTokensEqualBaseline = fastEqual;
  direct.fallbackChangedTokens = changed;
  direct.fallbackOutcome = fastEqual && !selectedEqual ? "worsened" : !fastEqual && selectedEqual ? "improved" : changed ? "changed-still-divergent" : selectedEqual ? "confirmed-equal" : "confirmed-divergent";
}

export function assertDirectFinalFormulaProgram(metadata: Record<string, unknown>, expected: Gemma4CompiledProgramStatus["finalFormulaMap"], runtime: Gemma4CompiledProgramStatus["finalFormulaRuntime"]): void {
  const program = metadata.finalFormulaProgram;
  if (!program || typeof program !== "object" || Array.isArray(program)) throw new Error("Worker compilado não autenticou final-formulas.json.");
  const value = program as Record<string, unknown>;
  if (value.execution !== "vectorized-shared-dag-output-program" || value.evaluator !== runtime.evaluator || value.runtimeContractSha256 !== runtime.sha256 || value.functions !== expected.functions || value.fileSha256 !== expected.sha256 || value.orderedRootsSha256 !== expected.orderedRootsSha256) throw new Error("Worker compilado publicou vínculo divergente para final-formulas.json ou seu runtime EVAL_EXACT_DAG.");
}

export function computeDirectExecutionMetrics(direct: DirectReport, inputTokens: number, baselineSeconds?: number, firstTokenWallSeconds?: number, comparisonAvailable = true): DirectExecutionMetrics {
  const forwardSeconds = Array.isArray(direct.steps) ? direct.steps.flatMap((step) => typeof step.forwardSeconds === "number" && Number.isFinite(step.forwardSeconds) && step.forwardSeconds >= 0 ? [step.forwardSeconds] : []) : [];
  const prefixTokensReused = nonNegativeInteger(direct.prefixTokensReused), prefillTokensComputed = nonNegativeInteger(direct.prefillTokensComputed);
  const prefillDomain = prefixTokensReused + prefillTokensComputed;
  const elapsed = typeof direct.elapsedSeconds === "number" && Number.isFinite(direct.elapsedSeconds) && direct.elapsedSeconds > 0 ? direct.elapsedSeconds : undefined;
  return {
    inputTokens, generatedTokens: direct.generatedTokenIds.length,
    firstTokenForwardSeconds: forwardSeconds[0] ?? null,
    firstTokenWallSeconds: firstTokenWallSeconds !== undefined && Number.isFinite(firstTokenWallSeconds) && firstTokenWallSeconds >= 0 ? firstTokenWallSeconds : null,
    decodeForwardSeconds: forwardSeconds.slice(1).reduce((sum, value) => sum + value, 0),
    totalForwardSeconds: forwardSeconds.reduce((sum, value) => sum + value, 0),
    sessionCacheHit: direct.sessionCacheHit === true, prefixTokensReused, prefillTokensComputed,
    prefillAvoidedRate: prefillDomain === 0 ? 0 : prefixTokensReused / prefillDomain,
    baselineSpeedup: elapsed !== undefined && baselineSeconds !== undefined && Number.isFinite(baselineSeconds) && baselineSeconds >= 0 ? baselineSeconds / elapsed : null,
    tokensEqualBaseline: comparisonAvailable ? direct.tokensEqualBaseline === true : null,
  };
}

function nonNegativeInteger(value: unknown): number { return Number.isSafeInteger(value) && (value as number) >= 0 ? value as number : 0; }
function finiteNonNegative(value: unknown): number { return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0; }
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

async function resolveComparisonInput(worker: Gemma4TokenizerTransport, request: Gemma4RealComparisonRequest, sessions: Map<number, SessionInput>): Promise<number[]> {
  const mode = request.conversationMode ?? "raw";
  if (request.continueSession) {
    const session = sessions.get(request.sessionId!); if (!session) throw new Error("Sessão de chat inexistente ou expirada.");
    if (session.mode !== mode) throw new Error("conversationMode não pode mudar durante uma sessão.");
    const suffix = await worker.encode(mode === "chat" ? formatGemma4ChatTurn(request.prompt, false, session.assistantTurnClosed) : request.prompt, false); validateEncodedIds(suffix.tokenIds);
    return [...session.tokenIds, ...suffix.tokenIds];
  }
  if (request.sessionId !== undefined) sessions.delete(request.sessionId);
  const encoded = await worker.encode(mode === "chat" ? formatGemma4ChatTurn(request.prompt, true) : request.prompt, true); validateEncodedIds(encoded.tokenIds); return [...encoded.tokenIds];
}
function formatGemma4ChatTurn(prompt: string, first: boolean, previousTurnClosed = false): string { return `${first ? "" : previousTurnClosed ? "\n" : "<turn|>\n"}<|turn>user\n${prompt.trim()}<turn|>\n<|turn>model\n`; }
function chatStopToken(request: Gemma4RealComparisonRequest): number | undefined { return request.conversationMode === "chat" ? GEMMA4_CHAT_EOT_TOKEN_ID : undefined; }
function validateEncodedIds(value: unknown): asserts value is number[] { if (!Array.isArray(value) || value.length < 1 || value.some((token) => !Number.isSafeInteger(token) || token < 0)) throw new Error("Tokenizer retornou inputIds inválidos."); }
function rememberSession(sessions: Map<number, SessionInput>, sessionId: number, fullTokenIds: readonly number[], mode: "raw" | "chat" = "raw"): void {
  validateEncodedIds(fullTokenIds); sessions.delete(sessionId); sessions.set(sessionId, { tokenIds: [...fullTokenIds], mode, assistantTurnClosed: mode === "chat" && fullTokenIds.at(-1) === GEMMA4_CHAT_EOT_TOKEN_ID });
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
  if (request.conversationMode === "chat" && (request.prompt.includes("<|turn>") || request.prompt.includes("<turn|>"))) throw new Error("prompt de chat não pode conter os delimitadores reservados <|turn> ou <turn|>.");
  if (request.measurementSchedule !== undefined && request.measurementSchedule !== "isolated" && request.measurementSchedule !== "parallel") throw new Error("measurementSchedule deve ser isolated ou parallel.");
  if (request.captureLayerHidden !== undefined && typeof request.captureLayerHidden !== "boolean") throw new Error("captureLayerHidden deve ser booleano.");
  if (request.captureLayerStages !== undefined && (!Number.isSafeInteger(request.captureLayerStages) || request.captureLayerStages < 0 || request.captureLayerStages >= 42)) throw new Error("captureLayerStages deve estar entre 0 e 41.");
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

class DecodedTokenNdjsonStream {
  readonly tokenIds: number[] = [];
  #decodedText = "";
  #pending: Promise<void> = Promise.resolve();
  constructor(readonly tokenizer: Gemma4TokenizerTransport, readonly response: ServerResponse) {}
  push(event: unknown, envelope: Record<string, unknown>): void {
    if (!event || typeof event !== "object" || Array.isArray(event) || !Number.isSafeInteger((event as { tokenId?: unknown }).tokenId) || ((event as { tokenId: number }).tokenId) < 0) throw new Error("Evento streaming direto não contém tokenId válido.");
    this.tokenIds.push((event as { tokenId: number }).tokenId); const prefix = [...this.tokenIds];
    this.#pending = this.#pending.then(async () => {
      const decoded = await this.tokenizer.decode(prefix);
      if (typeof decoded.text !== "string") throw new Error("Tokenizer não retornou texto streaming válido.");
      const textReset = !decoded.text.startsWith(this.#decodedText), deltaText = textReset ? decoded.text : decoded.text.slice(this.#decodedText.length);
      this.#decodedText = decoded.text;
      writeNdjson(this.response, { ...envelope, event, generatedTokenIds: prefix, generatedText: decoded.text, deltaText, textReset });
    });
  }
  write(value: unknown): void { this.#pending = this.#pending.then(() => writeNdjson(this.response, value)); }
  flush(): Promise<void> { return this.#pending; }
}

export function parseGemma4RealServerOptions(arguments_: readonly string[]): Gemma4RealComparisonRunnerOptions & { port: number; host: string } {
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index], value = arguments_[index + 1];
    if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
    values.set(flag, value);
  }
  const known = new Set(["--source", "--compiled-bundle", "--python", "--helper", "--tokenizer-helper", "--port", "--host", "--literal-artifact", "--binary-pool", "--direct-worker", "--direct-linear-helper", "--direct-mlx-helper", "--direct-linear-backend", "--direct-mlx-head-quantization", "--direct-mlx-decoder-quantization", "--direct-mlx-decoder-quantization-layers", "--direct-fused-mlp", "--direct-fused-ffn", "--direct-fused-decoder-layer", "--direct-fused-decoder-stack", "--direct-fused-ple", "--direct-fused-ple-prelude", "--direct-fused-token-forward", "--direct-resident-generation", "--direct-final-head", "--direct-final-head-read-mib", "--direct-native-attention", "--direct-fused-attention", "--direct-threads", "--direct-max-read-mib", "--direct-verification-margin", "--direct-verification-backend"]); for (const key of values.keys()) if (!known.has(key)) throw new Error(`Flag desconhecida: ${key}.`);
  const port = Number(values.get("--port") ?? "8787"); if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) throw new Error("--port inválido.");
  const compiledBundle = resolve(values.get("--compiled-bundle") ?? "artifacts/gemma4-compiled-global-runtime-bundle");
  const bundleReady = existsSync(join(compiledBundle, "constants.literal.json")) && existsSync(join(compiledBundle, "model.safetensors")) && existsSync(join(compiledBundle, "tokenizer.json")) && existsSync(join(compiledBundle, "config.json")) && existsSync(join(compiledBundle, "manifest.json")) && existsSync(join(compiledBundle, "global-formulas.ssa.json")) && existsSync(join(compiledBundle, "final-formulas.json")) && existsSync(join(compiledBundle, "final-formulas.runtime.json")) && existsSync(join(compiledBundle, "vectorized-real-lowering.json"));
  if (values.has("--compiled-bundle") && !bundleReady) throw new Error("--compiled-bundle não contém constants.literal.json, model.safetensors, tokenizer.json, config.json, manifest.json, global-formulas.ssa.json, final-formulas.json, final-formulas.runtime.json e vectorized-real-lowering.json.");
  const source = resolve(values.get("--source") ?? (bundleReady ? compiledBundle : "gemma-4-E4B-dense")), inferredLiteral = join(source, "constants.literal.json");
  const literalArtifact = values.get("--literal-artifact") ? resolve(values.get("--literal-artifact")!) : existsSync(inferredLiteral) ? inferredLiteral : (values.has("--compiled-bundle") || !values.has("--source")) && bundleReady ? join(compiledBundle, "constants.literal.json") : undefined;
  const artifactPool = literalArtifact ? dirname(literalArtifact) : undefined;
  const binaryPool = values.get("--binary-pool") ? resolve(values.get("--binary-pool")!) : artifactPool && existsSync(join(artifactPool, "model.safetensors")) ? artifactPool : literalArtifact && existsSync(join(source, "model.safetensors")) ? source : undefined;
  if ((literalArtifact === undefined) !== (binaryPool === undefined)) throw new Error("Backend direto requer --literal-artifact e --binary-pool juntos.");
  const compiledProgram = bundleReady ? readCompiledProgramStatus(compiledBundle) : undefined;
  const directLinearBackend = values.get("--direct-linear-backend") ?? "mlx";
  if (directLinearBackend !== "pytorch" && directLinearBackend !== "mlx") throw new Error("--direct-linear-backend deve ser pytorch ou mlx.");
  const directMlxHeadQuantization = values.get("--direct-mlx-head-quantization") ?? "off";
  if (directMlxHeadQuantization !== "off" && directMlxHeadQuantization !== "q8" && directMlxHeadQuantization !== "q8-shortlist" && directMlxHeadQuantization !== "q4") throw new Error("--direct-mlx-head-quantization deve ser off, q8, q8-shortlist ou q4.");
  if (directLinearBackend !== "mlx" && directMlxHeadQuantization !== "off") throw new Error("--direct-mlx-head-quantization requer backend MLX.");
  const directMlxDecoderQuantization = values.get("--direct-mlx-decoder-quantization") ?? (directLinearBackend === "mlx" ? "q8-ffn-gate-up" : "off");
  if (directMlxDecoderQuantization !== "off" && directMlxDecoderQuantization !== "q8-ffn" && directMlxDecoderQuantization !== "q8-ffn-gate-up" && directMlxDecoderQuantization !== "q8-ffn-gate-up-attention" && directMlxDecoderQuantization !== "q8-ffn-gate-up-down" && directMlxDecoderQuantization !== "q4-ffn-gate-up" && directMlxDecoderQuantization !== "q8-ffn-gate-up-first-half" && directMlxDecoderQuantization !== "q8-ffn-gate-up-last-half" && directMlxDecoderQuantization !== "q8-ffn-down" && directMlxDecoderQuantization !== "q8-attention" && directMlxDecoderQuantization !== "q8-all") throw new Error("--direct-mlx-decoder-quantization possui modo inválido.");
  if (directLinearBackend !== "mlx" && directMlxDecoderQuantization !== "off") throw new Error("--direct-mlx-decoder-quantization requer backend MLX.");
  const directMlxDecoderQuantizationLayers = values.has("--direct-mlx-decoder-quantization-layers") ? normalizeGemma4DecoderQuantizationLayers(values.get("--direct-mlx-decoder-quantization-layers")!) : directLinearBackend === "mlx" && directMlxDecoderQuantization === "q8-ffn-gate-up" ? "0-20" : undefined;
  if (directMlxDecoderQuantizationLayers !== undefined && directMlxDecoderQuantization !== "q8-ffn-gate-up" && directMlxDecoderQuantization !== "q8-ffn-gate-up-down" && directMlxDecoderQuantization !== "q4-ffn-gate-up") throw new Error("--direct-mlx-decoder-quantization-layers requer q8-ffn-gate-up, q8-ffn-gate-up-down ou q4-ffn-gate-up.");
  const verificationMarginValue = values.get("--direct-verification-margin") ?? (directLinearBackend === "mlx" ? directMlxDecoderQuantization === "q8-ffn-gate-up-down" ? "0.125" : directMlxDecoderQuantization === "q8-ffn-gate-up" && directMlxDecoderQuantizationLayers === "0-20" ? "0" : "0.25" : "off");
  const directVerificationMargin = verificationMarginValue === "off" ? undefined : Number(verificationMarginValue);
  if (directVerificationMargin !== undefined && (!Number.isFinite(directVerificationMargin) || directVerificationMargin < 0)) throw new Error("--direct-verification-margin deve ser off ou número finito não negativo.");
  if (directLinearBackend !== "mlx" && directVerificationMargin !== undefined) throw new Error("--direct-verification-margin requer backend mlx.");
  const directVerificationBackend = values.get("--direct-verification-backend") ?? (directMlxDecoderQuantization === "q8-ffn-gate-up-down" ? "mlx-shared-control" : "pytorch");
  if (directVerificationBackend !== "pytorch" && directVerificationBackend !== "mlx-control" && directVerificationBackend !== "mlx-shared-control") throw new Error("--direct-verification-backend deve ser pytorch, mlx-control ou mlx-shared-control.");
  if ((directVerificationBackend === "mlx-control" || directVerificationBackend === "mlx-shared-control") && directLinearBackend !== "mlx") throw new Error(`--direct-verification-backend ${directVerificationBackend} requer backend mlx.`);
  if (directVerificationBackend === "mlx-shared-control" && directMlxDecoderQuantization !== "q8-ffn-gate-up-down") throw new Error("--direct-verification-backend mlx-shared-control requer decoder q8-ffn-gate-up-down.");
  const directThreads = Number(values.get("--direct-threads") ?? "8"), directMaxReadMiB = Number(values.get("--direct-max-read-mib") ?? "16"), directFinalHeadReadMiB = Number(values.get("--direct-final-head-read-mib") ?? (directLinearBackend === "pytorch" ? "32" : String(directMaxReadMiB)));
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
  const directFusedDecoderStack = values.get("--direct-fused-decoder-stack") ?? (directLinearBackend === "mlx" ? "real" : "native-bf16");
  if (directFusedDecoderStack !== "off" && directFusedDecoderStack !== "real" && directFusedDecoderStack !== "native-bf16" && directFusedDecoderStack !== "native-bf16-ple") throw new Error("--direct-fused-decoder-stack deve ser off, real, native-bf16 ou native-bf16-ple.");
  if (directLinearBackend === "mlx" && directFusedDecoderStack === "native-bf16-ple") throw new Error("--direct-fused-decoder-stack native-bf16-ple requer backend pytorch.");
  if (directLinearBackend !== "mlx" && directFusedDecoderStack === "real") throw new Error("--direct-fused-decoder-stack real requer backend mlx.");
  const directFusedPle = values.get("--direct-fused-ple") ?? (directLinearBackend === "pytorch" ? "bf16" : "off");
  if (directFusedPle !== "off" && directFusedPle !== "bf16" && directFusedPle !== "real") throw new Error("--direct-fused-ple deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directFusedPle !== "off") throw new Error("--direct-fused-ple requer backend pytorch.");
  const directFusedPlePrelude = values.get("--direct-fused-ple-prelude") ?? "bf16";
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
    helper: resolve(values.get("--helper") ?? "scripts/gemma4-real-differential.py"), tokenizerHelper: resolve(values.get("--tokenizer-helper") ?? "scripts/gemma4-tokenizer-jsonl.py"), port, host: values.get("--host") ?? "127.0.0.1",
    directMlxDecoderQuantization,
    ...(directMlxDecoderQuantizationLayers === undefined ? {} : { directMlxDecoderQuantizationLayers }),
    ...(compiledProgram ? { compiledProgram } : {}),
    ...(literalArtifact && binaryPool ? { literalArtifact, binaryPool, ...(compiledProgram?.runtimeIndex && literalArtifact === join(compiledBundle, compiledProgram.constantPool.file) ? { directArtifactIndex: join(compiledBundle, compiledProgram.runtimeIndex.file), directArtifactIndexSha256: compiledProgram.runtimeIndex.sha256, directArtifactSha256: compiledProgram.runtimeIndex.constantPoolSha256 } : {}), directWorker: resolve(values.get("--direct-worker") ?? "dist/src/gemma4-paged-runtime-worker-cli.js"), directLinearHelper: resolve(values.get("--direct-linear-helper") ?? "scripts/gemma4-paged-linear-worker.py"), directMlxHelper: resolve(values.get("--direct-mlx-helper") ?? "scripts/gemma4-mlx-linear-worker.py"), directLinearBackend, directMlxHeadQuantization, directFusedMlp, directFusedFfn, directFusedDecoderLayer, directFusedDecoderStack, directFusedPle, directFusedPlePrelude, directFusedTokenForward, directResidentGeneration, directFinalHead, directNativeAttention, directFusedAttention, directThreads, directMaxReadMiB, directFinalHeadReadMiB, directVerificationBackend, ...(directVerificationMargin === undefined ? {} : { directVerificationMargin }) } : {}),
  };
}

function readCompiledProgramStatus(bundle: string): Gemma4CompiledProgramStatus {
  const manifest = JSON.parse(readFileSync(join(bundle, "manifest.json"), "utf8")) as {
    kind?: unknown; schemaVersion?: unknown; execution?: unknown; runtimeLowering?: { engine?: unknown; directlyExecutesGlobalFormula?: unknown; executesPersistedLoweringPlan?: unknown; plan?: unknown; functionBindingsSha256?: unknown; outputBindingsSha256?: unknown; standaloneSsaOutputsSha256?: unknown; outputFunctions?: unknown; realSimplifiedProgramSha256?: unknown; globalFormulaRole?: unknown; compiledOutputProgram?: unknown };
    formula?: { family?: unknown; dimension?: unknown; root?: unknown; expressionNodes?: unknown; inputTensor?: unknown; inputLength?: unknown; file?: unknown };
    globalProgram?: { file?: unknown; terminalLogits?: unknown; constantPool?: unknown }; finalFormulaMap?: { file?: unknown; functions?: unknown; inputTensor?: unknown; evaluator?: unknown; globalFormulaSha256?: unknown; orderedRootsSha256?: unknown }; finalFormulaRuntime?: { file?: unknown; sha256?: unknown; schemaVersion?: unknown; evaluator?: unknown }; runtimeIndex?: { file?: unknown; schemaVersion?: unknown; constantPoolSha256?: unknown; integrityRootSha256?: unknown }; files?: Array<{ role?: unknown; file?: unknown; sha256?: unknown }>;
  };
  const globalProgram = manifest.globalProgram, formula = manifest.formula, finalFormulaMap = manifest.finalFormulaMap, finalFormulaRuntime = manifest.finalFormulaRuntime, formulaFile = manifest.files?.find((entry) => entry.role === "formula-graph"), globalFile = manifest.files?.find((entry) => entry.role === "global-formulas"), finalFormulaFile = manifest.files?.find((entry) => entry.role === "final-formulas"), finalFormulaRuntimeFile = manifest.files?.find((entry) => entry.role === "final-formula-runtime"), constantFile = manifest.files?.find((entry) => entry.role === "constant-pool"), planFile = manifest.files?.find((entry) => entry.role === "vectorized-real-lowering"), runtimeIndexFile = manifest.files?.find((entry) => entry.role === "literal-runtime-index"), declaredRuntimeIndex = manifest.runtimeIndex;
  const runtimeIndexValid = declaredRuntimeIndex === undefined && runtimeIndexFile === undefined || declaredRuntimeIndex?.file === runtimeIndexFile?.file && (declaredRuntimeIndex?.schemaVersion === 1 || declaredRuntimeIndex?.schemaVersion === 2) && declaredRuntimeIndex?.constantPoolSha256 === constantFile?.sha256 && typeof declaredRuntimeIndex?.integrityRootSha256 === "string" && /^[0-9a-f]{64}$/.test(declaredRuntimeIndex.integrityRootSha256) && typeof runtimeIndexFile?.sha256 === "string" && /^[0-9a-f]{64}$/.test(runtimeIndexFile.sha256) && existsSync(join(bundle, runtimeIndexFile.file as string));
  if (manifest.kind !== "gemma4-compiled-shared-dag-bundle" || manifest.schemaVersion !== 4 || !runtimeIndexValid || manifest.execution !== "compiled-parametric-output-program-runtime" ||
    manifest.runtimeLowering?.engine !== "mlx-f32-real-decoder-stack-v1" || manifest.runtimeLowering.directlyExecutesGlobalFormula !== true || manifest.runtimeLowering.executesPersistedLoweringPlan !== true || manifest.runtimeLowering.globalFormulaRole !== "compiled-executable-shared-dag" ||
    manifest.runtimeLowering.plan !== "vectorized-real-lowering.json" || planFile?.file !== "vectorized-real-lowering.json" || typeof planFile.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(planFile.sha256) ||
    typeof manifest.runtimeLowering.functionBindingsSha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.runtimeLowering.functionBindingsSha256) ||
    typeof manifest.runtimeLowering.outputBindingsSha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.runtimeLowering.outputBindingsSha256) ||
    typeof manifest.runtimeLowering.standaloneSsaOutputsSha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.runtimeLowering.standaloneSsaOutputsSha256) || !Number.isSafeInteger(manifest.runtimeLowering.outputFunctions) || (manifest.runtimeLowering.outputFunctions as number) < 1 ||
    typeof manifest.runtimeLowering.realSimplifiedProgramSha256 !== "string" || !/^[0-9a-f]{64}$/.test(manifest.runtimeLowering.realSimplifiedProgramSha256) ||
    formula?.family !== "terminal_logit" || !Number.isSafeInteger(formula.dimension) || (formula.dimension as number) < 0 || typeof formula.root !== "string" || !/^sha256:[0-9a-f]{64}$/.test(formula.root) || !Number.isSafeInteger(formula.expressionNodes) || (formula.expressionNodes as number) < 1 || formula.inputTensor !== "x" || !Number.isSafeInteger(formula.inputLength) || (formula.inputLength as number) < 1 || formula.file !== formulaFile?.file || typeof formulaFile?.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(formulaFile.sha256) ||
    globalProgram?.file !== globalFile?.file || globalProgram?.constantPool !== constantFile?.file || !Number.isSafeInteger(globalProgram?.terminalLogits) ||
    finalFormulaMap?.file !== "final-formulas.json" || finalFormulaMap.file !== finalFormulaFile?.file || finalFormulaMap.inputTensor !== "x" || finalFormulaMap.evaluator !== "BF16_RNE(EVAL_EXACT_DAG(root,x))" || !Number.isSafeInteger(finalFormulaMap.functions) || finalFormulaMap.globalFormulaSha256 !== globalFile?.sha256 || typeof finalFormulaMap.orderedRootsSha256 !== "string" || !/^[0-9a-f]{64}$/.test(finalFormulaMap.orderedRootsSha256) || typeof finalFormulaFile?.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(finalFormulaFile.sha256) || !existsSync(join(bundle, finalFormulaFile.file as string)) ||
    finalFormulaRuntime?.file !== "final-formulas.runtime.json" || finalFormulaRuntime.file !== finalFormulaRuntimeFile?.file || finalFormulaRuntime.schemaVersion !== 1 || finalFormulaRuntime.evaluator !== "EVAL_EXACT_DAG" || typeof finalFormulaRuntime.sha256 !== "string" || finalFormulaRuntime.sha256 !== finalFormulaRuntimeFile?.sha256 || !/^[0-9a-f]{64}$/.test(finalFormulaRuntime.sha256) || !existsSync(join(bundle, finalFormulaRuntime.file)) ||
    typeof globalFile?.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(globalFile.sha256) || typeof constantFile?.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(constantFile.sha256)) {
    throw new Error("Manifesto do bundle compilado não declara honestamente a fórmula global e o lowering executado.");
  }
  const planBytes = readFileSync(join(bundle, "vectorized-real-lowering.json"));
  if (createHash("sha256").update(planBytes).digest("hex") !== planFile.sha256) throw new Error("Plano de lowering do bundle diverge do SHA-256 declarado no manifesto.");
  const plan = JSON.parse(planBytes.toString("utf8")) as unknown;
  validateGemma4VectorizedRealLoweringPlan(plan);
  if (createHash("sha256").update(readFileSync(join(bundle, finalFormulaFile!.file as string))).digest("hex") !== finalFormulaFile!.sha256) throw new Error("Mapa final de fórmulas diverge do SHA-256 declarado no manifesto.");
  const runtimeBytes = readFileSync(join(bundle, finalFormulaRuntime!.file as string));
  if (createHash("sha256").update(runtimeBytes).digest("hex") !== finalFormulaRuntime!.sha256) throw new Error("Runtime EVAL_EXACT_DAG diverge do SHA-256 declarado no manifesto.");
  const runtimeContract = JSON.parse(runtimeBytes.toString("utf8")) as unknown; validateGemma4FinalFormulaRuntime(runtimeContract);
  let terminalLogitOffset = 0; const outputFamilies = Object.entries(plan.contract.source.outputFamilies), terminalLogitFamily = plan.contract.source.outputFamilies.terminal_logit;
  for (const [name, family] of outputFamilies) { if (name === "terminal_logit") break; terminalLogitOffset += family.dimensions; }
  if (plan.contract.functionBindingsSha256 !== manifest.runtimeLowering.functionBindingsSha256 || plan.contract.source.outputBindingsSha256 !== manifest.runtimeLowering.outputBindingsSha256 ||
    plan.contract.source.standaloneSsaOutputsSha256 !== manifest.runtimeLowering.standaloneSsaOutputsSha256 || plan.contract.source.outputFunctions !== manifest.runtimeLowering.outputFunctions ||
    plan.contract.source.outputFamilies.terminal_logit?.dimensions !== globalProgram!.terminalLogits || finalFormulaMap!.functions !== terminalLogitFamily?.dimensions || plan.contract.source.realSimplifiedProgramSha256 !== manifest.runtimeLowering.realSimplifiedProgramSha256 ||
    JSON.stringify(plan.contract.compiledOutputProgram) !== JSON.stringify(manifest.runtimeLowering.compiledOutputProgram) ||
    runtimeContract.input.length !== formula!.inputLength || runtimeContract.output.functions !== finalFormulaMap!.functions || runtimeContract.artifacts.formulaMap.sha256 !== finalFormulaFile!.sha256 || runtimeContract.artifacts.formulaMap.orderedRootsSha256 !== finalFormulaMap!.orderedRootsSha256 || runtimeContract.artifacts.globalSsa.sha256 !== globalFile!.sha256 || runtimeContract.artifacts.globalSsa.standaloneOutputsSha256 !== manifest.runtimeLowering.standaloneSsaOutputsSha256 || runtimeContract.artifacts.constantPool.sha256 !== constantFile!.sha256 || runtimeContract.artifacts.loweringPlan.sha256 !== planFile!.sha256 || runtimeContract.artifacts.loweringPlan.functionBindingsSha256 !== manifest.runtimeLowering.functionBindingsSha256 || runtimeContract.artifacts.loweringPlan.outputBindingsSha256 !== manifest.runtimeLowering.outputBindingsSha256 || runtimeContract.artifacts.loweringPlan.realSimplifiedProgramSha256 !== manifest.runtimeLowering.realSimplifiedProgramSha256 || JSON.stringify(runtimeContract.execution.compiledOutputProgram) !== JSON.stringify(plan.contract.compiledOutputProgram) ||
    (manifest.runtimeIndex && plan.contract.source.artifactIntegritySha256 !== manifest.runtimeIndex.integrityRootSha256)) {
    throw new Error("Plano de lowering do bundle diverge dos compromissos declarados no manifesto.");
  }
  return {
    bundle, execution: manifest.execution, formulaSemantics: "gemma4-exact-real-simplified-v1",
    globalFormula: { file: globalFile.file as string, sha256: globalFile.sha256, terminalLogits: globalProgram!.terminalLogits as number, role: "compiled-executable-shared-dag" },
    finalFormulaMap: { file: finalFormulaFile!.file as string, sha256: finalFormulaFile!.sha256 as string, functions: finalFormulaMap!.functions as number, inputTensor: "x", evaluator: "BF16_RNE(EVAL_EXACT_DAG(root,x))", orderedRootsSha256: finalFormulaMap!.orderedRootsSha256 as string },
    finalFormulaRuntime: { file: "final-formulas.runtime.json", sha256: finalFormulaRuntime!.sha256 as string, schemaVersion: 1, evaluator: "EVAL_EXACT_DAG" },
    constantPool: { file: constantFile.file as string, sha256: constantFile.sha256 },
    exampleFormula: { key: `calc_final_${formula.dimension}`, family: formula.family, dimension: formula.dimension as number, root: formula.root as string, expressionNodes: formula.expressionNodes as number, inputTensor: "x", inputLength: formula.inputLength as number, file: formula.file as string, sha256: formulaFile.sha256 as string },
    terminalLogitOutputs: { offset: terminalLogitOffset, dimensions: terminalLogitFamily!.dimensions, operationId: terminalLogitFamily!.operationId, finalQuantization: terminalLogitFamily!.finalQuantization },
    ...(manifest.runtimeIndex ? { runtimeIndex: { file: manifest.runtimeIndex.file as string, schemaVersion: manifest.runtimeIndex.schemaVersion as 1 | 2, sha256: runtimeIndexFile!.sha256 as string, constantPoolSha256: manifest.runtimeIndex.constantPoolSha256 as string, integrityRootSha256: manifest.runtimeIndex.integrityRootSha256 as string } } : {}),
    directRuntime: { engine: "mlx-f32-real-decoder-stack-v1", directlyExecutesGlobalFormula: true, executesPersistedLoweringPlan: true, compiledOutputProgram: plan.contract.compiledOutputProgram, plan: { file: planFile.file as string, sha256: planFile.sha256, functionBindingsSha256: manifest.runtimeLowering.functionBindingsSha256, outputBindingsSha256: manifest.runtimeLowering.outputBindingsSha256, standaloneSsaOutputsSha256: manifest.runtimeLowering.standaloneSsaOutputsSha256, outputFunctions: manifest.runtimeLowering.outputFunctions as number, realSimplifiedProgramSha256: manifest.runtimeLowering.realSimplifiedProgramSha256 } },
  };
}
