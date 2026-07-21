import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gemma4RealCompareHtml } from "./gemma4-real-compare-ui.js";

export interface Gemma4RealComparisonRequest { prompt: string; maxNewTokens: number; threads?: number; precision?: "f32" | "f64"; roundingPolicy?: "none" | "layer-bf16" | "operation-bf16" }
export interface Gemma4RealComparisonRunnerOptions {
  source: string; python: string; helper: string;
  literalArtifact?: string; binaryPool?: string; directWorker?: string; directLinearHelper?: string; directMlxHelper?: string; directLinearBackend?: "pytorch" | "mlx"; directFusedMlp?: "off" | "bf16" | "real"; directFusedPle?: "off" | "bf16" | "real"; directFusedPlePrelude?: "off" | "bf16" | "real"; directFinalHead?: "f32" | "native-bf16" | "native-bf16-whole"; directNativeAttention?: "off" | "bf16" | "real"; directFusedAttention?: "off" | "bf16" | "real"; directThreads?: number; directMaxReadMiB?: number;
}

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
  const worker = new Gemma4PersistentComparisonWorker(options);
  const direct = options.literalArtifact && options.binaryPool
    ? new PersistentJsonlWorker(process.execPath, [options.directWorker ?? resolve("dist/src/gemma4-paged-runtime-worker-cli.js"), "--artifact", options.literalArtifact, "--binary-pool", options.binaryPool, "--python", options.python, "--linear-helper", options.directLinearHelper ?? resolve("scripts/gemma4-paged-linear-worker.py"), "--mlx-helper", options.directMlxHelper ?? resolve("scripts/gemma4-mlx-linear-worker.py"), "--linear-backend", options.directLinearBackend ?? "pytorch", "--fused-mlp", options.directFusedMlp ?? "real", "--fused-ple", options.directFusedPle ?? ((options.directLinearBackend ?? "pytorch") === "pytorch" ? "bf16" : "off"), "--fused-ple-prelude", options.directFusedPlePrelude ?? "off", "--final-head", options.directFinalHead ?? ((options.directLinearBackend ?? "pytorch") === "pytorch" ? "native-bf16" : "f32"), "--native-attention", options.directNativeAttention ?? ((options.directLinearBackend ?? "pytorch") === "pytorch" ? "real" : "off"), "--fused-attention", options.directFusedAttention ?? ((options.directLinearBackend ?? "pytorch") === "pytorch" ? "bf16" : "off"), "--threads", String(options.directThreads ?? 8), "--max-read-mib", String(options.directMaxReadMiB ?? 16)], "Gemma 4 literal direto")
    : undefined;
  const server = createServer(async (request, response) => {
    try {
      if (request.method === "GET" && request.url === "/") return send(response, 200, "text/html; charset=utf-8", gemma4RealCompareHtml);
      if (request.method === "GET" && request.url === "/api/status") return json(response, 200, { ready: worker.ready && (!direct || direct.ready), source: options.source, runtime: "persistent-jsonl", initializationSeconds: worker.initializationSeconds, direct: direct ? { enabled: true, ready: direct.ready, initializationSeconds: direct.initializationSeconds, ...direct.readyMetadata } : { enabled: false } });
      if (request.method === "POST" && request.url === "/api/compare") {
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        const report = await worker.compare(body) as ComparisonReport;
        if (!direct) return json(response, 200, report);
        const inputIds = report.inputIds?.[0]; if (!inputIds) throw new Error("Comparador original não retornou inputIds para o backend direto.");
        const directReport = await direct.send({ inputIds, maxNewTokens: body.maxNewTokens }) as DirectReport;
        const [generated, full] = await Promise.all([worker.decode(directReport.generatedTokenIds), worker.decode(directReport.fullTokenIds)]);
        directReport.generatedText = generated.text; directReport.fullText = full.text;
        directReport.tokensEqualBaseline = arraysEqual(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
        directReport.firstDivergentStep = firstDivergence(directReport.generatedTokenIds, report.baselineGeneratedTokenIds);
        return json(response, 200, { ...report, direct: directReport });
      }
      return json(response, 404, { error: "Rota não encontrada." });
    } catch (error) {
      return json(response, 400, { error: error instanceof Error ? error.message : String(error) });
    }
  });
  server.once("close", () => { worker.close(); direct?.close(); });
  return server;
}

class Gemma4PersistentComparisonWorker {
  readonly transport: PersistentJsonlWorker;
  constructor(options: Gemma4RealComparisonRunnerOptions) { this.transport = new PersistentJsonlWorker(options.python, [options.helper, "--source", options.source, "--serve-jsonl"], "Gemma 4 Transformers"); }
  get ready(): boolean { return this.transport.ready; }
  get initializationSeconds(): number | undefined { return this.transport.initializationSeconds; }
  compare(request: Gemma4RealComparisonRequest): Promise<unknown> { return this.transport.send(request); }
  decode(tokenIds: number[]): Promise<{ text: string }> { return this.transport.send({ mode: "decode", tokenIds }) as Promise<{ text: string }>; }
  close(): void { this.transport.close(); }
}

class PersistentJsonlWorker {
  readonly child: ChildProcessWithoutNullStreams;
  readonly pending = new Map<number, { accept(value: unknown): void; reject(error: Error): void }>();
  ready = false; initializationSeconds?: number; readyMetadata: Record<string, unknown> = {};
  #nextId = 1; #stdout = ""; #readyAccept!: () => void; #readyReject!: (error: Error) => void; readonly #readyPromise: Promise<void>;
  constructor(command: string, arguments_: string[], readonly label: string) {
    this.#readyPromise = new Promise<void>((accept, reject) => { this.#readyAccept = accept; this.#readyReject = reject; });
    this.child = spawn(command, arguments_, { stdio: ["pipe", "pipe", "pipe"] });
    this.child.stdout.setEncoding("utf8"); this.child.stdout.on("data", (chunk: string) => this.#consume(chunk));
    const errors: Buffer[] = []; let errorBytes = 0; this.child.stderr.on("data", (chunk: Buffer) => { if (errorBytes < 1024 * 1024) { errors.push(chunk); errorBytes += chunk.length; } });
    this.child.once("error", (error) => this.#fail(error));
    this.child.once("close", (code) => this.#fail(new Error(`${label} encerrou com código ${code}: ${Buffer.concat(errors).toString("utf8").trim()}`)));
  }
  async send(payload: object): Promise<unknown> { await this.#readyPromise; const id = this.#nextId++; const result = new Promise<unknown>((accept, reject) => this.pending.set(id, { accept, reject })); this.child.stdin.write(`${JSON.stringify({ id, ...payload })}\n`); return result; }
  close(): void { if (!this.child.killed) this.child.kill("SIGTERM"); }
  #consume(chunk: string): void {
    this.#stdout += chunk; if (this.#stdout.length > 64 * 1024 * 1024) return this.#fail(new Error(`${this.label} excedeu 64 MiB sem delimitar resposta.`));
    while (true) { const newline = this.#stdout.indexOf("\n"); if (newline < 0) break; const line = this.#stdout.slice(0, newline); this.#stdout = this.#stdout.slice(newline + 1); if (!line) continue;
      let message: { ready?: boolean; initializationSeconds?: number; id?: number; report?: unknown; error?: string }; try { message = JSON.parse(line) as typeof message; } catch (error) { return this.#fail(new Error(`${this.label} retornou JSON inválido: ${error instanceof Error ? error.message : String(error)}`)); }
      if (message.ready) { this.ready = true; this.readyMetadata = { ...message }; if (message.initializationSeconds !== undefined) this.initializationSeconds = message.initializationSeconds; this.#readyAccept(); continue; }
      if (!Number.isSafeInteger(message.id)) return this.#fail(new Error(`${this.label} respondeu sem id válido.`)); const pending = this.pending.get(message.id!); if (!pending) return this.#fail(new Error(`${this.label} respondeu id desconhecido ${message.id}.`)); this.pending.delete(message.id!); if (message.error) pending.reject(new Error(message.error)); else pending.accept(message.report);
    }
  }
  #fail(error: Error): void { this.ready = false; this.#readyReject(error); for (const pending of this.pending.values()) pending.reject(error); this.pending.clear(); }
}

interface ComparisonReport { inputIds?: number[][]; baselineGeneratedTokenIds: number[] }
interface DirectReport { generatedTokenIds: number[]; fullTokenIds: number[]; generatedText?: string; fullText?: string; tokensEqualBaseline?: boolean; firstDivergentStep?: number | null; [key: string]: unknown }
function arraysEqual(left: readonly number[], right: readonly number[]): boolean { return left.length === right.length && left.every((value, index) => value === right[index]); }
function firstDivergence(left: readonly number[], right: readonly number[]): number | null { const length = Math.max(left.length, right.length); for (let index = 0; index < length; index += 1) if (left[index] !== right[index]) return index; return null; }

function validateRequest(request: Gemma4RealComparisonRequest): void {
  if (typeof request.prompt !== "string" || request.prompt.length === 0 || request.prompt.length > 16_384) throw new Error("prompt deve conter entre 1 e 16.384 caracteres.");
  if (!Number.isSafeInteger(request.maxNewTokens) || request.maxNewTokens < 1 || request.maxNewTokens > 64) throw new Error("maxNewTokens deve estar entre 1 e 64.");
  if (request.threads !== undefined && (!Number.isSafeInteger(request.threads) || request.threads < 0 || request.threads > 256)) throw new Error("threads deve estar entre 0 e 256.");
  if (request.precision !== undefined && request.precision !== "f32" && request.precision !== "f64") throw new Error("precision deve ser f32 ou f64.");
  if (request.roundingPolicy !== undefined && request.roundingPolicy !== "none" && request.roundingPolicy !== "layer-bf16" && request.roundingPolicy !== "operation-bf16") throw new Error("roundingPolicy deve ser none, layer-bf16 ou operation-bf16.");
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

export function parseGemma4RealServerOptions(arguments_: readonly string[]): Gemma4RealComparisonRunnerOptions & { port: number; host: string } {
  const values = new Map<string, string>();
  for (let index = 0; index < arguments_.length; index += 2) {
    const flag = arguments_[index], value = arguments_[index + 1];
    if (!flag?.startsWith("--") || value === undefined || values.has(flag)) throw new Error(`Argumento inválido: ${flag ?? "fim"}.`);
    values.set(flag, value);
  }
  const known = new Set(["--source", "--python", "--helper", "--port", "--host", "--literal-artifact", "--binary-pool", "--direct-worker", "--direct-linear-helper", "--direct-mlx-helper", "--direct-linear-backend", "--direct-fused-mlp", "--direct-fused-ple", "--direct-fused-ple-prelude", "--direct-final-head", "--direct-native-attention", "--direct-fused-attention", "--direct-threads", "--direct-max-read-mib"]); for (const key of values.keys()) if (!known.has(key)) throw new Error(`Flag desconhecida: ${key}.`);
  const port = Number(values.get("--port") ?? "8787"); if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) throw new Error("--port inválido.");
  const source = resolve(values.get("--source") ?? "gemma-4-E4B-dense"), inferredLiteral = join(source, "constants.literal.json");
  const literalArtifact = values.get("--literal-artifact") ? resolve(values.get("--literal-artifact")!) : existsSync(inferredLiteral) ? inferredLiteral : undefined;
  const binaryPool = values.get("--binary-pool") ? resolve(values.get("--binary-pool")!) : literalArtifact && existsSync(join(source, "model.safetensors")) ? source : undefined;
  if ((literalArtifact === undefined) !== (binaryPool === undefined)) throw new Error("Backend direto requer --literal-artifact e --binary-pool juntos.");
  const directThreads = Number(values.get("--direct-threads") ?? "8"), directMaxReadMiB = Number(values.get("--direct-max-read-mib") ?? "16");
  if (!Number.isSafeInteger(directThreads) || directThreads < 1 || directThreads > 256 || !Number.isSafeInteger(directMaxReadMiB) || directMaxReadMiB < 1 || directMaxReadMiB > 1024) throw new Error("Configuração direta inválida.");
  const directLinearBackend = values.get("--direct-linear-backend") ?? "pytorch";
  if (directLinearBackend !== "pytorch" && directLinearBackend !== "mlx") throw new Error("--direct-linear-backend deve ser pytorch ou mlx.");
  const directFusedMlp = values.get("--direct-fused-mlp") ?? "real";
  if (directFusedMlp !== "off" && directFusedMlp !== "bf16" && directFusedMlp !== "real") throw new Error("--direct-fused-mlp deve ser off, bf16 ou real.");
  const directFusedPle = values.get("--direct-fused-ple") ?? (directLinearBackend === "pytorch" ? "bf16" : "off");
  if (directFusedPle !== "off" && directFusedPle !== "bf16" && directFusedPle !== "real") throw new Error("--direct-fused-ple deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directFusedPle !== "off") throw new Error("--direct-fused-ple requer backend pytorch.");
  const directFusedPlePrelude = values.get("--direct-fused-ple-prelude") ?? "off";
  if (directFusedPlePrelude !== "off" && directFusedPlePrelude !== "bf16" && directFusedPlePrelude !== "real") throw new Error("--direct-fused-ple-prelude deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directFusedPlePrelude !== "off") throw new Error("--direct-fused-ple-prelude requer backend pytorch.");
  const directFinalHead = values.get("--direct-final-head") ?? (directLinearBackend === "pytorch" ? "native-bf16" : "f32");
  if (directFinalHead !== "f32" && directFinalHead !== "native-bf16" && directFinalHead !== "native-bf16-whole") throw new Error("--direct-final-head deve ser f32, native-bf16 ou native-bf16-whole.");
  if (directLinearBackend === "mlx" && directFinalHead !== "f32") throw new Error("--direct-final-head BF16 requer backend pytorch.");
  const directNativeAttention = values.get("--direct-native-attention") ?? (directLinearBackend === "pytorch" ? "real" : "off");
  if (directNativeAttention !== "off" && directNativeAttention !== "bf16" && directNativeAttention !== "real") throw new Error("--direct-native-attention deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directNativeAttention !== "off") throw new Error("--direct-native-attention requer backend pytorch.");
  const directFusedAttention = values.get("--direct-fused-attention") ?? (directLinearBackend === "pytorch" ? "bf16" : "off");
  if (directFusedAttention !== "off" && directFusedAttention !== "bf16" && directFusedAttention !== "real") throw new Error("--direct-fused-attention deve ser off, bf16 ou real.");
  if (directLinearBackend === "mlx" && directFusedAttention !== "off") throw new Error("--direct-fused-attention requer backend pytorch.");
  return {
    source, python: values.get("--python") ?? (existsSync("venv/bin/python") ? resolve("venv/bin/python") : "python3"),
    helper: resolve(values.get("--helper") ?? "scripts/gemma4-real-differential.py"), port, host: values.get("--host") ?? "127.0.0.1",
    ...(literalArtifact && binaryPool ? { literalArtifact, binaryPool, directWorker: resolve(values.get("--direct-worker") ?? "dist/src/gemma4-paged-runtime-worker-cli.js"), directLinearHelper: resolve(values.get("--direct-linear-helper") ?? "scripts/gemma4-paged-linear-worker.py"), directMlxHelper: resolve(values.get("--direct-mlx-helper") ?? "scripts/gemma4-mlx-linear-worker.py"), directLinearBackend, directFusedMlp, directFusedPle, directFusedPlePrelude, directFinalHead, directNativeAttention, directFusedAttention, directThreads, directMaxReadMiB } : {}),
  };
}
