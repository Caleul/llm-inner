import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gemma4RealCompareHtml } from "./gemma4-real-compare-ui.js";

export interface Gemma4RealComparisonRequest { prompt: string; maxNewTokens: number; threads?: number; precision?: "f32" | "f64"; roundingPolicy?: "none" | "layer-bf16" | "operation-bf16" }
export interface Gemma4RealComparisonRunnerOptions { source: string; python: string; helper: string }

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
  const server = createServer(async (request, response) => {
    try {
      if (request.method === "GET" && request.url === "/") return send(response, 200, "text/html; charset=utf-8", gemma4RealCompareHtml);
      if (request.method === "GET" && request.url === "/api/status") return json(response, 200, { ready: worker.ready, source: options.source, runtime: "persistent-jsonl", initializationSeconds: worker.initializationSeconds });
      if (request.method === "POST" && request.url === "/api/compare") {
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        validateRequest(body);
        return json(response, 200, await worker.compare(body));
      }
      return json(response, 404, { error: "Rota não encontrada." });
    } catch (error) {
      return json(response, 400, { error: error instanceof Error ? error.message : String(error) });
    }
  });
  server.once("close", () => worker.close());
  return server;
}

class Gemma4PersistentComparisonWorker {
  readonly child: ChildProcessWithoutNullStreams;
  readonly pending = new Map<number, { accept(value: unknown): void; reject(error: Error): void }>();
  ready = false;
  initializationSeconds?: number;
  #nextId = 1;
  #stdout = "";
  #readyAccept!: () => void;
  #readyReject!: (error: Error) => void;
  readonly #readyPromise: Promise<void>;

  constructor(options: Gemma4RealComparisonRunnerOptions) {
    this.#readyPromise = new Promise<void>((accept, reject) => { this.#readyAccept = accept; this.#readyReject = reject; });
    this.child = spawn(options.python, [options.helper, "--source", options.source, "--serve-jsonl"], { stdio: ["pipe", "pipe", "pipe"] });
    this.child.stdout.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => this.#consume(chunk));
    this.child.once("error", (error) => this.#fail(error));
    this.child.once("close", (code) => this.#fail(new Error(`Worker Gemma 4 persistente encerrou com código ${code}.`)));
  }

  async compare(request: Gemma4RealComparisonRequest): Promise<unknown> {
    await this.#readyPromise;
    const id = this.#nextId++;
    const result = new Promise<unknown>((accept, reject) => this.pending.set(id, { accept, reject }));
    this.child.stdin.write(`${JSON.stringify({ id, ...request })}\n`);
    return result;
  }

  close(): void { if (!this.child.killed) this.child.kill("SIGTERM"); }

  #consume(chunk: string): void {
    this.#stdout += chunk;
    if (this.#stdout.length > 64 * 1024 * 1024) return this.#fail(new Error("Worker Gemma 4 excedeu 64 MiB sem delimitar resposta."));
    while (true) {
      const newline = this.#stdout.indexOf("\n"); if (newline < 0) break;
      const line = this.#stdout.slice(0, newline); this.#stdout = this.#stdout.slice(newline + 1);
      if (!line) continue;
      const message = JSON.parse(line) as { ready?: boolean; initializationSeconds?: number; id?: number; report?: unknown; error?: string };
      if (message.ready) { this.ready = true; if (message.initializationSeconds !== undefined) this.initializationSeconds = message.initializationSeconds; this.#readyAccept(); continue; }
      if (!Number.isSafeInteger(message.id)) return this.#fail(new Error("Worker Gemma 4 respondeu sem id válido."));
      const pending = this.pending.get(message.id!); if (!pending) return this.#fail(new Error(`Worker Gemma 4 respondeu id desconhecido ${message.id}.`));
      this.pending.delete(message.id!);
      if (message.error) pending.reject(new Error(message.error)); else pending.accept(message.report);
    }
  }

  #fail(error: Error): void {
    this.ready = false; this.#readyReject(error);
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
  }
}

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
  const known = new Set(["--source", "--python", "--helper", "--port", "--host"]); for (const key of values.keys()) if (!known.has(key)) throw new Error(`Flag desconhecida: ${key}.`);
  const port = Number(values.get("--port") ?? "8787"); if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) throw new Error("--port inválido.");
  return {
    source: resolve(values.get("--source") ?? "gemma-4-E4B-dense"), python: values.get("--python") ?? (existsSync("venv/bin/python") ? resolve("venv/bin/python") : "python3"),
    helper: resolve(values.get("--helper") ?? "scripts/gemma4-real-differential.py"), port, host: values.get("--host") ?? "127.0.0.1",
  };
}
