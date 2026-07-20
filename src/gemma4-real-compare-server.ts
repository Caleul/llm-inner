import { spawn } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { gemma4RealCompareHtml } from "./gemma4-real-compare-ui.js";

export interface Gemma4RealComparisonRequest { prompt: string; maxNewTokens: number; threads?: number }
export interface Gemma4RealComparisonRunnerOptions { source: string; python: string; helper: string }

export async function runGemma4RealComparison(request: Gemma4RealComparisonRequest, options: Gemma4RealComparisonRunnerOptions): Promise<unknown> {
  validateRequest(request);
  const directory = await mkdtemp(join(tmpdir(), "gemma4-real-compare-"));
  const output = join(directory, "report.json");
  try {
    await runProcess(options.python, [options.helper, "--source", options.source, "--prompt", request.prompt, "--max-new-tokens", String(request.maxNewTokens), "--threads", String(request.threads ?? 0), "--output", output]);
    return JSON.parse(await readFile(output, "utf8")) as unknown;
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

export function createGemma4RealComparisonServer(options: Gemma4RealComparisonRunnerOptions) {
  return createServer(async (request, response) => {
    try {
      if (request.method === "GET" && request.url === "/") return send(response, 200, "text/html; charset=utf-8", gemma4RealCompareHtml);
      if (request.method === "GET" && request.url === "/api/status") return json(response, 200, { ready: true, source: options.source });
      if (request.method === "POST" && request.url === "/api/compare") {
        const body = JSON.parse(await readBody(request)) as Gemma4RealComparisonRequest;
        return json(response, 200, await runGemma4RealComparison(body, options));
      }
      return json(response, 404, { error: "Rota não encontrada." });
    } catch (error) {
      return json(response, 400, { error: error instanceof Error ? error.message : String(error) });
    }
  });
}

function validateRequest(request: Gemma4RealComparisonRequest): void {
  if (typeof request.prompt !== "string" || request.prompt.length === 0 || request.prompt.length > 16_384) throw new Error("prompt deve conter entre 1 e 16.384 caracteres.");
  if (!Number.isSafeInteger(request.maxNewTokens) || request.maxNewTokens < 1 || request.maxNewTokens > 64) throw new Error("maxNewTokens deve estar entre 1 e 64.");
  if (request.threads !== undefined && (!Number.isSafeInteger(request.threads) || request.threads < 0 || request.threads > 256)) throw new Error("threads deve estar entre 0 e 256.");
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
