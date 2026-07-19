#!/usr/bin/env node

/**
 * Durable parent for the autonomous loop. It is intended to run under macOS
 * launchd, not under a Codex terminal session. A crashed runner is restarted
 * after a short delay; an explicit STOP flag or a terminal mission state ends
 * the supervisor cleanly.
 */
import { access, readFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

// launchd may start with a transient/unavailable cwd. Never derive the
// repository root from process.cwd(); use the explicit LaunchAgent value or
// the module's file URL instead.
const root = process.env.LLM_INNER_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), "..");
const restartDelayMs = 5_000;
const command = process.argv[2] ?? "run";

function absolute(value) {
  return isAbsolute(value) ? value : join(root, value);
}

async function exists(file) {
  try { await access(file); return true; } catch { return false; }
}

async function loadConfig() {
  return JSON.parse(await readFile(join(root, "agent-loop.config.json"), "utf8"));
}

async function readState(config) {
  return JSON.parse(await readFile(absolute(config.stateFile), "utf8"));
}

function runRunner() {
  return new Promise((resolveRun) => {
    const child = spawn(process.execPath, [join(root, "scripts", "agent-loop-runner.mjs"), "start"], {
      cwd: root,
      stdio: "inherit",
      env: process.env,
    });
    child.once("error", (error) => resolveRun({ code: 1, signal: null, error }));
    child.once("close", (code, signal) => resolveRun({ code: code ?? 1, signal, error: null }));
  });
}

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));

async function shouldExit(config) {
  if (await exists(absolute(config.stopFile))) return "STOP flag exists";
  const state = await readState(config);
  if (state.completedLoops >= config.maxLoops) return "configured loop budget reached";
  if (state.consecutiveFailures >= config.maxConsecutiveFailures) return `maximum consecutive failures reached (${state.consecutiveFailures})`;
  if (state.missionStatus === "complete" || state.missionStatus === "blocked") return `mission status is ${state.missionStatus}`;
  return null;
}

async function main() {
  if (command !== "run" && command !== "once") throw new Error("Use: agent-loop-supervisor.mjs [run|once]");
  const config = await loadConfig();
  do {
    const exitReason = await shouldExit(config);
    if (exitReason) { console.log(`Supervisor exiting: ${exitReason}.`); return; }

    const result = await runRunner();
    const afterRun = await shouldExit(config);
    if (afterRun) { console.log(`Supervisor exiting: ${afterRun}.`); return; }
    console.error(`Runner exited unexpectedly (code=${result.code}${result.signal ? ` signal=${result.signal}` : ""}${result.error ? ` error=${result.error.message}` : ""}); restarting in ${restartDelayMs}ms.`);
    if (command === "once") process.exitCode = result.code || 1;
    else await delay(restartDelayMs);
  } while (command === "run");
}

main().catch((error) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
