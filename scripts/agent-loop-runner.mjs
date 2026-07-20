#!/usr/bin/env node

import { createWriteStream } from "node:fs";
import { access, copyFile, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { buildProviderInvocation, providerAvailability, providerRunPaths, validateProviders } from "./agent-loop-providers.mjs";
import { classifyTransientLimit, clearProviderRetry, stateAfterTransientLimit } from "./agent-loop-retry-policy.mjs";
import {
  OUTCOME,
  buildAgentTaskPrompt,
  buildContinuityRecord,
  classifyObjectiveOutcome,
  resolveObjectiveStatus,
} from "./agent-task-contract.mjs";

const root = resolve(import.meta.dirname, "..");
const command = process.argv[2] ?? "status";

function absolute(path) {
  return isAbsolute(path) ? path : join(root, path);
}

function now() {
  return new Date().toISOString();
}

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function readJson(path) {
  return JSON.parse(await readFile(path, "utf8"));
}

async function writeJsonAtomically(path, value) {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporary, path);
}

function fail(message) {
  throw new Error(message);
}

async function git(args, { allowFailure = false } = {}) {
  const result = await new Promise((done, reject) => {
    const child = spawn("git", ["-C", root, ...args], { stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    child.once("close", (code) => done({ code: code ?? 1, stdout: stdout.trim(), stderr: stderr.trim() }));
  });
  if (!allowFailure && result.code !== 0) {
    fail(`Git command failed: git ${args.join(" ")}\n${result.stderr || result.stdout}`);
  }
  return result;
}

async function cleanGitTree() {
  return (await git(["status", "--porcelain"])).stdout.length === 0;
}

async function load() {
  const configPath = join(root, "agent-loop.config.json");
  const config = await readJson(configPath);
  if (config.version !== 1 || !Number.isInteger(config.maxLoops) || config.maxLoops < 1) {
    fail("agent-loop.config.json is not a supported version-1 configuration.");
  }
  validateProviders(config.providers);
  if (typeof config.missionGoal !== "string" || config.missionGoal.trim() === "") {
    fail("agent-loop.config.json must declare a non-empty missionGoal.");
  }
  if (!Number.isInteger(config.maxNoProgressMinutes) || config.maxNoProgressMinutes < 1) {
    fail("agent-loop.config.json must declare maxNoProgressMinutes as a positive integer.");
  }
  if (!Number.isInteger(config.transientLimitRetryInitialSeconds) || config.transientLimitRetryInitialSeconds < 1) {
    fail("agent-loop.config.json must declare transientLimitRetryInitialSeconds as a positive integer.");
  }
  if (!Number.isInteger(config.transientLimitRetryMaximumSeconds)
    || config.transientLimitRetryMaximumSeconds < config.transientLimitRetryInitialSeconds) {
    fail("agent-loop.config.json must declare transientLimitRetryMaximumSeconds at least as large as the initial delay.");
  }
  const requiredPaths = [
    config.promptFile,
    config.stateFile,
    join("schemas", "agent-loop-config.schema.json"),
  ];
  for (const path of requiredPaths) {
    if (!(await exists(absolute(path)))) fail(`Required loop file is missing: ${path}`);
  }
  return config;
}

async function ensureRuntime(config) {
  for (const path of [config.handoffDirectory, config.processingDirectory, config.rejectedDirectory, config.runsDirectory]) {
    await mkdir(absolute(path), { recursive: true });
  }
  const statePath = absolute(config.stateFile);
  if (!(await exists(statePath))) {
    await copyFile(join(root, ".agent-loop", "state.example.json"), statePath);
  }
}

async function assertCanRun(config) {
  if (!config.enabled) fail("The loop is disabled in agent-loop.config.json.");
  if (await exists(absolute(config.stopFile))) fail(`Stop flag exists: ${config.stopFile}`);
  await git(["rev-parse", "--is-inside-work-tree"]);
  const name = (await git(["config", "user.name"], { allowFailure: true })).stdout;
  const email = (await git(["config", "user.email"], { allowFailure: true })).stdout;
  if (!name || !email) {
    fail("Git identity missing: configure user.name and user.email before starting the loop.");
  }
  if (config.requireCleanGitBeforeNextLoop && !(await cleanGitTree())) {
    fail("Git tree is not clean. Commit, stash, or remove unrelated changes before starting the loop.");
  }
  for (const provider of config.providers) {
    if (!(await exists(provider.command))) fail(`Provider executable not found for ${provider.id}: ${provider.command}`);
  }
}

function isProcessAlive(pid) {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM proves that a process exists but belongs to another user; ESRCH
    // means the PID recorded by the lock has gone away.
    return error?.code === "EPERM";
  }
}

async function inspectLock(config) {
  const lockPath = absolute(config.lockFile);
  if (!(await exists(lockPath))) return { status: "absent", lockPath };
  try {
    const lock = await readJson(lockPath);
    return {
      status: isProcessAlive(lock.pid) ? "active" : "stale",
      lockPath,
      lock,
    };
  } catch (error) {
    return {
      status: "stale",
      lockPath,
      lock: null,
      parseError: error.message,
    };
  }
}

async function recordInterruptedRun(config, state, reason) {
  const detectedAt = now();
  const interruption = {
    runId: state.lastRunId ?? null,
    sequence: state.currentSequence,
    detectedAt,
    reason,
  };
  if (state.lastRunId) {
    await writeJsonAtomically(join(absolute(config.runsDirectory), `${state.lastRunId}.interrupted.json`), interruption);
  }
  const recovered = {
    ...state,
    status: "interrupted",
    // A stale lock means the host process died; it is not evidence that the
    // autonomous agent or its implementation failed. The durable supervisor
    // may safely restart this state without exhausting the agent-failure
    // budget.
    consecutiveFailures: state.consecutiveFailures,
    lastCompletedAt: detectedAt,
    lastInterruption: interruption,
  };
  await writeJsonAtomically(absolute(config.stateFile), recovered);
  return recovered;
}

/**
 * A runner can be terminated by a host restart, sleep/crash, or external kill
 * before its finally block removes the lock. Recover that state before every
 * start instead of permanently reporting a nonexistent Codex as "running".
 */
async function recoverInterruptedRuntime(config) {
  let state = await readJson(absolute(config.stateFile));
  const inspection = await inspectLock(config);
  if (inspection.status === "active") {
    fail(`Loop lock belongs to active PID ${inspection.lock.pid}: ${config.lockFile}`);
  }
  const needsRecovery = state.status === "running" || inspection.status === "stale";
  if (!needsRecovery) return { state, recovered: false, inspection };

  const reason = inspection.status === "stale"
    ? `Runner lock is stale (PID ${inspection.lock?.pid ?? "unknown"} is not alive).`
    : "State says running but no runner lock exists.";
  if (state.status === "running") state = await recordInterruptedRun(config, state, reason);
  if (inspection.status === "stale") await rm(inspection.lockPath, { force: true });
  if (!(await cleanGitTree())) {
    const recovery = await checkpointFailedWork(
      state.lastRunId ?? "interrupted-unknown-run",
      state.currentSequence,
      reason,
    );
    state = { ...state, lastRecovery: recovery, lastCompletedAt: now() };
    await writeJsonAtomically(absolute(config.stateFile), state);
  }
  console.error(`Recovered interrupted loop state: ${reason}`);
  return { state, recovered: true, inspection };
}

async function acquireLock(config) {
  const lockPath = absolute(config.lockFile);
  try {
    const handle = await open(lockPath, "wx");
    await handle.writeFile(`${JSON.stringify({ pid: process.pid, startedAt: now(), root })}\n`);
    await handle.close();
  } catch (error) {
    if (error?.code === "EEXIST") fail(`Loop lock already exists: ${config.lockFile}`);
    throw error;
  }
  return lockPath;
}

async function completedHandoffs(config) {
  const directory = absolute(config.handoffDirectory);
  return new Set((await readdir(directory)).filter((name) => new RegExp(config.handoffPattern).test(name)));
}

function validateHandoffShape(handoff, expectedSequence, expectedRunId) {
  if (handoff?.schemaVersion !== 1) fail("Handoff schemaVersion must be 1.");
  if (handoff.sequence !== expectedSequence) fail(`Handoff sequence must be ${expectedSequence}.`);
  if (handoff.runId !== expectedRunId) fail(`Handoff runId must be ${expectedRunId}.`);
  if (!Number.isFinite(Date.parse(handoff.createdAt))) fail("Handoff createdAt must be an ISO timestamp.");
  if (!["continue", "complete", "blocked"].includes(handoff.missionStatus)) fail("Handoff missionStatus is invalid.");
  if (!["success", "partial", "failed"].includes(handoff.loopStatus)) fail("Handoff loopStatus is invalid.");
  for (const field of ["summary", "title"]) {
    if (typeof handoff[field] !== "string" || handoff[field].trim() === "") fail(`Handoff ${field} is required.`);
  }
  if (!Array.isArray(handoff.validation?.commands)) fail("Handoff validation.commands must be an array.");
  if (!Array.isArray(handoff.missionGates?.satisfied) || !Array.isArray(handoff.missionGates?.unsatisfied)) {
    fail("Handoff missionGates must include satisfied and unsatisfied arrays.");
  }
  if (handoff.endingState?.dirty !== false) fail("Handoff must declare endingState.dirty as false.");
  if (!Array.isArray(handoff.knownLimitations) || !Array.isArray(handoff.blockedBy)) {
    fail("Handoff must include concise knownLimitations and blockedBy arrays.");
  }
  if ("nextRecommendedMilestone" in handoff || "nextSteps" in handoff) {
    fail("Handoffs must be backward-looking evidence, not assignments for a successor.");
  }
  if (!Array.isArray(handoff.bottlenecks)) fail("Handoff must include a bottlenecks array.");
  for (const bottleneck of handoff.bottlenecks) {
    if (
      !bottleneck ||
      typeof bottleneck.description !== "string" ||
      typeof bottleneck.impact !== "string" ||
      typeof bottleneck.evidence !== "string"
    ) {
      fail("Each bottleneck requires description, impact, and evidence.");
    }
  }
}

async function moveRejected(config, name, reason) {
  const source = join(absolute(config.handoffDirectory), name);
  const target = join(absolute(config.rejectedDirectory), `${name}.rejected.json`);
  await writeFile(`${target}.reason.txt`, `${reason}\n`, "utf8");
  await rename(source, target);
}

function terminateProcessGroup(child, signal) {
  if (!child.pid) return;
  try {
    // Codex may own MCP/tool descendants. Kill the process group so a timed
    // out parent cannot leave descendants alive indefinitely.
    process.kill(-child.pid, signal);
  } catch {
    child.kill(signal);
  }
}

const delay = (milliseconds) => new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));

async function waitUntilRetryOrStop(config, nextRetryAt) {
  while (true) {
    if (await exists(absolute(config.stopFile))) return false;
    const remaining = Date.parse(nextRetryAt) - Date.now();
    if (remaining <= 0) return true;
    await delay(Math.min(remaining, 5_000));
  }
}

function withoutLimitRetry(state) {
  const { limitRetry: _limitRetry, ...remaining } = state;
  return remaining;
}

async function checkpointFailedWork(runId, sequence, reason) {
  if (await cleanGitTree()) return null;
  const startingCommit = (await git(["rev-parse", "HEAD"])).stdout;
  await git(["add", "-A"]);
  await git(["commit", "-m", `chore: checkpoint failed autonomous loop ${sequence}`]);
  const commit = (await git(["rev-parse", "HEAD"])).stdout;
  return { runId, sequence, reason, startingCommit, commit, checkpointedAt: now() };
}

async function runProvider(config, provider, prompt, runId, sequence) {
  const runDirectory = absolute(config.runsDirectory);
  const paths = providerRunPaths(runDirectory, runId);
  paths.promptPath = join(tmpdir(), `llm-inner-gemma-task-${process.pid}-${Date.now()}.md`);
  await writeFile(paths.promptPath, prompt, "utf8");
  const invocation = buildProviderInvocation(provider, {
    root,
    promptPath: paths.promptPath,
    outputPath: paths.outputPath,
  });
  const logPath = paths.logPath;
  const log = createWriteStream(logPath, { flags: "a" });
  await writeJsonAtomically(paths.metadataPath, {
    runId,
    sequence,
    provider: { id: provider.id, kind: provider.kind, model: provider.model },
    startedAt: now(),
    command: invocation.command,
    args: invocation.args,
    promptPath: paths.promptPath,
  });
  const result = await new Promise((done) => {
    const child = spawn(invocation.command, invocation.args, {
      cwd: root,
      detached: true,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ...(provider.environment ?? {}) },
    });
    let timeoutReason = null;
    let limitError = null;
    let recentOutput = "";
    let lastProgressAt = Date.now();
    let forceKill = null;
    let settled = false;
    const stopFor = (reason) => {
      if (timeoutReason) return;
      timeoutReason = reason;
      console.error(`Terminating ${runId}: ${reason}`);
      terminateProcessGroup(child, "SIGTERM");
      forceKill = setTimeout(() => terminateProcessGroup(child, "SIGKILL"), 15_000);
    };
    const timeout = setTimeout(
      () => stopFor(`maximum duration of ${config.maxLoopDurationMinutes} minutes exceeded`),
      config.maxLoopDurationMinutes * 60_000,
    );
    const progressWatchdog = setInterval(() => {
      if (Date.now() - lastProgressAt >= config.maxNoProgressMinutes * 60_000) {
        stopFor(`no provider stdout/stderr progress for ${config.maxNoProgressMinutes} minutes`);
      }
    }, 5_000);
    const recordProgress = (chunk) => {
      lastProgressAt = Date.now();
      recentOutput = `${recentOutput}${chunk.toString("utf8")}`.slice(-16_384);
      limitError ??= classifyTransientLimit(recentOutput);
      process.stdout.write(chunk);
      log.write(chunk);
    };
    child.stdout.on("data", recordProgress);
    child.stderr.on("data", recordProgress);
    child.once("error", (error) => {
      clearTimeout(timeout);
      clearInterval(progressWatchdog);
      if (forceKill) clearTimeout(forceKill);
      log.end();
      if (settled) return;
      settled = true;
      done({ code: 127, signal: null, timeoutReason: null, limitError: "provider_unavailable", spawnError: error.message });
    });
    child.once("close", (code, signal) => {
      if (settled) return;
      settled = true;
      clearTimeout(timeout);
      clearInterval(progressWatchdog);
      if (forceKill) clearTimeout(forceKill);
      log.end();
      done({ code: code ?? 1, signal, timeoutReason, limitError, recentOutput });
    });
    child.stdin.end(invocation.stdin === "prompt" ? prompt : undefined);
  });
  await rm(paths.promptPath, { force: true });
  return { ...result, outputPath: paths.outputPath };
}

async function readProviderResult(result) {
  if (result.outputPath && await exists(result.outputPath)) {
    const message = (await readFile(result.outputPath, "utf8")).trim();
    if (message) return message;
  }
  return (result.recentOutput ?? "").trim();
}

async function runValidationCommand(command, timeoutMinutes = 60) {
  return new Promise((done) => {
    const child = spawn("/bin/zsh", ["-lc", command], {
      cwd: root,
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    let timedOut = false;
    const collect = (chunk) => { output = `${output}${chunk.toString("utf8")}`.slice(-32_768); };
    child.stdout.on("data", collect);
    child.stderr.on("data", collect);
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMinutes * 60_000);
    child.once("error", (error) => {
      clearTimeout(timeout);
      done({ command, status: "failed", error: error.message });
    });
    child.once("close", (code) => {
      clearTimeout(timeout);
      done({
        command,
        status: code === 0 && !timedOut ? "passed" : "failed",
        ...(code === 0 ? {} : { exitCode: code ?? 1 }),
        ...(timedOut ? { error: `timed out after ${timeoutMinutes} minutes` } : {}),
        output: output.trim().slice(-4_000),
      });
    });
  });
}

async function runConfiguredValidation(config) {
  const results = [];
  for (const command of config.validationCommands) {
    results.push(await runValidationCommand(command));
    if (results.at(-1).status !== "passed") break;
  }
  return results;
}

function continuityName(sequence, createdAt, status) {
  const timestamp = createdAt.replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
  return `HANDOFF-${String(sequence).padStart(4, "0")}-${timestamp}-gemma-objective-${status}.json`;
}

async function start(config) {
  await ensureRuntime(config);
  await recoverInterruptedRuntime(config);
  await assertCanRun(config);
  const lockPath = await acquireLock(config);
  try {
    let state = await readJson(absolute(config.stateFile));
    while (state.completedLoops < config.maxLoops) {
      if (await exists(absolute(config.stopFile))) {
        state = { ...state, status: "stopped", lastCompletedAt: now() };
        await writeJsonAtomically(absolute(config.stateFile), state);
        console.log("Stop flag found; no additional Codex will be started.");
        return;
      }
      const availability = providerAvailability(config.providers, state.providerRetries);
      if (availability.available.length === 0 && availability.nextRetryAt) {
        state = { ...state, status: "waiting_limit", nextProviderRetryAt: availability.nextRetryAt };
        await writeJsonAtomically(absolute(config.stateFile), state);
        console.error(`All providers are cooling down; waiting until ${availability.nextRetryAt}.`);
        if (!(await waitUntilRetryOrStop(config, availability.nextRetryAt))) {
          state = { ...state, status: "stopped", lastCompletedAt: now() };
          await writeJsonAtomically(absolute(config.stateFile), state);
          return;
        }
        continue;
      }
      const provider = availability.available[0];
      if (config.requireCleanGitBeforeNextLoop && !(await cleanGitTree())) fail("Git tree became dirty before a new task.");
      const sequence = state.currentSequence + 1;
      const runId = `run-${String(sequence).padStart(4, "0")}-${now().replace(/[-:.]/g, "").replace("Z", "Z")}-${provider.id}`;
      const startingCommit = (await git(["rev-parse", "HEAD"])).stdout;
      const before = await completedHandoffs(config);
      const masterPrompt = await readFile(absolute(config.promptFile), "utf8");
      state = {
        ...state,
        status: "running",
        currentSequence: sequence,
        lastRunId: runId,
        lastStartedAt: now(),
        activeProvider: { id: provider.id, kind: provider.kind, model: provider.model },
      };
      await writeJsonAtomically(absolute(config.stateFile), state);
      console.log(`Starting task ${sequence} with provider ${provider.id} (${provider.kind}/${provider.model}).`);
      const prompt = buildAgentTaskPrompt(masterPrompt, config.missionGoal);
      const result = await runProvider(config, provider, prompt, runId, sequence);
      const after = await completedHandoffs(config);
      const created = [...after].filter((name) => !before.has(name));
      let failure = null;
      let accepted = null;
      let recovery = null;
      if (created.length !== 0) {
        failure = `The agent wrote ${created.length} operational continuity record(s); product sessions must not inspect or modify controller state.`;
        for (const name of created) await moveRejected(config, name, failure);
      } else if (result.code !== 0 || result.timeoutReason || result.spawnError) {
        failure = result.timeoutReason
          ?? result.spawnError
          ?? `Provider ${provider.id} exited with ${result.code}${result.signal ? ` (${result.signal})` : ""}.`;
      } else if (!result.limitError) {
        try {
          const message = await readProviderResult(result);
          const outcome = classifyObjectiveOutcome(message);
          const endingCommit = (await git(["rev-parse", "HEAD"])).stdout;
          const repositoryChanged = endingCommit !== startingCommit;
          const resolved = resolveObjectiveStatus({ outcome, repositoryChanged });
          if (resolved === OUTCOME.invalid) {
            fail(`Invalid objective result: marker=${outcome}, repositoryChanged=${repositoryChanged}.`);
          }
          if (config.requireCleanGitBeforeNextLoop && !(await cleanGitTree())) {
            fail("Git tree is dirty after the engineering task.");
          }
          const validation = await runConfiguredValidation(config);
          const failedValidation = validation.find(({ status }) => status !== "passed");
          if (failedValidation) {
            const details = [
              failedValidation.error,
              Number.isInteger(failedValidation.exitCode) ? `exit=${failedValidation.exitCode}` : null,
              failedValidation.output,
            ].filter(Boolean).join("\n");
            fail(`Controller validation failed: ${failedValidation.command}.${details ? `\n${details}` : ""}`);
          }
          const createdAt = now();
          const record = buildContinuityRecord({
            sequence,
            runId,
            createdAt,
            status: resolved,
            summary: message || `Provider ${provider.id} returned no final summary.`,
            endingCommit,
            validation,
            missionGateIds: config.missionGates.filter(({ required }) => required).map(({ id }) => id),
          });
          const name = continuityName(sequence, createdAt, resolved);
          validateHandoffShape(record, sequence, runId);
          await writeJsonAtomically(join(absolute(config.handoffDirectory), name), record);
          accepted = { name, handoff: record, endingCommit };
        } catch (error) {
          failure = error.message;
        }
      }
      if (!accepted) {
        const recoveryReason = result.timeoutReason
          ?? (result.limitError ? `transient provider limit: ${result.limitError}` : failure);
        try {
          recovery = await checkpointFailedWork(runId, sequence, recoveryReason);
        } catch (error) {
          failure = `${failure} Failed-worktree checkpoint failed: ${error.message}`;
        }
        if (result.limitError) {
          state = stateAfterTransientLimit(state, {
            kind: result.limitError,
            providerId: provider.id,
            runId,
            sequence,
            detectedAt: now(),
            initialSeconds: config.transientLimitRetryInitialSeconds,
            maximumSeconds: config.transientLimitRetryMaximumSeconds,
          });
          if (recovery) state = { ...state, lastRecovery: recovery };
          await writeJsonAtomically(absolute(config.stateFile), state);
          console.error(`Provider ${provider.id} hit ${result.limitError}; task ${sequence} and failure counters were restored. Provider cooldown ${state.limitRetry.attempts} is ${state.limitRetry.retryAfterSeconds} seconds; trying the next available provider.`);
          continue;
        }
        state = withoutLimitRetry(state);
        state = {
          ...state,
          status: "failed",
          consecutiveFailures: state.consecutiveFailures + 1,
          lastCompletedAt: now(),
          ...(recovery ? { lastRecovery: recovery } : {}),
        };
        await writeJsonAtomically(absolute(config.stateFile), state);
        console.error(`Task ${sequence} rejected: ${failure}`);
        if (state.consecutiveFailures >= config.maxConsecutiveFailures) {
          console.error(`Stopping after ${state.consecutiveFailures} consecutive failures.`);
          return;
        }
        await delay(config.cooldownSeconds * 1000);
      } else {
        state = {
          ...withoutLimitRetry(clearProviderRetry(state, provider.id)),
          status: accepted.handoff.missionStatus === "continue" ? "idle" : accepted.handoff.missionStatus,
          completedLoops: state.completedLoops + 1,
          consecutiveFailures: 0,
          lastHandoff: accepted.name,
          lastCompletedAt: now(),
          missionStatus: accepted.handoff.missionStatus,
        };
        await writeJsonAtomically(absolute(config.stateFile), state);
        console.log(`Accepted ${accepted.name} at ${accepted.endingCommit}.`);
        if (accepted.handoff.missionStatus !== "continue") return;
        if (state.completedLoops < config.maxLoops) await delay(config.cooldownSeconds * 1000);
      }
    }
    console.log(`Configured maximum of ${config.maxLoops} completed tasks reached.`);
  } finally {
    await rm(lockPath, { force: true });
  }
}

async function status(config) {
  await ensureRuntime(config);
  const state = await readJson(absolute(config.stateFile));
  const lock = await inspectLock(config);
  const orphaned = state.status === "running" && lock.status !== "active";
  console.log(JSON.stringify({
    state,
    maxLoops: config.maxLoops,
    providers: config.providers.map(({ id, kind, model }) => ({ id, kind, model })),
    activeProvider: state.activeProvider ?? null,
    lockPresent: lock.status !== "absent",
    lockStatus: lock.status,
    lockPid: lock.lock?.pid ?? null,
    orphanedRunDetected: orphaned,
    stopRequested: await exists(absolute(config.stopFile)),
  }, null, 2));
}

async function stop(config) {
  await ensureRuntime(config);
  await writeFile(absolute(config.stopFile), `Stop requested at ${now()}\n`, "utf8");
  console.log(`Stop requested. The active Codex may finish its coherent cycle; no successor will start.`);
}

try {
  const config = await load();
  if (command === "start") await start(config);
  else if (command === "status") await status(config);
  else if (command === "stop") await stop(config);
  else fail(`Unknown command: ${command}. Use start, status, or stop.`);
} catch (error) {
  console.error(`agent-loop-runner: ${error.message}`);
  process.exitCode = 1;
}
