#!/usr/bin/env node

import { createWriteStream } from "node:fs";
import { access, copyFile, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { classifyTransientLimit, stateAfterTransientLimit } from "./agent-loop-retry-policy.mjs";

const root = resolve(import.meta.dirname, "..");
const codex = process.env.CODEX_BIN ?? "/Applications/ChatGPT.app/Contents/Resources/codex";
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
  if (!config.agent || typeof config.agent.model !== "string" || !["low", "medium", "high"].includes(config.agent.reasoningEffort)) {
    fail("agent-loop.config.json must declare an agent model and low, medium, or high reasoningEffort.");
  }
  if (typeof config.missionGoal !== "string" || config.missionGoal.trim() === "") {
    fail("agent-loop.config.json must declare a non-empty missionGoal.");
  }
  if (config.goalCommand !== "/goal") {
    fail("agent-loop.config.json must declare goalCommand as /goal.");
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
  if (!(await exists(codex))) fail(`Codex executable not found: ${codex}`);
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

async function runCodex(config, prompt, runId, sequence) {
  const runDirectory = absolute(config.runsDirectory);
  const outputPath = join(runDirectory, `${runId}.last-message.md`);
  const logPath = join(runDirectory, `${runId}.log`);
  const log = createWriteStream(logPath, { flags: "a" });
  const args = [
    "exec",
    "-C", root,
    "-m", config.agent.model,
    "-c", `model_reasoning_effort=${JSON.stringify(config.agent.reasoningEffort)}`,
    // The user explicitly authorizes this autonomous runner to work without
    // sandboxing or interactive approvals, including repository metadata.
    "--dangerously-bypass-approvals-and-sandbox",
    "--color", "never",
    "--output-last-message", outputPath,
    "-",
  ];
  await writeJsonAtomically(join(runDirectory, `${runId}.json`), {
    runId,
    sequence,
    startedAt: now(),
    command: codex,
    args: args.slice(0, -1),
  });
  return new Promise((done, reject) => {
    const child = spawn(codex, args, { cwd: root, detached: true, stdio: ["pipe", "pipe", "pipe"] });
    let timeoutReason = null;
    let limitError = null;
    let recentOutput = "";
    let lastProgressAt = Date.now();
    let forceKill = null;
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
        stopFor(`no Codex stdout/stderr progress for ${config.maxNoProgressMinutes} minutes`);
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
      reject(error);
    });
    child.once("close", (code, signal) => {
      clearTimeout(timeout);
      clearInterval(progressWatchdog);
      if (forceKill) clearTimeout(forceKill);
      log.end();
      done({ code: code ?? 1, signal, timeoutReason, limitError });
    });
    child.stdin.end(prompt);
  });
}

function agentPrompt(masterPrompt, config, state, runId, sequence, previousHandoff) {
  return `${masterPrompt}\n\n---\n\n${config.goalCommand}\n\n# Mandatory exclusive goal for loop ${sequence}\n\n${config.missionGoal}\n\n---\n\n# Runner invocation context\n\nYou are the single active Codex for loop ${sequence} (${runId}). This exclusive goal overrides generic opportunity selection: select only the largest coherent Gemma 4 artifact-delivery boundary that directly advances it, and leave an evidence-backed blocker only when no such work remains. Apply Clean Code and SOLID pragmatically while completing that product. Do not stop at a micro-change, isolated test, or small commit merely to create a handoff. A successful loop may not consist only of adding the next layer ID to a reduction/profile allowlist or rerunning the same probe for the next assignment; solve the compatible operation class across the artifact or move to another implementable artifact gap. Do not ask for a plan or wait for human input unless an actual external resource or decision is required.\n\nDo every safe, coherent improvement that current evidence reveals. Never write that a future implementation or next loop should perform ordinary work you can perform now. A predecessor's handoff is evidence only: independently review its claimed results, then choose and execute the current loop's own highest-impact artifact work.\n\nA predecessor can propose a candidate acceptance claim, but never certify its own work. Before advancing any claimed gate or reporting an objective achieved, independently inspect the predecessor's diff and rerun or strengthen its evidence. Your independent review, not the implementer's self-assessment, is the authority for accepting the claim.\n\nCurrent state:\n\n\`\`\`json\n${JSON.stringify(state, null, 2)}\n\`\`\`\n\nPrevious accepted handoff: ${previousHandoff ?? "none; this is the first loop"}.\n\nThe external runner, not you, starts the next loop. Before exiting, run every command in agent-loop.config.json.validationCommands, create one local Git commit, and atomically create exactly one handoff named HANDOFF-${String(sequence).padStart(4, "0")}-<UTC timestamp>-<slug>.json in ${config.handoffDirectory}. Its runId must be \`${runId}\`, sequence must be ${sequence}, and endingState.gitCommit must equal the new HEAD commit. The handoff is your final repository-changing action. It must be backward-looking: include only completed results, reproducible validation, known limits and evidence-backed bottlenecks. Do not include nextRecommendedMilestone, nextSteps, or a proposal for the successor.\n`;
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
      if (state.status === "waiting_limit" && state.limitRetry?.nextRetryAt) {
        console.error(`Waiting until ${state.limitRetry.nextRetryAt} after ${state.limitRetry.kind} attempt ${state.limitRetry.attempts}.`);
        if (!(await waitUntilRetryOrStop(config, state.limitRetry.nextRetryAt))) {
          state = { ...state, status: "stopped", lastCompletedAt: now() };
          await writeJsonAtomically(absolute(config.stateFile), state);
          return;
        }
      }
      if (config.requireCleanGitBeforeNextLoop && !(await cleanGitTree())) fail("Git tree became dirty before a new loop.");
      const sequence = state.currentSequence + 1;
      const runId = `run-${String(sequence).padStart(4, "0")}-${now().replace(/[-:.]/g, "").replace("Z", "Z")}`;
      const startingCommit = (await git(["rev-parse", "HEAD"])).stdout;
      const before = await completedHandoffs(config);
      const masterPrompt = await readFile(absolute(config.promptFile), "utf8");
      state = { ...state, status: "running", currentSequence: sequence, lastRunId: runId, lastStartedAt: now() };
      await writeJsonAtomically(absolute(config.stateFile), state);
      const result = await runCodex(config, agentPrompt(masterPrompt, config, state, runId, sequence, state.lastHandoff), runId, sequence);
      const after = await completedHandoffs(config);
      const created = [...after].filter((name) => !before.has(name));
      let failure = null;
      let accepted = null;
      let recovery = null;
      if (created.length !== 1) {
        failure = `Expected exactly one new completed handoff, found ${created.length}. Codex exit=${result.code}${result.signal ? ` signal=${result.signal}` : ""}.`;
      } else {
        const name = created[0];
        try {
          const handoffPath = join(absolute(config.handoffDirectory), name);
          const handoff = await readJson(handoffPath);
          validateHandoffShape(handoff, sequence, runId);
          const endingCommit = (await git(["rev-parse", "HEAD"])).stdout;
          if (config.requireNewCommitPerLoop && endingCommit === startingCommit) fail("No new Git commit was created for this loop.");
          if (handoff.endingState.gitCommit !== endingCommit) fail("Handoff endingState.gitCommit does not equal HEAD.");
          if (config.requireCleanGitBeforeNextLoop && !(await cleanGitTree())) fail("Git tree is dirty after the loop.");
          accepted = { name, handoff, endingCommit };
        } catch (error) {
          failure = error.message;
          await moveRejected(config, name, failure);
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
            runId,
            sequence,
            detectedAt: now(),
            initialSeconds: config.transientLimitRetryInitialSeconds,
            maximumSeconds: config.transientLimitRetryMaximumSeconds,
          });
          if (recovery) state = { ...state, lastRecovery: recovery };
          await writeJsonAtomically(absolute(config.stateFile), state);
          console.error(`Attempted loop ${sequence} hit ${result.limitError}; sequence and failure counters were restored. Retry ${state.limitRetry.attempts} in ${state.limitRetry.retryAfterSeconds} seconds.`);
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
        console.error(`Loop ${sequence} rejected: ${failure}`);
        if (state.consecutiveFailures >= config.maxConsecutiveFailures) {
          console.error(`Stopping after ${state.consecutiveFailures} consecutive failures.`);
          return;
        }
        await delay(config.cooldownSeconds * 1000);
      } else {
        state = {
          ...withoutLimitRetry(state),
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
    console.log(`Configured maximum of ${config.maxLoops} completed loops reached.`);
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
    agent: config.agent,
    lockPresent: lock.status !== "absent",
    lockStatus: lock.status,
    lockPid: lock.lock?.pid ?? null,
    orphanedRunDetected: orphaned,
    stopRequested: await exists(absolute(config.stopFile)),
    codex,
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
