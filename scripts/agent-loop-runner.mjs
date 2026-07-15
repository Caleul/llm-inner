#!/usr/bin/env node

import { createWriteStream } from "node:fs";
import { access, copyFile, mkdir, open, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { spawn } from "node:child_process";

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
  if (handoff.missionStatus === "continue") {
    const next = handoff.nextRecommendedMilestone;
    if (
      !next ||
      typeof next.title !== "string" || next.title.trim() === "" ||
      typeof next.reason !== "string" || next.reason.trim() === "" ||
      !Array.isArray(next.acceptanceCriteria) || next.acceptanceCriteria.length === 0
    ) {
      fail("Continuing handoffs must name one next substantial milestone with acceptance criteria.");
    }
    if (!Array.isArray(handoff.nextSteps) || handoff.nextSteps.length < 1 || handoff.nextSteps.length > 3) {
      fail("Continuing handoffs must contain one to three ordered nextSteps.");
    }
    for (const step of handoff.nextSteps) {
      if (
        !step ||
        typeof step.title !== "string" || step.title.trim() === "" ||
        typeof step.reason !== "string" || step.reason.trim() === "" ||
        !Array.isArray(step.acceptanceCriteria) || step.acceptanceCriteria.length === 0
      ) {
        fail("Each nextSteps entry requires title, reason, and acceptanceCriteria.");
      }
    }
    if (!Array.isArray(handoff.bottlenecks)) fail("Continuing handoffs must include a bottlenecks array.");
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
}

async function moveRejected(config, name, reason) {
  const source = join(absolute(config.handoffDirectory), name);
  const target = join(absolute(config.rejectedDirectory), `${name}.rejected.json`);
  await writeFile(`${target}.reason.txt`, `${reason}\n`, "utf8");
  await rename(source, target);
}

async function runCodex(config, prompt, runId, sequence) {
  const runDirectory = absolute(config.runsDirectory);
  const outputPath = join(runDirectory, `${runId}.last-message.md`);
  const logPath = join(runDirectory, `${runId}.log`);
  const log = createWriteStream(logPath, { flags: "a" });
  const args = [
    "exec",
    "-C", root,
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
    const child = spawn(codex, args, { cwd: root, stdio: ["pipe", "pipe", "pipe"] });
    const timeout = setTimeout(() => child.kill("SIGTERM"), config.maxLoopDurationMinutes * 60_000);
    child.stdout.on("data", (chunk) => { process.stdout.write(chunk); log.write(chunk); });
    child.stderr.on("data", (chunk) => { process.stderr.write(chunk); log.write(chunk); });
    child.once("error", (error) => { clearTimeout(timeout); log.end(); reject(error); });
    child.once("close", (code, signal) => {
      clearTimeout(timeout);
      log.end();
      done({ code: code ?? 1, signal });
    });
    child.stdin.end(prompt);
  });
}

function agentPrompt(masterPrompt, config, state, runId, sequence, previousHandoff) {
  return `${masterPrompt}\n\n---\n\n# Runner invocation context\n\nYou are the single active Codex for loop ${sequence} (${runId}). You own the decompiler mission, not a narrow ticket. Work autonomously: inspect the live repository, independently reassess the previous handoff, and choose the strategic boundary that most limits faithful execution, validated format support, or end-to-end evidence. Use this fresh context to complete the largest coherent, validated vertical slice supported by the evidence; connect adjacent parser, IR, materialization, executor, differential-validation, documentation and report layers whenever that closes one material mission gap. Do not stop at a micro-change, isolated test, or small commit merely to create a handoff. Do not ask for a plan or wait for human input unless an actual external resource or decision is required.\n\nCurrent state:\n\n\`\`\`json\n${JSON.stringify(state, null, 2)}\n\`\`\`\n\nPrevious accepted handoff: ${previousHandoff ?? "none; this is the first loop"}.\n\nThe external runner, not you, starts the next loop. Before exiting, run every command in agent-loop.config.json.validationCommands, create one local Git commit, and atomically create exactly one handoff named HANDOFF-${String(sequence).padStart(4, "0")}-<UTC timestamp>-<slug>.json in ${config.handoffDirectory}. Its runId must be \`${runId}\`, sequence must be ${sequence}, and endingState.gitCommit must equal the new HEAD commit. The handoff is your final repository-changing action. If missionStatus is continue, it must contain exactly one decision-useful nextRecommendedMilestone, one to three ordered nextSteps with acceptance criteria, and a bottlenecks array whose entries state description, impact, and evidence. Keep only unresolved bottlenecks and limitations; it is not a chronological work log.\n`;
}

async function start(config) {
  await ensureRuntime(config);
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
        state = { ...state, status: "failed", consecutiveFailures: state.consecutiveFailures + 1, lastCompletedAt: now() };
        await writeJsonAtomically(absolute(config.stateFile), state);
        console.error(`Loop ${sequence} rejected: ${failure}`);
        if (state.consecutiveFailures >= config.maxConsecutiveFailures) {
          console.error(`Stopping after ${state.consecutiveFailures} consecutive failures.`);
          return;
        }
      } else {
        state = {
          ...state,
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
        if (state.completedLoops < config.maxLoops) await new Promise((resolve) => setTimeout(resolve, config.cooldownSeconds * 1000));
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
  console.log(JSON.stringify({
    state,
    maxLoops: config.maxLoops,
    lockPresent: await exists(absolute(config.lockFile)),
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
