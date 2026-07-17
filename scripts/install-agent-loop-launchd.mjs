#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";

const root = resolve(import.meta.dirname, "..");
const label = "tech.lilka.llm-inner-agent-loop";
const destination = join(homedir(), "Library", "LaunchAgents", `${label}.plist`);
const template = join(root, "launchd", `${label}.plist`);
const uid = process.getuid?.();
if (uid === undefined) throw new Error("Installing the LaunchAgent requires a macOS user id.");

function launchctl(args, allowFailure = false) {
  return new Promise((resolveCommand, reject) => {
    const child = spawn("launchctl", args, { stdio: "inherit" });
    child.once("error", reject);
    child.once("close", (code) => {
      if (code === 0 || allowFailure) resolveCommand();
      else reject(new Error(`launchctl ${args.join(" ")} exited ${code}`));
    });
  });
}

async function main() {
  await mkdir(join(homedir(), "Library", "LaunchAgents"), { recursive: true });
  const rendered = (await readFile(template, "utf8")).replaceAll("__LLM_INNER_ROOT__", root);
  await writeFile(destination, rendered, "utf8");
  await launchctl(["bootout", `gui/${uid}`, destination], true);
  await launchctl(["bootstrap", `gui/${uid}`, destination]);
  console.log(`Installed and started ${label}: ${destination}`);
}

main().catch((error) => { console.error(error instanceof Error ? error.stack ?? error.message : error); process.exitCode = 1; });
