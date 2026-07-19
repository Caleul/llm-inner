import { join } from "node:path";

const PROVIDER_KINDS = new Set(["codex", "opencode", "kiro", "cursor"]);

export function validateProviders(providers) {
  if (!Array.isArray(providers) || providers.length === 0) {
    throw new Error("agent-loop.config.json must declare at least one provider.");
  }
  const ids = new Set();
  for (const provider of providers) {
    if (!provider || typeof provider.id !== "string" || !/^[a-z0-9-]+$/.test(provider.id)) {
      throw new Error("Every provider must have a lowercase id containing only letters, digits, and hyphens.");
    }
    if (ids.has(provider.id)) throw new Error(`Provider id is duplicated: ${provider.id}`);
    ids.add(provider.id);
    if (!PROVIDER_KINDS.has(provider.kind)) throw new Error(`Unsupported provider kind: ${provider.kind}`);
    if (typeof provider.command !== "string" || provider.command.length === 0) {
      throw new Error(`Provider ${provider.id} must declare its executable path.`);
    }
    if (typeof provider.model !== "string" || provider.model.length === 0) {
      throw new Error(`Provider ${provider.id} must declare a model.`);
    }
    if (provider.kind === "codex" && !["low", "medium", "high"].includes(provider.reasoningEffort)) {
      throw new Error(`Codex provider ${provider.id} must declare low, medium, or high reasoningEffort.`);
    }
    if (provider.environment && (typeof provider.environment !== "object" || Array.isArray(provider.environment)
      || Object.values(provider.environment).some((value) => typeof value !== "string"))) {
      throw new Error(`Provider ${provider.id} environment values must be strings.`);
    }
  }
}

export function providerAvailability(providers, retries = {}, at = Date.now()) {
  const available = [];
  const cooling = [];
  for (const provider of providers) {
    const retry = retries[provider.id];
    const retryAt = retry?.nextRetryAt ? Date.parse(retry.nextRetryAt) : Number.NaN;
    if (!Number.isFinite(retryAt) || retryAt <= at) available.push(provider);
    else cooling.push({ provider, retry, retryAt });
  }
  cooling.sort((left, right) => left.retryAt - right.retryAt);
  return { available, cooling, nextRetryAt: cooling[0]?.retry.nextRetryAt ?? null };
}

export function buildProviderInvocation(provider, { root, promptPath, outputPath }) {
  const promptReference = `Read ${promptPath} completely and follow it as the authoritative mission for this autonomous run. Work in ${root}. Do not ask for authorization or wait for interactive input.`;
  switch (provider.kind) {
    case "codex":
      return {
        command: provider.command,
        args: [
          "exec", "-C", root, "-m", provider.model,
          "-c", `model_reasoning_effort=${JSON.stringify(provider.reasoningEffort)}`,
          "--dangerously-bypass-approvals-and-sandbox", "--color", "never",
          "--output-last-message", outputPath, "-",
        ],
        stdin: "prompt",
      };
    case "opencode":
      return {
        command: provider.command,
        args: [
          "run", "--pure", "--dir", root, "--model", provider.model,
          ...(provider.variant ? ["--variant", provider.variant] : []),
          "--format", "json", "--dangerously-skip-permissions", promptReference,
        ],
        stdin: "none",
      };
    case "kiro":
      return {
        command: provider.command,
        args: ["chat", "--no-interactive", "--trust-all-tools", "--model", provider.model, promptReference],
        stdin: "none",
      };
    case "cursor":
      return {
        command: provider.command,
        args: [
          "-p", "--output-format", "stream-json", "--force", "--sandbox", "disabled",
          "--approve-mcps", "--trust", "--workspace", root, "--model", provider.model, promptReference,
        ],
        stdin: "none",
      };
    default:
      throw new Error(`Unsupported provider kind: ${provider.kind}`);
  }
}

export function providerRunPaths(runsDirectory, runId) {
  return {
    promptPath: join(runsDirectory, `${runId}.prompt.md`),
    outputPath: join(runsDirectory, `${runId}.last-message.md`),
    logPath: join(runsDirectory, `${runId}.log`),
    metadataPath: join(runsDirectory, `${runId}.json`),
  };
}
