import assert from "node:assert/strict";
import test from "node:test";
import {
  buildProviderInvocation,
  providerAvailability,
  validateProviders,
} from "./agent-loop-providers.mjs";

const root = "/repo";
const promptPath = "/repo/.agent-loop/runs/run.prompt.md";
const outputPath = "/repo/.agent-loop/runs/run.last-message.md";

const providers = [
  { id: "sol", kind: "codex", command: "/bin/codex", model: "gpt-5.6-sol", reasoningEffort: "high" },
  { id: "spark", kind: "codex", command: "/bin/codex", model: "gpt-5.3-codex-spark", reasoningEffort: "high" },
  { id: "opencode", kind: "opencode", command: "/bin/opencode", model: "opencode/north-mini-code-free", environment: { XDG_DATA_HOME: "/data" } },
  { id: "kiro", kind: "kiro", command: "/bin/kiro", model: "gpt-5.6-sol" },
  { id: "cursor", kind: "cursor", command: "/bin/cursor", model: "gpt-5.6-sol-high" },
];

test("validates the ordered autonomous provider set", () => {
  assert.doesNotThrow(() => validateProviders(providers));
  assert.throws(() => validateProviders([{ ...providers[0], id: "Bad ID" }]));
  assert.throws(() => validateProviders([providers[0], providers[0]]));
});

test("selects providers in configured order and reports the earliest cooldown", () => {
  const availability = providerAvailability(providers, {
    sol: { nextRetryAt: "2026-07-19T20:02:00.000Z" },
    spark: { nextRetryAt: "2026-07-19T20:01:00.000Z" },
  }, Date.parse("2026-07-19T20:00:00.000Z"));
  assert.deepEqual(availability.available.map((provider) => provider.id), ["opencode", "kiro", "cursor"]);
  assert.equal(availability.nextRetryAt, "2026-07-19T20:01:00.000Z");
});

test("builds non-interactive approval-free invocations for every provider", () => {
  const invocations = Object.fromEntries(providers.map((provider) => [
    provider.id,
    buildProviderInvocation(provider, { root, promptPath, outputPath }),
  ]));
  assert.deepEqual(invocations.sol.args.slice(0, 4), ["exec", "-C", root, "-m"]);
  assert.ok(invocations.sol.args.includes("--dangerously-bypass-approvals-and-sandbox"));
  assert.ok(invocations.opencode.args.includes("--dangerously-skip-permissions"));
  assert.ok(invocations.opencode.args.includes("--pure"));
  assert.ok(invocations.kiro.args.includes("--no-interactive"));
  assert.ok(invocations.kiro.args.includes("--trust-all-tools"));
  assert.ok(invocations.cursor.args.includes("--force"));
  assert.ok(invocations.cursor.args.includes("--trust"));
  assert.ok(invocations.cursor.args.every((argument) => !argument.includes("undefined")));
});
