import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { probeGemma4LiteralLinearReductionProfiles, type Gemma4LinearReductionProfile } from "./gemma4-linear-reduction-probe.js";
import type { ReductionSchedule } from "./types.js";

const argv = process.argv.slice(2);
const artifact = resolve(required("--artifact"));
const traces = values("--trace").map((trace) => resolve(trace));
const operationId = required("--operation-id");
const output = resolve(required("--output"));
const maxReadMiB = integer(optional("--max-read-mib") ?? "16", "--max-read-mib");
const laneCounts = (optional("--lane-counts") ?? "2,4,8,16,32,64,128").split(",").map((entry) => integer(entry, "--lane-counts"));
const tiledLaneCounts = optional("--tiled-lane-counts")?.split(",").map((entry) => integer(entry, "--tiled-lane-counts")) ?? [];
const tiledTermsPerLane = optional("--tiled-terms-per-lane")?.split(",").map((entry) => integer(entry, "--tiled-terms-per-lane")) ?? [];
const blockedTermsPerBlock = optional("--blocked-terms-per-block")?.split(",").map((entry) => integer(entry, "--blocked-terms-per-block")) ?? [];
const blockedTiledLaneCounts = optional("--blocked-tiled-lane-counts")?.split(",").map((entry) => integer(entry, "--blocked-tiled-lane-counts")) ?? [];
const blockedTiledTermsPerLane = optional("--blocked-tiled-terms-per-lane")?.split(",").map((entry) => integer(entry, "--blocked-tiled-terms-per-lane")) ?? [];
const laneReductionOrders = (optional("--lane-reduction-orders") ?? "ascending,descending,balanced-pairwise").split(",").map(laneReductionOrder);
const minDistinctInputs = integer(optional("--min-distinct-inputs") ?? "1", "--min-distinct-inputs");
const selectedProfileIds = values("--profile-id");
if (traces.length < 2 || maxReadMiB <= 0 || laneCounts.some((count) => count < 2) || tiledLaneCounts.some((count) => count < 2) || tiledTermsPerLane.some((count) => count < 2) || blockedTermsPerBlock.some((count) => count < 2) || blockedTiledLaneCounts.some((count) => count < 2) || blockedTiledTermsPerLane.some((count) => count < 2) || (tiledLaneCounts.length === 0) !== (tiledTermsPerLane.length === 0) || (blockedTiledLaneCounts.length === 0) !== (blockedTiledTermsPerLane.length === 0)) {
  throw new Error("Opções numéricas inválidas; informe ao menos dois --trace distintos e os dois parâmetros tiled quando usar redução tiled.");
}
const maxReadBytes = maxReadMiB * 1024 * 1024;
if (!Number.isSafeInteger(maxReadBytes)) throw new Error("--max-read-mib excede limite seguro.");
const candidateProfiles: Gemma4LinearReductionProfile[] = [
  { id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } },
  { id: "ordered-fma", accumulationDtype: "F32", reduction: { kind: "ordered-fma", indexOrder: "ascending" } },
  { id: "ordered-f64", accumulationDtype: "F64", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } },
  armNeonBf16DotProfile(32, 4, "ascending"),
  armNeonBf16DotProfile(32, 4, "pairwise"),
  armNeonBf16DotProfile(64, 8, "ascending"),
  armNeonBf16DotProfile(64, 8, "pairwise"),
  ...[...new Set(blockedTermsPerBlock)].flatMap((termsPerBlock) => [
    blockedProfile(termsPerBlock, "ascending", "separately-rounded-f32"),
    blockedProfile(termsPerBlock, "ascending", "fused-fma"),
    blockedProfile(termsPerBlock, "descending", "separately-rounded-f32"),
    blockedProfile(termsPerBlock, "descending", "fused-fma"),
  ]),
  ...[...new Set(blockedTiledLaneCounts)].flatMap((laneCount) => [...new Set(blockedTiledTermsPerLane)].flatMap((termsPerLane) => [...new Set(laneReductionOrders)].flatMap((laneReductionOrder) => [
    blockedTiledProfile(laneCount, termsPerLane, laneReductionOrder, "separately-rounded-f32"),
    blockedTiledProfile(laneCount, termsPerLane, laneReductionOrder, "fused-fma"),
  ]))),
  ...[...new Set(laneCounts)].flatMap((laneCount) => [...new Set(laneReductionOrders)].flatMap((laneReductionOrder) => [
    laneProfile("interleaved-f32-lanes", laneCount, laneReductionOrder),
    laneProfile("interleaved-fma-lanes", laneCount, laneReductionOrder),
  ])),
  ...[...new Set(tiledLaneCounts)].flatMap((laneCount) => [...new Set(tiledTermsPerLane)].flatMap((termsPerLane) => [...new Set(laneReductionOrders)].flatMap((laneReductionOrder) => [
    tiledLaneProfile("tiled-f32-lanes", laneCount, termsPerLane, laneReductionOrder),
    tiledLaneProfile("tiled-fma-lanes", laneCount, termsPerLane, laneReductionOrder),
  ]))),
];
const profiles = selectProfiles(candidateProfiles, selectedProfileIds);
const report = await probeGemma4LiteralLinearReductionProfiles({ artifact, traces, operationId, profiles, maxReadBytes, minDistinctInputs });
await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, "utf8");
console.log(`Probe de redução Gemma 4 escrito em ${output}; perfis exatos: ${report.exactProfileIds.join(", ") || "nenhum"}.`);

function optional(flag: string): string | undefined {
  const index = argv.indexOf(flag);
  if (index < 0) return undefined;
  const value = argv[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`Valor ausente para ${flag}.`);
  return value;
}

function values(flag: string): string[] {
  const result: string[] = [];
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] !== flag) continue;
    const value = argv[index + 1];
    if (!value || value.startsWith("--")) throw new Error(`Valor ausente para ${flag}.`);
    result.push(value);
    index += 1;
  }
  return result;
}

/**
 * A broad candidate sweep is useful while discovering a native reduction,
 * but subsequent source-removed campaigns must be able to re-test only the
 * profiles already implicated by evidence.  Selection is by generated
 * immutable ID, never by a partial schedule or a shape-derived default.
 */
function selectProfiles(candidates: readonly Gemma4LinearReductionProfile[], requestedIds: readonly string[]): Gemma4LinearReductionProfile[] {
  if (requestedIds.length === 0) return [...candidates];
  const requested = new Set<string>();
  for (const id of requestedIds) {
    if (!id || requested.has(id)) throw new Error("--profile-id requer IDs não vazios e sem repetição.");
    requested.add(id);
  }
  const selected = candidates.filter((profile) => requested.has(profile.id));
  if (selected.length !== requested.size) {
    const available = new Set(candidates.map((profile) => profile.id));
    const unknown = requestedIds.filter((id) => !available.has(id));
    throw new Error(`--profile-id não reconhece: ${unknown.join(", ")}.`);
  }
  return selected;
}

function required(flag: string): string {
  const value = optional(flag);
  if (!value) throw new Error(`Valor ausente para ${flag}.`);
  return value;
}

function integer(value: string, flag: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed)) throw new Error(`${flag} requer inteiro seguro.`);
  return parsed;
}

function laneReductionOrder(value: string): Extract<ReductionSchedule, { kind: "interleaved-f32-lanes" }>['laneReductionOrder'] {
  if (value === "ascending" || value === "descending" || value === "balanced-pairwise") return value;
  throw new Error("--lane-reduction-orders requer ascending, descending ou balanced-pairwise.");
}

function laneProfile(
  kind: "interleaved-f32-lanes" | "interleaved-fma-lanes",
  laneCount: number,
  laneReductionOrder: Extract<ReductionSchedule, { kind: "interleaved-f32-lanes" }>['laneReductionOrder'],
): Gemma4LinearReductionProfile {
  return { id: `${kind}-${laneCount}-${laneReductionOrder}`, accumulationDtype: "F32", reduction: { kind, laneCount, inputLane: "index-modulo-lane-count", laneReductionOrder } };
}

function armNeonBf16DotProfile(laneCount: 32 | 64, lanesPerRegister: 4 | 8, horizontalFold: "ascending" | "pairwise"): Gemma4LinearReductionProfile {
  return {
    id: `arm-neon-bf16-dot-fma-${laneCount}-${horizontalFold}`,
    accumulationDtype: "F32",
    reduction: {
      kind: "arm-neon-bf16-dot-fma", laneCount, registerCount: 8, lanesPerRegister,
      inputLane: "index-modulo-vector-lane-count", horizontalFold,
    },
  };
}

function tiledLaneProfile(
  kind: "tiled-f32-lanes" | "tiled-fma-lanes",
  laneCount: number,
  termsPerLane: number,
  laneReductionOrder: Extract<ReductionSchedule, { kind: "tiled-f32-lanes" }>['laneReductionOrder'],
): Gemma4LinearReductionProfile {
  return {
    id: `${kind}-${laneCount}-terms-${termsPerLane}-${laneReductionOrder}`,
    accumulationDtype: "F32",
    reduction: { kind, laneCount, termsPerLane, inputLane: "tile-contiguous-terms", laneReductionOrder },
  };
}

function blockedProfile(
  termsPerBlock: number,
  termOrder: "ascending" | "descending",
  productBoundary: "separately-rounded-f32" | "fused-fma",
): Gemma4LinearReductionProfile {
  return {
    id: `blocked-f32-terms-${termsPerBlock}-${termOrder}-${productBoundary}`,
    accumulationDtype: "F32",
    reduction: { kind: "blocked-f32-terms", termsPerBlock, inputBlock: "contiguous-terms", termOrder, productBoundary, blockOrder: "ascending" },
  };
}

function blockedTiledProfile(
  laneCount: number,
  termsPerLane: number,
  laneReductionOrder: Extract<ReductionSchedule, { kind: "tiled-f32-lanes" }>['laneReductionOrder'],
  productBoundary: "separately-rounded-f32" | "fused-fma",
): Gemma4LinearReductionProfile {
  return {
    id: `blocked-tiled-f32-lanes-${laneCount}-terms-${termsPerLane}-${laneReductionOrder}-${productBoundary}`,
    accumulationDtype: "F32",
    reduction: { kind: "blocked-tiled-f32-lanes", laneCount, termsPerLane, inputBlock: "tile-contiguous-terms", laneReductionOrder, productBoundary, blockOrder: "ascending" },
  };
}
