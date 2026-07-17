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
const laneReductionOrders = (optional("--lane-reduction-orders") ?? "ascending,descending,balanced-pairwise").split(",").map(laneReductionOrder);
if (traces.length < 2 || maxReadMiB <= 0 || laneCounts.some((count) => count < 2)) throw new Error("Opções numéricas inválidas; informe ao menos dois --trace distintos.");
const maxReadBytes = maxReadMiB * 1024 * 1024;
if (!Number.isSafeInteger(maxReadBytes)) throw new Error("--max-read-mib excede limite seguro.");
const profiles: Gemma4LinearReductionProfile[] = [
  { id: "ordered-f32", accumulationDtype: "F32", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } },
  { id: "ordered-f64", accumulationDtype: "F64", reduction: { kind: "ordered-scalar", indexOrder: "ascending" } },
  ...[...new Set(laneCounts)].flatMap((laneCount) => [...new Set(laneReductionOrders)].flatMap((laneReductionOrder) => [
    laneProfile("interleaved-f32-lanes", laneCount, laneReductionOrder),
    laneProfile("interleaved-fma-lanes", laneCount, laneReductionOrder),
  ])),
];
const report = await probeGemma4LiteralLinearReductionProfiles({ artifact, traces, operationId, profiles, maxReadBytes });
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
