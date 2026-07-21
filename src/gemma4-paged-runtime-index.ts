import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, rename, stat, writeFile } from "node:fs/promises";
import { isDeepStrictEqual } from "node:util";
import type { Gemma4LiteralGreedyGenerationProgram } from "./gemma4-composite-literal.js";
import { openGemma4CompositeLiteralArtifact, type IndexedLiteralConstant, type OpenGemma4CompositeLiteralArtifact } from "./gemma4-composite-literal-reader.js";
import type { Gemma4CompositeProgram } from "./gemma4-composite.js";
import {
  validateGemma4LiteralArtifactIntegrityCommitment,
  validateGemma4LiteralArtifactIntegritySection,
  type Gemma4LiteralArtifactIntegrityManifest,
} from "./gemma4-literal-artifact-integrity.js";
import {
  assertGemma4VectorizedRealLoweringPlanMatches,
  validateGemma4VectorizedRealLoweringPlan,
  type Gemma4VectorizedRealLoweringContract,
  type Gemma4VectorizedRealLoweringPlan,
} from "./gemma4-vectorized-real-lowering.js";

const MAX_RUNTIME_INDEX_BYTES = 64 * 1024 * 1024;

export interface Gemma4PagedRuntimeIndexDescriptor {
  file: string;
  schemaVersion: 2;
  bytes: number;
  sha256: string;
  constantPoolSha256: string;
  integrityRootSha256: string;
}

export interface Gemma4PagedRuntimeLoweringCertificate {
  planSha256: string;
  contract: Gemma4VectorizedRealLoweringContract;
}

interface Gemma4PagedRuntimeIndexSnapshot {
  kind: "gemma4-paged-runtime-index";
  schemaVersion: 2;
  artifact: { bytes: number; sha256: string; integrityRootSha256: string };
  constants: IndexedLiteralConstant[];
  program: Gemma4CompositeProgram;
  generation: Gemma4LiteralGreedyGenerationProgram;
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest;
  realLowering?: Gemma4PagedRuntimeLoweringCertificate;
}

export interface OpenGemma4PagedRuntimeArtifact {
  runtimeIndexSchemaVersion: 2;
  artifact: string;
  artifactBytes: number;
  constants: ReadonlyMap<string, IndexedLiteralConstant>;
  program: Gemma4CompositeProgram;
  generation: Gemma4LiteralGreedyGenerationProgram;
  generationValidation: "artifact-integrity-section-v1";
  integrityManifest: Gemma4LiteralArtifactIntegrityManifest;
  realLowering?: Gemma4PagedRuntimeLoweringCertificate;
  close(): Promise<void>;
}

export async function createGemma4PagedRuntimeIndex(
  artifact: string,
  output: string,
  constantPoolSha256: string,
  loweringPlan?: { path: string; sha256: string },
): Promise<Gemma4PagedRuntimeIndexDescriptor> {
  assertSha256(constantPoolSha256, "SHA-256 do constant pool");
  const opened = await openGemma4CompositeLiteralArtifact(artifact);
  try {
    const realLowering = loweringPlan ? await createLoweringCertificate(opened, loweringPlan) : undefined;
    const snapshot: Gemma4PagedRuntimeIndexSnapshot = {
      kind: "gemma4-paged-runtime-index",
      schemaVersion: 2,
      artifact: { bytes: opened.artifactBytes, sha256: constantPoolSha256, integrityRootSha256: opened.integrityManifest.rootSha256 },
      constants: [...opened.constants.values()],
      program: opened.program,
      generation: opened.generation,
      integrityManifest: opened.integrityManifest,
      ...(realLowering ? { realLowering } : {}),
    };
    validateRuntimeSnapshot(snapshot, opened.artifactBytes, constantPoolSha256);
    const temporary = `${output}.next-${process.pid}`;
    await writeFile(temporary, `${JSON.stringify(snapshot)}\n`, { encoding: "utf8", flag: "wx" });
    await rename(temporary, output);
    const info = await stat(output), sha256 = await sha256File(output);
    return { file: output, schemaVersion: 2, bytes: info.size, sha256, constantPoolSha256, integrityRootSha256: opened.integrityManifest.rootSha256 };
  } finally { await opened.close(); }
}

export async function openGemma4PagedRuntimeArtifact(
  artifact: string,
  runtimeIndex: { path: string; sha256: string; constantPoolSha256: string },
): Promise<OpenGemma4PagedRuntimeArtifact | OpenGemma4CompositeLiteralArtifact> {
  assertSha256(runtimeIndex.sha256, "SHA-256 do índice runtime");
  assertSha256(runtimeIndex.constantPoolSha256, "SHA-256 do constant pool");
  const [artifactInfo, indexInfo] = await Promise.all([stat(artifact), stat(runtimeIndex.path)]);
  if (!artifactInfo.isFile() || artifactInfo.size <= 0) throw new Error(`Artefato literal Gemma 4 inválido: ${artifact}.`);
  if (!indexInfo.isFile() || indexInfo.size <= 0 || indexInfo.size > MAX_RUNTIME_INDEX_BYTES) {
    // Schema 1 is the larger audit index retained for backward compatibility.
    return openGemma4CompositeLiteralArtifact(artifact, runtimeIndex);
  }
  const bytes = await readFile(runtimeIndex.path);
  const actualSha256 = createHash("sha256").update(bytes).digest("hex");
  if (actualSha256 !== runtimeIndex.sha256) throw new Error("Índice runtime Gemma 4 diverge do SHA-256 declarado.");
  const snapshot = JSON.parse(bytes.toString("utf8")) as Partial<Gemma4PagedRuntimeIndexSnapshot>;
  if (snapshot.kind !== "gemma4-paged-runtime-index" || snapshot.schemaVersion !== 2) {
    return openGemma4CompositeLiteralArtifact(artifact, runtimeIndex);
  }
  validateRuntimeSnapshot(snapshot as Gemma4PagedRuntimeIndexSnapshot, artifactInfo.size, runtimeIndex.constantPoolSha256);
  const constants = new Map<string, IndexedLiteralConstant>();
  for (const constant of snapshot.constants!) constants.set(constant.name, constant);
  return {
    runtimeIndexSchemaVersion: 2,
    artifact,
    artifactBytes: artifactInfo.size,
    constants,
    program: snapshot.program!,
    generation: snapshot.generation!,
    generationValidation: "artifact-integrity-section-v1",
    integrityManifest: snapshot.integrityManifest!,
    ...(snapshot.realLowering ? { realLowering: snapshot.realLowering } : {}),
    close: async () => {},
  };
}

export function assertGemma4VectorizedRealLoweringPlanMatchesRuntime(
  persisted: unknown,
  actualPlanSha256: string,
  artifact: Pick<OpenGemma4PagedRuntimeArtifact, "integrityManifest" | "realLowering">,
): asserts persisted is Gemma4VectorizedRealLoweringPlan {
  const certificate = artifact.realLowering;
  if (!certificate) throw new Error("Índice runtime Gemma 4 não contém certificado do lowering real.");
  assertSha256(actualPlanSha256, "SHA-256 do plano de lowering");
  validateGemma4VectorizedRealLoweringPlan(persisted);
  const realSection = artifact.integrityManifest.sections.find((entry) => entry.name === "realSimplifiedProgram");
  if (actualPlanSha256 !== certificate.planSha256 || !isDeepStrictEqual(persisted.contract, certificate.contract) ||
    persisted.contract.source.artifactIntegritySha256 !== artifact.integrityManifest.rootSha256 ||
    persisted.contract.source.realSimplifiedProgramSha256 !== realSection?.sha256) {
    throw new Error("Plano de lowering persistido diverge do certificado runtime autenticado.");
  }
}

async function createLoweringCertificate(
  artifact: OpenGemma4CompositeLiteralArtifact,
  loweringPlan: { path: string; sha256: string },
): Promise<Gemma4PagedRuntimeLoweringCertificate> {
  assertSha256(loweringPlan.sha256, "SHA-256 do plano de lowering");
  const bytes = await readFile(loweringPlan.path), actualSha256 = createHash("sha256").update(bytes).digest("hex");
  if (actualSha256 !== loweringPlan.sha256) throw new Error("Plano de lowering diverge do SHA-256 declarado durante a compilação do índice.");
  const persisted = JSON.parse(bytes.toString("utf8")) as unknown;
  assertGemma4VectorizedRealLoweringPlanMatches(persisted, artifact.realSimplifiedProgram, artifact.calculationGraph, artifact.integrityManifest);
  return { planSha256: actualSha256, contract: persisted.contract };
}

function validateRuntimeSnapshot(snapshot: Gemma4PagedRuntimeIndexSnapshot, artifactBytes: number, constantPoolSha256: string): void {
  if (snapshot.kind !== "gemma4-paged-runtime-index" || snapshot.schemaVersion !== 2 || !snapshot.artifact ||
    snapshot.artifact.bytes !== artifactBytes || snapshot.artifact.sha256 !== constantPoolSha256 ||
    !Array.isArray(snapshot.constants) || snapshot.constants.length === 0 || !snapshot.program || !snapshot.generation || !snapshot.integrityManifest) {
    throw new Error("Índice runtime paginado Gemma 4 não corresponde ao constant pool declarado.");
  }
  assertSha256(snapshot.artifact.integrityRootSha256, "raiz de integridade do índice runtime");
  validateGemma4LiteralArtifactIntegrityCommitment(snapshot.integrityManifest);
  if (snapshot.integrityManifest.rootSha256 !== snapshot.artifact.integrityRootSha256) throw new Error("Índice runtime paginado diverge da raiz estrutural declarada.");
  const names = new Set<string>();
  for (const constant of snapshot.constants) {
    validateRuntimeConstant(constant);
    if (names.has(constant.name)) throw new Error(`Índice runtime paginado contém constante duplicada: ${constant.name}.`);
    names.add(constant.name);
  }
  validateGemma4LiteralArtifactIntegritySection(snapshot.integrityManifest, "constantMetadata", constantMetadata(snapshot.constants));
  validateGemma4LiteralArtifactIntegritySection(snapshot.integrityManifest, "program", snapshot.program);
  validateGemma4LiteralArtifactIntegritySection(snapshot.integrityManifest, "generation", snapshot.generation);
  if (snapshot.realLowering) {
    assertSha256(snapshot.realLowering.planSha256, "SHA-256 do plano de lowering");
    const contract = snapshot.realLowering.contract, realSection = snapshot.integrityManifest.sections.find((entry) => entry.name === "realSimplifiedProgram");
    if (contract?.kind !== "gemma4-vectorized-real-lowering-contract" || contract.schemaVersion !== 3 || contract.execution?.directlyExecutesGlobalFormula !== true ||
      contract.source?.artifactIntegritySha256 !== snapshot.integrityManifest.rootSha256 ||
      contract.source.realSimplifiedProgramSha256 !== realSection?.sha256 || !/^[0-9a-f]{64}$/.test(contract.functionBindingsSha256) ||
      !/^[0-9a-f]{64}$/.test(contract.source.outputBindingsSha256) || !/^[0-9a-f]{64}$/.test(contract.source.standaloneSsaOutputsSha256) ||
      Object.values(contract.source.outputFamilies ?? {}).reduce((sum, family) => sum + (Number.isSafeInteger(family.dimensions) ? family.dimensions : 0), 0) !== contract.source.outputFunctions) {
      throw new Error("Certificado de lowering do índice runtime paginado é inválido.");
    }
  }
}

function constantMetadata(constants: readonly IndexedLiteralConstant[]): unknown[] {
  return constants.map(({ payloadOffset: _offset, payloadBase64Characters: _characters, payloadBytes: _bytes, ...metadata }) => metadata);
}

function validateRuntimeConstant(constant: IndexedLiteralConstant): void {
  if (!constant?.name || (constant.storageDtype !== "F32" && constant.storageDtype !== "F16" && constant.storageDtype !== "BF16") ||
    constant.layout !== "row-major" || constant.byteOrder !== "little-endian" || constant.encoding !== "base64" || constant.quantization ||
    !validShape(constant.storageShape) || !sameShape(constant.storageShape, constant.logicalShape) ||
    !Number.isSafeInteger(constant.payloadOffset) || constant.payloadOffset < 0 || !Number.isSafeInteger(constant.payloadBase64Characters) || constant.payloadBase64Characters < 1) {
    throw new Error(`${constant?.name ?? "constante"}: índice runtime paginado requer metadados densos válidos.`);
  }
  const payloadBytes = product(constant.storageShape) * (constant.storageDtype === "F32" ? 4 : 2);
  if (!Number.isSafeInteger(payloadBytes) || payloadBytes !== constant.payloadBytes) throw new Error(`${constant.name}: payload runtime não corresponde ao shape e dtype.`);
}

function validShape(shape: readonly number[]): boolean { return Array.isArray(shape) && shape.every((dimension) => Number.isSafeInteger(dimension) && dimension > 0); }
function sameShape(left: readonly number[], right: readonly number[]): boolean { return Array.isArray(right) && left.length === right.length && left.every((value, index) => value === right[index]); }
function product(shape: readonly number[]): number { return shape.reduce((total, dimension) => total * dimension, 1); }
function assertSha256(value: string, label: string): void { if (!/^[0-9a-f]{64}$/.test(value)) throw new Error(`${label} inválido.`); }
async function sha256File(path: string): Promise<string> { const digest = createHash("sha256"); for await (const chunk of createReadStream(path)) digest.update(chunk as Buffer); return digest.digest("hex"); }
