import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";

export const GEMMA4_LITERAL_INTEGRITY_SECTION_NAMES = [
  "sourceIdentity",
  "authoritativeExecution",
  "numericPolicy",
  "inputs",
  "constantMetadata",
  "unreachableConstants",
  "storageDecoders",
  "denseDecoderLanguage",
  "program",
  "assignments",
  "calculationDomains",
  "learnedOperands",
  "scalarCalculations",
  "formulaLanguage",
  "transcendentalPrograms",
  "numericLiterals",
  "calculationGraph",
  "realSimplifiedProgram",
  "fidelityGate",
  "forwardControl",
  "inputContract",
  "outputContract",
  "outputs",
  "generation",
  "payloadIntegrity",
] as const;

export type Gemma4LiteralIntegritySectionName = typeof GEMMA4_LITERAL_INTEGRITY_SECTION_NAMES[number];
export type Gemma4LiteralIntegritySections = Record<Gemma4LiteralIntegritySectionName, unknown>;

export interface Gemma4LiteralArtifactIntegrityEntry {
  name: Gemma4LiteralIntegritySectionName;
  canonicalBytes: number;
  sha256: string;
}

/**
 * Binds every non-payload field and every payload commitment in the JSON.
 * Payload bytes stay outside this manifest because their per-tensor hashes are
 * already committed by `payloadIntegrity`; the manifest commits that complete
 * ordered table together with constant metadata and executable semantics.
 */
export interface Gemma4LiteralArtifactIntegrityManifest {
  kind: "gemma4-literal-artifact-integrity-manifest";
  schemaVersion: 1;
  algorithm: "SHA-256";
  canonicalization: "UTF-8 bytes of ECMAScript JSON.stringify for each named section, in declared order";
  sections: Gemma4LiteralArtifactIntegrityEntry[];
  rootSha256: string;
}

export function buildGemma4LiteralArtifactIntegrityManifest(
  sections: Gemma4LiteralIntegritySections,
): Gemma4LiteralArtifactIntegrityManifest {
  const entries = GEMMA4_LITERAL_INTEGRITY_SECTION_NAMES.map((name) => digestSection(name, sections[name]));
  return {
    kind: "gemma4-literal-artifact-integrity-manifest",
    schemaVersion: 1,
    algorithm: "SHA-256",
    canonicalization: "UTF-8 bytes of ECMAScript JSON.stringify for each named section, in declared order",
    sections: entries,
    rootSha256: rootDigest(entries),
  };
}

export function validateGemma4LiteralArtifactIntegrityManifest(
  manifest: Gemma4LiteralArtifactIntegrityManifest,
  sections: Gemma4LiteralIntegritySections,
): void {
  const expected = buildGemma4LiteralArtifactIntegrityManifest(sections);
  if (!isDeepStrictEqual(manifest, expected)) {
    throw new Error("Artefato literal Gemma 4 possui compromisso estrutural incompleto ou divergente.");
  }
}

/**
 * Validates the manifest as a self-contained commitment without requiring all
 * committed sections to be materialized. Runtime projections use this before
 * checking only the sections they actually execute.
 */
export function validateGemma4LiteralArtifactIntegrityCommitment(
  manifest: Gemma4LiteralArtifactIntegrityManifest,
): void {
  if (manifest.kind !== "gemma4-literal-artifact-integrity-manifest" || manifest.schemaVersion !== 1 ||
    manifest.algorithm !== "SHA-256" ||
    manifest.canonicalization !== "UTF-8 bytes of ECMAScript JSON.stringify for each named section, in declared order" ||
    !Array.isArray(manifest.sections) || manifest.sections.length !== GEMMA4_LITERAL_INTEGRITY_SECTION_NAMES.length) {
    throw new Error("Manifesto de integridade literal Gemma 4 inválido.");
  }
  for (let index = 0; index < GEMMA4_LITERAL_INTEGRITY_SECTION_NAMES.length; index += 1) {
    const expectedName = GEMMA4_LITERAL_INTEGRITY_SECTION_NAMES[index], entry = manifest.sections[index];
    if (!entry || entry.name !== expectedName || !Number.isSafeInteger(entry.canonicalBytes) || entry.canonicalBytes < 0 || !/^[0-9a-f]{64}$/.test(entry.sha256)) {
      throw new Error(`Manifesto de integridade literal Gemma 4 diverge na seção ${expectedName}.`);
    }
  }
  if (!/^[0-9a-f]{64}$/.test(manifest.rootSha256) || rootDigest(manifest.sections) !== manifest.rootSha256) {
    throw new Error("Manifesto de integridade literal Gemma 4 diverge da raiz declarada.");
  }
}

/** Proves that one materialized runtime section is the artifact-committed value. */
export function validateGemma4LiteralArtifactIntegritySection(
  manifest: Gemma4LiteralArtifactIntegrityManifest,
  name: Gemma4LiteralIntegritySectionName,
  value: unknown,
): void {
  const expected = manifest.sections.find((entry) => entry.name === name), actual = digestSection(name, value);
  if (!expected || !isDeepStrictEqual(expected, actual)) {
    throw new Error(`Projeção runtime Gemma 4 diverge da seção autenticada ${name}.`);
  }
}

function digestSection(
  name: Gemma4LiteralIntegritySectionName,
  value: unknown,
): Gemma4LiteralArtifactIntegrityEntry {
  const canonical = JSON.stringify(value);
  if (canonical === undefined) throw new Error(`Seção estrutural Gemma 4 ${name} não pode ser serializada.`);
  return {
    name,
    canonicalBytes: Buffer.byteLength(canonical, "utf8"),
    sha256: createHash("sha256").update(canonical, "utf8").digest("hex"),
  };
}

function rootDigest(entries: readonly Gemma4LiteralArtifactIntegrityEntry[]): string {
  const digest = createHash("sha256");
  for (const entry of entries) digest.update(`${entry.name}\0${entry.canonicalBytes}\0${entry.sha256}\n`, "utf8");
  return digest.digest("hex");
}
