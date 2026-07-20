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
