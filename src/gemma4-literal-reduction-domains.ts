import { isDeepStrictEqual } from "node:util";

/**
 * A reduction bound is either an architecture constant or the live extent of
 * one declared tensor axis.  Tensor-axis bounds are deliberately name-bound:
 * calculation-graph instantiation rewrites the tensor name at each vision,
 * audio, and text call site, so no reader has to infer what `patches`, `width`,
 * `context`, or `in_features` means.
 */
export type Gemma4LiteralReductionExtent =
  | { kind: "constant"; value: number }
  | { kind: "tensor-axis"; tensor: string; axis: number };

export interface Gemma4LiteralReductionIndexDomain {
  index: string;
  startInclusive: 0;
  endExclusive: Gemma4LiteralReductionExtent;
  order: "ascending";
}

export interface Gemma4LiteralReductionDomainLanguage {
  kind: "gemma4-literal-reduction-domain-language";
  schemaVersion: 1;
  semantics: {
    constant: string;
    tensorAxis: string;
    interval: string;
    binding: string;
    failure: string;
  };
}

export interface Gemma4LiteralEvaluatedReductionIndexDomain {
  index: string;
  startInclusive: 0;
  endExclusive: number;
  order: "ascending";
}

export function gemma4LiteralReductionDomainLanguage(): Gemma4LiteralReductionDomainLanguage {
  return {
    kind: "gemma4-literal-reduction-domain-language",
    schemaVersion: 1,
    semantics: {
      constant: "endExclusive is the embedded positive exact safe integer value",
      tensorAxis: "endExclusive is tensorShapes[tensor][axis] after calculation-graph call-site binding",
      interval: "visit index=startInclusive..endExclusive-1 exactly once in ascending order unless the owning reduction schedule declares a different operation order",
      binding: "the domain index names the same scalar index token used by the owning formula and reduction schedule",
      failure: "missing tensors, missing axes, non-positive or non-safe extents, duplicate indices, and altered language contracts fail closed before reduction",
    },
  };
}

export function validateGemma4LiteralReductionDomainLanguage(
  language: Gemma4LiteralReductionDomainLanguage,
): void {
  if (!isDeepStrictEqual(language, gemma4LiteralReductionDomainLanguage())) {
    throw new Error("Programa literal Gemma 4 possui linguagem de domínios de redução ausente ou divergente.");
  }
}

export function validateGemma4LiteralReductionIndexDomains(
  domains: readonly Gemma4LiteralReductionIndexDomain[],
  owner: string,
): void {
  if (domains.length === 0) throw new Error(`${owner}: redução não declara domínios executáveis.`);
  const indices = new Set<string>();
  for (const domain of domains) {
    if (!/^[a-z][a-z0-9_]*$/.test(domain.index) || indices.has(domain.index) ||
      domain.startInclusive !== 0 || domain.order !== "ascending") {
      throw new Error(`${owner}: domínio de redução inválido ou duplicado para ${domain.index}.`);
    }
    indices.add(domain.index);
    validateExtent(domain.endExclusive, owner, domain.index);
  }
}

/** Binds subprogram-local tensor names to their concrete calculation-graph call site. */
export function bindGemma4LiteralReductionIndexDomains(
  domains: readonly Gemma4LiteralReductionIndexDomain[],
  bindings: ReadonlyMap<string, string>,
): Gemma4LiteralReductionIndexDomain[] {
  return domains.map((domain) => ({
    ...structuredClone(domain),
    endExclusive: domain.endExclusive.kind === "tensor-axis"
      ? { ...domain.endExclusive, tensor: bindings.get(domain.endExclusive.tensor) ?? domain.endExclusive.tensor }
      : structuredClone(domain.endExclusive),
  }));
}

/**
 * Evaluates only the serialized bound language.  The caller supplies shapes
 * of declared inputs or already-computed predecessors; no model metadata,
 * checkpoint, architecture code, or host default participates.
 */
export function evaluateGemma4LiteralReductionIndexDomains(
  language: Gemma4LiteralReductionDomainLanguage,
  domains: readonly Gemma4LiteralReductionIndexDomain[],
  tensorShapes: Readonly<Record<string, readonly number[]>>,
): Gemma4LiteralEvaluatedReductionIndexDomain[] {
  validateGemma4LiteralReductionDomainLanguage(language);
  validateGemma4LiteralReductionIndexDomains(domains, "avaliação de redução Gemma 4");
  return domains.map((domain) => ({
    index: domain.index,
    startInclusive: 0,
    endExclusive: evaluateExtent(domain.endExclusive, tensorShapes, domain.index),
    order: "ascending",
  }));
}

export function constantGemma4LiteralReductionExtent(value: number): Gemma4LiteralReductionExtent {
  const extent: Gemma4LiteralReductionExtent = { kind: "constant", value };
  validateExtent(extent, "construção de redução Gemma 4", "constant");
  return extent;
}

export function tensorAxisGemma4LiteralReductionExtent(
  tensor: string,
  axis: number,
): Gemma4LiteralReductionExtent {
  const extent: Gemma4LiteralReductionExtent = { kind: "tensor-axis", tensor, axis };
  validateExtent(extent, "construção de redução Gemma 4", "tensor-axis");
  return extent;
}

function validateExtent(extent: Gemma4LiteralReductionExtent, owner: string, index: string): void {
  if (extent.kind === "constant") {
    if (!Number.isSafeInteger(extent.value) || extent.value <= 0) {
      throw new Error(`${owner}: extent constante inválido para ${index}.`);
    }
    return;
  }
  if (extent.kind !== "tensor-axis" || !extent.tensor || !Number.isSafeInteger(extent.axis) || extent.axis < 0) {
    throw new Error(`${owner}: extent tensor-axis inválido para ${index}.`);
  }
}

function evaluateExtent(
  extent: Gemma4LiteralReductionExtent,
  tensorShapes: Readonly<Record<string, readonly number[]>>,
  index: string,
): number {
  if (extent.kind === "constant") return extent.value;
  const shape = tensorShapes[extent.tensor];
  const value = shape?.[extent.axis];
  if (!shape || value === undefined || shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0) ||
    !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${index}: redução não pode resolver ${extent.tensor}.shape[${extent.axis}].`);
  }
  return value;
}
