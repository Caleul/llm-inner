import { once } from "node:events";
import type { Writable } from "node:stream";

export function asObject(value: unknown, label: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label} precisa ser um objeto JSON.`);
  }
  return value as Record<string, unknown>;
}

export function numberFrom(
  object: Record<string, unknown>,
  keys: string[],
  label: string,
  fallback?: number,
): number {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  if (fallback !== undefined) return fallback;
  throw new Error(`Configuração obrigatória ausente: ${label} (${keys.join(" | ")}).`);
}

export function optionalNumber(
  object: Record<string, unknown>,
  keys: string[],
): number | undefined {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return undefined;
}

export function stringFrom(
  object: Record<string, unknown>,
  keys: string[],
  label: string,
  fallback?: string,
): string {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "string" && value.length > 0) return value;
  }
  if (fallback !== undefined) return fallback;
  throw new Error(`Configuração obrigatória ausente: ${label} (${keys.join(" | ")}).`);
}

export function optionalString(
  object: Record<string, unknown>,
  keys: string[],
): string | undefined {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "string" && value.length > 0) return value;
  }
  return undefined;
}

export function arrayOfStrings(value: unknown): string[] | undefined {
  if (!Array.isArray(value) || !value.every((item) => typeof item === "string")) return undefined;
  return value as string[];
}

export async function writeChunk(stream: Writable, chunk: string): Promise<void> {
  if (!stream.write(chunk)) await once(stream, "drain");
}

export function product(values: number[]): number {
  return values.reduce((acc, value) => acc * value, 1);
}

export function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${label} deve ser inteiro positivo; recebido ${value}.`);
  }
}

export function stableUnique<T>(values: Iterable<T>): T[] {
  return [...new Set(values)];
}

/** Exact IEEE-754 binary16 widening; storage decoding is not an arithmetic policy. */
export function decodeIeeeF16ToF32(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

const F32_BITS_SCRATCH = new DataView(new ArrayBuffer(4));

/** Exact IEEE-754 bfloat16 widening from its stored upper binary32 bits. */
export function decodeIeeeBF16ToF32(bits: number): number {
  F32_BITS_SCRATCH.setUint32(0, bits << 16, true);
  return F32_BITS_SCRATCH.getFloat32(0, true);
}

/** Round a binary32 result to BF16 (nearest, ties to even), then widen it back to binary32. */
export function roundF32ToBF16(value: number): number {
  if (!Number.isFinite(value)) return value;
  F32_BITS_SCRATCH.setFloat32(0, Math.fround(value), true);
  const bits = F32_BITS_SCRATCH.getUint32(0, true);
  const rounded = (bits + 0x7fff + ((bits >>> 16) & 1)) >>> 0;
  F32_BITS_SCRATCH.setUint32(0, rounded & 0xffff0000, true);
  return F32_BITS_SCRATCH.getFloat32(0, true);
}
