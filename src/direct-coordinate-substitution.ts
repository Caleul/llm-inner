import { createReadStream, createWriteStream } from "node:fs";
import { stat } from "node:fs/promises";
import { once } from "node:events";
import { finished } from "node:stream/promises";

/** Count exact UTF-8 output bytes before materializing a repeated scalar producer. */
export async function estimateDirectCoordinateExpansion(
  inputPath: string,
  frontier: string,
  position: string,
  coordinatePaths: readonly string[],
): Promise<{ bytes: number; replacements: number }> {
  if (!/^[a-z][A-Za-z0-9]*$/.test(frontier) || !/^[a-z][a-z0-9]*$/.test(position)) {
    throw new Error("Invalid scalar frontier");
  }
  const sizes = await Promise.all(coordinatePaths.map(async (path) => (await stat(path)).size));
  const pattern = new RegExp(`\\b${frontier}\\[${position}\\]\\[(\\d+)\\]`, "g");
  let pending = "", bytes = 0, replacements = 0;
  const consume = (source: string, final: boolean): void => {
    const limit = final ? source.length : Math.max(0, source.length - 256);
    let cursor = 0, writableEnd = limit;
    for (const match of source.matchAll(pattern)) {
      if (match.index >= limit) break;
      if (!final && match.index + match[0].length > limit) { writableEnd = match.index; break; }
      const size = sizes[Number(match[1])];
      if (size === undefined) throw new RangeError(`Coordinate ${match[1]} outside the discovered width`);
      bytes += Buffer.byteLength(source.slice(cursor, match.index)) + size;
      if (!Number.isSafeInteger(bytes)) throw new RangeError("Direct expansion size exceeds safe integer range");
      replacements++;
      cursor = match.index + match[0].length;
    }
    bytes += Buffer.byteLength(source.slice(cursor, writableEnd));
    pending = source.slice(writableEnd);
  };
  for await (const chunk of createReadStream(inputPath, { encoding: "utf8", highWaterMark: 64 * 1024 })) {
    consume(pending + chunk, false);
  }
  consume(pending, true);
  return { bytes, replacements };
}

/** Copy complete scalar producers at every coordinate reference, with bounded memory. */
export async function substituteDirectCoordinates(
  inputPath: string,
  outputPath: string,
  frontier: string,
  position: string,
  coordinatePaths: readonly string[],
  maxBytes = Number.MAX_SAFE_INTEGER,
): Promise<{ bytes: number; replacements: number }> {
  if (inputPath === outputPath) throw new Error("Input and output paths must differ");
  if (!/^[a-z][A-Za-z0-9]*$/.test(frontier) || !/^[a-z][a-z0-9]*$/.test(position)) {
    throw new Error("Invalid scalar frontier");
  }
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 0) throw new RangeError("Invalid byte limit");
  const estimate = await estimateDirectCoordinateExpansion(inputPath, frontier, position, coordinatePaths);
  if (estimate.bytes > maxBytes) {
    throw new RangeError(`Direct expansion requires ${estimate.bytes} bytes, exceeds ${maxBytes}`);
  }
  const paths = await Promise.all(coordinatePaths.map(async (path) => ({ path, size: (await stat(path)).size })));
  const pattern = new RegExp(`\\b${frontier}\\[${position}\\]\\[(\\d+)\\]`, "g");
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  let pending = "", bytes = 0, replacements = 0;
  const write = async (source: string | Buffer): Promise<void> => {
    const size = Buffer.byteLength(source);
    if (size > maxBytes - bytes) throw new RangeError(`Direct expansion exceeds ${maxBytes} bytes`);
    bytes += size;
    if (!output.write(source)) await once(output, "drain");
  };
  const copy = async (coordinate: number): Promise<void> => {
    const entry = paths[coordinate];
    if (!entry) throw new RangeError(`Coordinate ${coordinate} outside the discovered width`);
    if (entry.size > maxBytes - bytes) throw new RangeError(`Direct expansion exceeds ${maxBytes} bytes`);
    for await (const chunk of createReadStream(entry.path)) await write(chunk as Buffer);
  };
  const consume = async (source: string, final: boolean): Promise<void> => {
    const limit = final ? source.length : Math.max(0, source.length - 256);
    let cursor = 0, writableEnd = limit;
    for (const match of source.matchAll(pattern)) {
      if (match.index >= limit) break;
      if (!final && match.index + match[0].length > limit) { writableEnd = match.index; break; }
      await write(source.slice(cursor, match.index));
      await copy(Number(match[1]));
      replacements++;
      cursor = match.index + match[0].length;
    }
    await write(source.slice(cursor, writableEnd));
    pending = source.slice(writableEnd);
  };
  try {
    for await (const chunk of createReadStream(inputPath, { encoding: "utf8", highWaterMark: 64 * 1024 })) {
      await consume(pending + chunk, false);
    }
    await consume(pending, true);
    output.end();
    await finished(output);
    if (bytes !== estimate.bytes || replacements !== estimate.replacements) {
      throw new Error("Streaming substitution differs from its size audit");
    }
    return { bytes, replacements };
  } catch (error) {
    output.on("error", () => {});
    output.destroy();
    throw error;
  }
}
