import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import * as path from "node:path";
import type { SourceFormat } from "./types.js";

const WIDTH = 256;
const GROUP_SIZE = 32;

export interface BenchmarkFixture {
  id: "safetensors-f32" | "mlx-affine-u32" | "gguf-q8_0";
  source: string;
  format: SourceFormat;
  quantization: string;
  files: Array<{ path: string; sha256: string }>;
}

/**
 * Creates three intentionally small but non-trivial native-container
 * workloads. They are performance fixtures, not authoritative-model traces:
 * every package is deterministic, self-contained, and removed by the caller.
 */
export async function writeNativeBenchmarkFixtures(root: string): Promise<BenchmarkFixture[]> {
  const safetensorsSource = path.join(root, "safetensors-f32");
  const mlxSource = path.join(root, "mlx-affine-u32");
  const ggufSource = path.join(root, "gguf-q8_0.gguf");
  const config = llamaConfig();
  const dense = denseTensors();

  await mkdir(safetensorsSource);
  await mkdir(mlxSource);
  await writeFile(path.join(safetensorsSource, "config.json"), JSON.stringify(config));
  await writeSafetensors(path.join(safetensorsSource, "model.safetensors"), dense);

  await writeFile(path.join(mlxSource, "config.json"), JSON.stringify({
    ...config,
    quantization: { bits: 4, group_size: GROUP_SIZE, mode: "affine" },
  }));
  await writeSafetensors(path.join(mlxSource, "model.safetensors"), mlxAffineTensors());

  await writeQ8Gguf(ggufSource);

  return [
    await fixture("safetensors-f32", safetensorsSource, "safetensors", "dense F32", ["config.json", "model.safetensors"]),
    await fixture("mlx-affine-u32", mlxSource, "mlx-safetensors", "MLX affine U32 4-bit group_size=32 with F32 scales and biases", ["config.json", "model.safetensors"]),
    await fixture("gguf-q8_0", ggufSource, "gguf", "GGML_TYPE_Q8_0", [path.basename(ggufSource)]),
  ];
}

async function fixture(
  id: BenchmarkFixture["id"],
  source: string,
  format: SourceFormat,
  quantization: string,
  files: string[],
): Promise<BenchmarkFixture> {
  const base = format === "gguf" ? path.dirname(source) : source;
  return {
    id,
    source,
    format,
    quantization,
    files: await Promise.all(files.map(async (file) => ({
      path: file,
      sha256: createHash("sha256").update(await readFile(path.join(base, file))).digest("hex"),
    }))),
  };
}

type FixtureDtype = "F32" | "U32";
type FixtureTensor = [name: string, dtype: FixtureDtype, shape: number[], values: number[]];

function llamaConfig() {
  return {
    model_type: "llama",
    hidden_size: WIDTH,
    intermediate_size: WIDTH,
    num_hidden_layers: 1,
    num_attention_heads: 4,
    num_key_value_heads: 4,
    head_dim: WIDTH / 4,
    vocab_size: WIDTH,
    rms_norm_eps: 1e-6,
    hidden_act: "silu",
  };
}

function denseTensors(): FixtureTensor[] {
  const matrices = matrixNames().map((name, tensorIndex): FixtureTensor => [name, "F32", [WIDTH, WIDTH], denseMatrix(tensorIndex)]);
  return [
    ...matrices,
    ...normNames().map((name): FixtureTensor => [name, "F32", [WIDTH], Array.from({ length: WIDTH }, (_, index) => Math.fround(0.75 + (index % 5) * 0.0625))]),
  ];
}

function mlxAffineTensors(): FixtureTensor[] {
  const tensors: FixtureTensor[] = [];
  for (const [tensorIndex, name] of matrixNames().entries()) {
    const scales = Array.from({ length: WIDTH * (WIDTH / GROUP_SIZE) }, (_, index) => Math.fround(0.03125 * (1 + ((index + tensorIndex) % 4))));
    const biases = Array.from({ length: WIDTH * (WIDTH / GROUP_SIZE) }, (_, index) => Math.fround(-0.25 + 0.0625 * ((index + tensorIndex) % 5)));
    const codes = Array.from({ length: WIDTH * WIDTH }, (_, index) => (index * 7 + tensorIndex * 3 + Math.floor(index / WIDTH)) & 0x0f);
    const module = name.slice(0, -".weight".length);
    tensors.push(
      [name, "U32", [WIDTH, WIDTH / 8], packMlxCodes(codes)],
      [`${module}.scales`, "F32", [WIDTH, WIDTH / GROUP_SIZE], scales],
      [`${module}.biases`, "F32", [WIDTH, WIDTH / GROUP_SIZE], biases],
    );
  }
  return [
    ...tensors,
    ...normNames().map((name): FixtureTensor => [name, "F32", [WIDTH], Array.from({ length: WIDTH }, (_, index) => Math.fround(0.75 + (index % 5) * 0.0625))]),
  ];
}

function matrixNames(): string[] {
  return [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "lm_head.weight",
  ];
}

function normNames(): string[] {
  return [
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
  ];
}

function denseMatrix(tensorIndex: number): number[] {
  return Array.from({ length: WIDTH * WIDTH }, (_, index) => Math.fround((((index * 13 + tensorIndex * 17) % 29) - 14) / 64));
}

function packMlxCodes(codes: readonly number[]): number[] {
  const packed = new Array<number>(WIDTH * (WIDTH / 8)).fill(0);
  for (let row = 0; row < WIDTH; row += 1) {
    for (let column = 0; column < WIDTH; column += 1) {
      const word = row * (WIDTH / 8) + Math.floor(column / 8);
      packed[word] = (packed[word]! | (codes[row * WIDTH + column]! << ((column % 8) * 4))) >>> 0;
    }
  }
  return packed;
}

async function writeSafetensors(file: string, tensors: FixtureTensor[]): Promise<void> {
  const header: Record<string, unknown> = {};
  let offset = 0;
  const payloads = tensors.map(([name, dtype, shape, values]) => {
    const payload = Buffer.alloc(values.length * 4);
    values.forEach((value, index) => dtype === "F32" ? payload.writeFloatLE(value, index * 4) : payload.writeUInt32LE(value, index * 4));
    header[name] = { dtype, shape, data_offsets: [offset, offset + payload.length] };
    offset += payload.length;
    return payload;
  });
  const encoded = Buffer.from(JSON.stringify(header));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(encoded.length));
  await writeFile(file, Buffer.concat([prefix, encoded, ...payloads]));
}

async function writeQ8Gguf(file: string): Promise<void> {
  const text = (value: string) => { const bytes = Buffer.from(value); const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(bytes.length)); return Buffer.concat([length, bytes]); };
  const u32 = (value: number) => { const bytes = Buffer.alloc(4); bytes.writeUInt32LE(value); return bytes; };
  const u64 = (value: number) => { const bytes = Buffer.alloc(8); bytes.writeBigUInt64LE(BigInt(value)); return bytes; };
  const metadata = (key: string, type: number, value: Buffer) => Buffer.concat([text(key), u32(type), value]);
  const metadataU32 = (key: string, value: number) => metadata(key, 4, u32(value));
  const metadataF32 = (key: string, value: number) => { const bytes = Buffer.alloc(4); bytes.writeFloatLE(value); return metadata(key, 6, bytes); };
  const metadataEntries = [
    metadata("general.architecture", 8, text("llama")), metadataU32("general.alignment", 32), metadataU32("llama.embedding_length", WIDTH),
    metadataU32("llama.block_count", 1), metadataU32("llama.attention.head_count", 4), metadataU32("llama.attention.head_count_kv", 4),
    metadataU32("llama.attention.key_length", WIDTH / 4), metadataU32("llama.feed_forward_length", WIDTH), metadataF32("llama.attention.layer_norm_rms_epsilon", 1e-6),
  ];
  const tensors: Array<[string, number[], number, Buffer]> = [
    ["token_embd.weight", [WIDTH, WIDTH], 8, q8Payload()],
    ["blk.0.attn_norm.weight", [WIDTH], 0, f32Payload(normValues())],
    ...["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"].map((name): [string, number[], number, Buffer] => [`blk.0.${name}.weight`, [WIDTH, WIDTH], 8, q8Payload()]),
    ["blk.0.ffn_norm.weight", [WIDTH], 0, f32Payload(normValues())],
    ["output_norm.weight", [WIDTH], 0, f32Payload(normValues())],
    ["output.weight", [WIDTH, WIDTH], 8, q8Payload()],
  ];
  const payloads: Buffer[] = [];
  const directory: Buffer[] = [];
  let offset = 0;
  for (const [name, shape, type, payload] of tensors) {
    const padding = (32 - (offset % 32)) % 32;
    if (padding) { payloads.push(Buffer.alloc(padding)); offset += padding; }
    directory.push(Buffer.concat([text(name), u32(shape.length), ...shape.map(u64), u32(type), u64(offset)]));
    payloads.push(payload);
    offset += payload.length;
  }
  const prefix = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(tensors.length), u64(metadataEntries.length), ...metadataEntries, ...directory]);
  await writeFile(file, Buffer.concat([prefix, Buffer.alloc((32 - (prefix.length % 32)) % 32), ...payloads]));
}

function normValues(): number[] {
  return Array.from({ length: WIDTH }, (_, index) => Math.fround(0.75 + (index % 5) * 0.0625));
}

function f32Payload(values: readonly number[]): Buffer {
  const payload = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
  return payload;
}

function q8Payload(): Buffer {
  const blocks = WIDTH * WIDTH / 32;
  const payload = Buffer.alloc(blocks * 34);
  for (let block = 0; block < blocks; block += 1) {
    payload.writeUInt16LE(0x3800, block * 34); // F16 0.5
    for (let index = 0; index < 32; index += 1) payload.writeInt8(((block * 32 + index) % 255) - 127, block * 34 + 2 + index);
  }
  return payload;
}
