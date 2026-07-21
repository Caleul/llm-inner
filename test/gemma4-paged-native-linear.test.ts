import assert from "node:assert/strict";
import test from "node:test";
import { pagedLinearBatchF32, pagedLinearF32, type PagedDenseF32Matrix, type PagedLinearTileKernel } from "../src/paged-dense.js";

const matrix: PagedDenseF32Matrix = {
  tensor: { name: "weight", storageDtype: "F32", storageShape: [2, 2], logicalShape: [2, 2] },
  shape: [2, 2], maxReadBytes: 16,
  async readRows(startRow, rowCount) {
    assert.equal(startRow, 0); assert.equal(rowCount, 2);
    return { shape: [2, 2], values: Float32Array.from([3, 4, 5, 6]) };
  },
};

test("linear paginado delega um tile completo ao kernel nativo opt-in", async () => {
  let calls = 0;
  const kernel: PagedLinearTileKernel = {
    backend: "test-native",
    async multiply(input, weight, rows, outputs, features) {
      calls += 1; assert.deepEqual([rows, outputs, features], [1, 2, 2]);
      assert.deepEqual([...input], [1, 2]); assert.deepEqual([...weight], [3, 4, 5, 6]);
      return Float32Array.from([11, 17]);
    },
  };
  const result = await pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, matrix, { tileKernel: kernel });
  assert.equal(calls, 1); assert.deepEqual(result.shape, [1, 2]); assert.deepEqual([...result.values], [11, 17]);
});

test("linear paginado rejeita resposta nativa truncada", async () => {
  const kernel: PagedLinearTileKernel = { backend: "broken", async multiply() { return Float32Array.from([1]); } };
  await assert.rejects(() => pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, matrix, { tileKernel: kernel }), /tile inválido/);
});

test("kernel nativo recebe bytes BF16 sem expansão JavaScript", async () => {
  const raw = Buffer.from([0x40, 0x40, 0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40]);
  const storageMatrix: PagedDenseF32Matrix = { ...matrix, tensor: { name: "weight", storageDtype: "BF16", storageShape: [2, 2], logicalShape: [2, 2] }, async readStorageRows() { return raw; } };
  let storageCalls = 0;
  const kernel: PagedLinearTileKernel = {
    backend: "raw-bf16", async multiply() { throw new Error("não deve expandir"); },
    async multiplyStorage(input, weight, dtype, rows, outputs, features) {
      storageCalls += 1; assert.equal(weight, raw); assert.equal(dtype, "BF16"); assert.deepEqual([rows, outputs, features], [1, 2, 2]); assert.deepEqual([...input], [1, 2]);
      return Float32Array.from([11, 17]);
    },
  };
  const result = await pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, storageMatrix, { tileKernel: kernel });
  assert.equal(storageCalls, 1); assert.deepEqual([...result.values], [11, 17]);
});

test("kernel nativo referenciado evita transportar bytes da matriz pelo JavaScript", async () => {
  const referencedMatrix: PagedDenseF32Matrix = {
    ...matrix,
    tensor: { name: "weight", storageDtype: "BF16", storageShape: [2, 2], logicalShape: [2, 2], shard: "model.safetensors", byteOffset: 128, byteLength: 8 },
    async readRows() { throw new Error("não deve ler o pool em JavaScript"); },
  };
  let referenceCalls = 0;
  const kernel: PagedLinearTileKernel = {
    backend: "mmap-reference", async multiply() { throw new Error("não deve transportar matriz"); },
    async multiplyStorageReference(input, tensor, startOutput, outputCount, rows) {
      referenceCalls += 1; assert.equal(tensor, referencedMatrix.tensor); assert.deepEqual([startOutput, outputCount, rows], [0, 2, 1]); assert.deepEqual([...input], [1, 2]);
      return Float32Array.from([11, 17]);
    },
  };
  const result = await pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, referencedMatrix, { tileKernel: kernel });
  assert.equal(referenceCalls, 1); assert.deepEqual([...result.values], [11, 17]);
});

test("duas projeções com a mesma entrada compartilham um único despacho referenciado", async () => {
  const gate: PagedDenseF32Matrix = { ...matrix, tensor: { ...matrix.tensor, name: "gate", shard: "model.safetensors", byteOffset: 64, byteLength: 16 } };
  const up: PagedDenseF32Matrix = { ...matrix, tensor: { ...matrix.tensor, name: "up", shard: "model.safetensors", byteOffset: 80, byteLength: 16 } };
  let batchCalls = 0;
  const kernel: PagedLinearTileKernel = {
    backend: "batch-reference", async multiply() { throw new Error("não deve executar projeção isolada"); },
    async multiplyStorageReferences(input, requests, rows) {
      batchCalls += 1; assert.deepEqual([...input], [1, 2]); assert.equal(rows, 1);
      assert.deepEqual(requests.map(({ tensor, startOutput, outputCount }) => [tensor.name, startOutput, outputCount]), [["gate", 0, 2], ["up", 0, 2]]);
      return [Float32Array.from([11, 17]), Float32Array.from([23, 29])];
    },
  };
  const result = await pagedLinearBatchF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, [gate, up], [{ tileKernel: kernel }, { tileKernel: kernel }]);
  assert.equal(batchCalls, 1); assert.deepEqual(result.map((entry) => [...entry.values]), [[11, 17], [23, 29]]);
});

test("linear em lote rejeita resposta truncada antes de publicar saída parcial", async () => {
  const kernel: PagedLinearTileKernel = { backend: "broken-batch", async multiply() { throw new Error("não usado"); }, async multiplyStorageReferences() { return [Float32Array.from([1, 2])]; } };
  await assert.rejects(() => pagedLinearBatchF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, [matrix, matrix], [{ tileKernel: kernel }, { tileKernel: kernel }]), /retornou 1 tiles; esperados 2/);
});

test("projeção terminal opta explicitamente pelo GEMM BF16 nativo", async () => {
  const referencedMatrix: PagedDenseF32Matrix = { ...matrix, tensor: { name: "head", storageDtype: "BF16", storageShape: [2, 2], logicalShape: [2, 2], shard: "model.safetensors", byteOffset: 128, byteLength: 8 } };
  let nativeCalls = 0;
  const kernel: PagedLinearTileKernel = {
    backend: "native-bf16", async multiply() { throw new Error("não deve ampliar para F32"); },
    async multiplyStorageReferenceNativeBf16(input, tensor, startOutput, outputCount, rows) {
      nativeCalls += 1; assert.equal(tensor.name, "head"); assert.deepEqual([startOutput, outputCount, rows], [0, 2, 1]); assert.deepEqual([...input], [1, 2]);
      return Float32Array.from([11, 17]);
    },
  };
  const result = await pagedLinearF32({ shape: [1, 2], values: Float32Array.from([1, 2]) }, referencedMatrix, { tileKernel: kernel, nativeBf16: true, outputDtype: "BF16" });
  assert.equal(nativeCalls, 1); assert.deepEqual([...result.values], [11, 17]);
});
