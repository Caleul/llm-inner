import assert from "node:assert/strict";
import test from "node:test";
import { runNativeFixtureBenchmarks } from "../src/benchmark.js";

test("native benchmark reports independently timed supported container paths without performance thresholds", async () => {
  const report = await runNativeFixtureBenchmarks({ samples: 1, warmupSamples: 0 });
  assert.equal(report.schemaVersion, 1);
  assert.equal(report.benchmark, "native-container-and-reference-executor");
  assert.deepEqual(report.workloads.map((workload) => workload.id), ["safetensors-f32", "mlx-affine-u32", "gguf-q8_0"]);
  assert.deepEqual(report.workloads.map((workload) => workload.containerFormat), ["safetensors", "mlx-safetensors", "gguf"]);
  for (const workload of report.workloads) {
    assert.equal(workload.fixtureFiles.length, workload.id === "gguf-q8_0" ? 1 : 2);
    assert.ok(workload.fixtureFiles.every((file) => /^[a-f0-9]{64}$/.test(file.sha256)));
    assert.ok(workload.memory.materializedConstantBytes > 0);
    assert.ok(workload.memory.processAfterMaterializationBytes.rss > 0);
    assert.ok(workload.memory.processAfterMaterializationBytes.heapUsed > 0);
    for (const stage of Object.values(workload.stages)) {
      assert.equal(stage.samplesNanoseconds.length, 1);
      assert.ok(stage.minimumNanoseconds >= 0);
      assert.equal(stage.minimumNanoseconds, stage.medianNanoseconds);
      assert.equal(stage.medianNanoseconds, stage.p95Nanoseconds);
    }
  }
});

test("native benchmark rejects invalid sampling instead of silently changing workload evidence", async () => {
  await assert.rejects(() => runNativeFixtureBenchmarks({ samples: 0 }), /samples deve ser inteiro positivo/);
  await assert.rejects(() => runNativeFixtureBenchmarks({ warmupSamples: -1 }), /warmupSamples deve ser inteiro não negativo/);
});
