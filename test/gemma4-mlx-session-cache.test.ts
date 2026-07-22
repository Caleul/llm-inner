import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { resolve } from "node:path";

test("cache Metal recua ao maior prefixo token-idêntico e recalcula o token terminal", () => {
  const python = resolve("venv/bin/python"), worker = resolve("scripts/gemma4-mlx-linear-worker.py");
  const program = `
import importlib.util, json, numpy as np
spec = importlib.util.spec_from_file_location("gemma4_mlx_worker", ${JSON.stringify(worker)})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cache = {0: (np.zeros((1, 2, 6, 3), dtype=np.float32), np.ones((1, 2, 6, 3), dtype=np.float32))}
session = {"id": 7, "token_ids": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32), "caches": cache}
rewound, rewound_count = module.resolve_resident_generation_session(session, 7, np.array([[1, 2, 3, 9, 10]], dtype=np.int32))
exact, exact_count = module.resolve_resident_generation_session(session, 7, np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32))
miss, miss_count = module.resolve_resident_generation_session(session, 8, np.array([[1, 2, 3]], dtype=np.int32))
print(json.dumps({"rewoundCount": rewound_count, "rewoundTokens": rewound["token_ids"].tolist(), "rewoundKeyTokens": rewound["caches"][0][0].shape[2], "exactCount": exact_count, "exactKeyTokens": exact["caches"][0][0].shape[2], "miss": miss is None, "missCount": miss_count}))
`;
  const result = spawnSync(python, ["-c", program], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), { rewoundCount: 3, rewoundTokens: [1, 2, 3], rewoundKeyTokens: 3, exactCount: 5, exactKeyTokens: 5, miss: true, missCount: 0 });
});
