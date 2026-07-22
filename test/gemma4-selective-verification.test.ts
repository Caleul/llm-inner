import assert from "node:assert/strict";
import test from "node:test";
import { planGemma4SelectiveVerificationTail } from "../src/gemma4-selective-verification.js";

const hash = "a".repeat(64);

test("verificação seletiva encerra depois do último empate comprovado", () => {
  const tail = planGemma4SelectiveVerificationTail({
    fastGeneratedTokenIds: [10, 11, 12, 13],
    fastTerminalLogitsSha256: hash,
    inputLength: 5,
    currentStep: 1,
    lastSensitiveStep: 1,
    divergenceStep: null,
    captureEnabled: false,
  });
  assert.deepEqual(tail, {
    generatedTokenIds: [12, 13],
    steps: [
      { step: 2, tokenId: 12, contextLength: 7, topLogits: [], verificationSkipped: true },
      { step: 3, tokenId: 13, contextLength: 8, topLogits: [], verificationSkipped: true },
    ],
    terminalLogitsSha256: hash,
    earlyExitStep: 1,
    decoderStepsAvoided: 2,
  });
});

test("verificação seletiva não encerra com divergência, captura ou empate posterior", () => {
  const base = { fastGeneratedTokenIds: [10, 11, 12], fastTerminalLogitsSha256: hash, inputLength: 2, currentStep: 0, lastSensitiveStep: 0, divergenceStep: null, captureEnabled: false };
  assert.equal(planGemma4SelectiveVerificationTail({ ...base, divergenceStep: 0 }), null);
  assert.equal(planGemma4SelectiveVerificationTail({ ...base, captureEnabled: true }), null);
  assert.equal(planGemma4SelectiveVerificationTail({ ...base, lastSensitiveStep: 1 }), null);
  assert.equal(planGemma4SelectiveVerificationTail({ ...base, currentStep: 2, lastSensitiveStep: 2 }), null);
});
