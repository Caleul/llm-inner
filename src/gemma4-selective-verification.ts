export interface Gemma4TrustedVerificationStep extends Record<string, unknown> {
  step: number;
  tokenId: number;
  contextLength: number;
  topLogits: [];
  verificationSkipped: true;
}

export interface Gemma4SelectiveVerificationTail {
  generatedTokenIds: number[];
  steps: Gemma4TrustedVerificationStep[];
  terminalLogitsSha256: string;
  earlyExitStep: number;
  decoderStepsAvoided: number;
}

/**
 * Projects the already authenticated fast-path suffix after the last
 * sensitive decision matched the exact verifier.  A changed context or a
 * diagnostic capture must continue through the exact decoder instead.
 */
export function planGemma4SelectiveVerificationTail(options: {
  fastGeneratedTokenIds: readonly number[];
  fastTerminalLogitsSha256: string;
  inputLength: number;
  currentStep: number;
  lastSensitiveStep: number;
  divergenceStep: number | null;
  captureEnabled: boolean;
}): Gemma4SelectiveVerificationTail | null {
  const { fastGeneratedTokenIds, fastTerminalLogitsSha256, inputLength, currentStep, lastSensitiveStep, divergenceStep, captureEnabled } = options;
  if (!Number.isSafeInteger(inputLength) || inputLength < 1 || !Number.isSafeInteger(currentStep) || !Number.isSafeInteger(lastSensitiveStep)) throw new Error("Estado da verificação seletiva é inválido.");
  if (!/^[0-9a-f]{64}$/.test(fastTerminalLogitsSha256)) throw new Error("Hash terminal do caminho rápido é inválido.");
  if (fastGeneratedTokenIds.length < 1 || fastGeneratedTokenIds.some((token) => !Number.isSafeInteger(token) || token < 0)) throw new Error("Tokens do caminho rápido são inválidos.");
  if (captureEnabled || divergenceStep !== null || currentStep !== lastSensitiveStep || currentStep + 1 >= fastGeneratedTokenIds.length) return null;
  const generatedTokenIds = fastGeneratedTokenIds.slice(currentStep + 1);
  return {
    generatedTokenIds: [...generatedTokenIds],
    steps: generatedTokenIds.map((tokenId, offset) => {
      const step = currentStep + 1 + offset;
      return { step, tokenId, contextLength: inputLength + step, topLogits: [], verificationSkipped: true };
    }),
    terminalLogitsSha256: fastTerminalLogitsSha256,
    earlyExitStep: currentStep,
    decoderStepsAvoided: generatedTokenIds.length,
  };
}
