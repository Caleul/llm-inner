import { evaluateGemma4CalibrationPromotion, parseGemma4CalibrationPromotionOptions } from "./gemma4-calibration-promotion.js";

const report = await evaluateGemma4CalibrationPromotion(parseGemma4CalibrationPromotionOptions(process.argv.slice(2)));
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (!report.accepted) process.exitCode = 1;
