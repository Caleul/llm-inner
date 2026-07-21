import { parseGemma4CalibrationCliOptions, runGemma4ThreeWayCalibration } from "./gemma4-real-calibration.js";

const options = await parseGemma4CalibrationCliOptions(process.argv.slice(2));
const report = await runGemma4ThreeWayCalibration(options);
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
