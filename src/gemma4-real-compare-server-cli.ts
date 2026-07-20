import { createGemma4RealComparisonServer, parseGemma4RealServerOptions } from "./gemma4-real-compare-server.js";

const options = parseGemma4RealServerOptions(process.argv.slice(2));
const server = createGemma4RealComparisonServer(options);
server.listen(options.port, options.host, () => process.stdout.write(`Gemma 4 real comparison: http://${options.host}:${options.port}\n`));
