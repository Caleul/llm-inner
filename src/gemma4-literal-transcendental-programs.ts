import { createHash } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import {
  SLEEF_REMPITABSP_F32_LE_BASE64,
  SLEEF_REMPITABSP_F32_LE_SHA256,
  sleefCosF32,
  sleefExpF32,
  sleefSinF32,
  sleefTanhF32,
} from "./sleef-f32.js";

export type Gemma4LiteralF32Transcendental =
  | "SLEEF_EXP_F32"
  | "SLEEF_SIN_F32"
  | "SLEEF_COS_F32"
  | "SLEEF_TANH_F32";

export interface Gemma4LiteralF32Constant {
  name: string;
  binary32Hex: string;
  value: number;
}

export interface Gemma4LiteralF32TranscendentalProgram {
  kernel: string;
  input: "input:F32";
  output: "result:F32";
  specialCases: string[];
  scalarAssignments: string[];
}

/**
 * Finite scalar transcription of every transcendental used by the registered
 * dense Gemma 4 calculation. This is artifact data rather than a request to
 * load PyTorch, SLEEF, libc or JavaScript Math during replay.
 */
export interface Gemma4LiteralTranscendentalPrograms {
  kind: "gemma4-literal-f32-transcendental-programs";
  schemaVersion: 1;
  language: "binary32-pair-register-program-v1";
  authority: {
    runtime: "pytorch-eager-cpu-darwin-arm64";
    pytorchSourceCommit: "7269437d655783a26cba32aa88195b741ff496aa";
    sleefSourceCommit: "5a1d179df9cf652951b59010a2d2075372d67f68";
  };
  evaluation: {
    statementOrder: string;
    f32: string;
    fma: string;
    pair: string;
    roundTiesToEven: string;
    bitOperations: string;
    specialValues: string;
  };
  constants: Gemma4LiteralF32Constant[];
  rempiTable: {
    storageDtype: "F32";
    byteOrder: "little-endian";
    entries: 416;
    payloadBase64: string;
    payloadSha256: string;
    indexProgram: string[];
  };
  sharedSubprograms: Array<{ name: string; scalarAssignments: string[] }>;
  programs: Record<Gemma4LiteralF32Transcendental, Gemma4LiteralF32TranscendentalProgram>;
}

const literal = (name: string, value: number): Gemma4LiteralF32Constant => {
  const rounded = Math.fround(value);
  const bytes = Buffer.allocUnsafe(4);
  bytes.writeFloatLE(rounded);
  return { name, binary32Hex: `0x${bytes.readUInt32LE().toString(16).padStart(8, "0")}`, value: rounded };
};

export function buildGemma4LiteralTranscendentalPrograms(): Gemma4LiteralTranscendentalPrograms {
  const constants = [
    literal("INV_PI", 0.31830988618379067154),
    literal("NEG_PI_HI", -3.1414794921875),
    literal("NEG_PI_LO", -0.00011315941810607910156),
    literal("NEG_PI_TINY", -1.9841872589410058936e-9),
    literal("NEG_HALF_PI_HI", -1.5707963705062866),
    literal("HALF_PI_TINY", 4.371138828673793e-8),
    literal("TWO_PI_HI", 6.2831854820251465),
    literal("TWO_PI_LO", -1.7484555314695172e-7),
    literal("INV_LN2", 1.4426950408889634),
    literal("NEG_LN2_HI", -0.693145751953125),
    literal("NEG_LN2_LO", -1.428606765330187e-6),
    literal("SIN_C0", -0.166666597127914428710938),
    literal("SIN_C1", 0.00833307858556509017944336),
    literal("SIN_C2", -0.0001981069071916863322258),
    literal("SIN_C3", 2.6083159809786593541503e-6),
    literal("EXP_C0", 0.5),
    literal("EXP_C1", 0.166666671633720397949219),
    literal("EXP_C2", 0.0416664853692054748535156),
    literal("EXP_C3", 0.00833336077630519866943359),
    literal("EXP_C4", 0.00139304355252534151077271),
    literal("EXP_C5", 0.000198527617612853646278381),
    literal("EXP_PAIR_C0", 0.5),
    literal("EXP_PAIR_C1", 0.16666665941423424),
    literal("EXP_PAIR_C2", 0.04166637361),
    literal("EXP_PAIR_C3", 0.008333456703),
    literal("EXP_PAIR_C4", 0.001394256484),
    literal("EXP_PAIR_C5", 0.0001980960224),
    literal("TANH_SATURATION", 8.664339742),
  ];
  return {
    kind: "gemma4-literal-f32-transcendental-programs",
    schemaVersion: 1,
    language: "binary32-pair-register-program-v1",
    authority: {
      runtime: "pytorch-eager-cpu-darwin-arm64",
      pytorchSourceCommit: "7269437d655783a26cba32aa88195b741ff496aa",
      sleefSourceCommit: "5a1d179df9cf652951b59010a2d2075372d67f68",
    },
    evaluation: {
      statementOrder: "evaluate scalarAssignments in array order; assignments replace the named register exactly once at that statement",
      f32: "F32(x) rounds x to IEEE-754 binary32, round-to-nearest ties-to-even; every +,-,*,/ helper named F32_* materializes this boundary",
      fma: "F32_FMA(a,b,c) computes exact binary32 a*b+c and rounds once to binary32; no separately rounded product is permitted",
      pair: "PAIR(x,y) is two ordered binary32 registers .x and .y; every PAIR_* subprogram is expanded from sharedSubprograms without host double-float behavior",
      roundTiesToEven: "ROUND_TIES_EVEN(x) selects floor(x), floor(x)+1, or the even integer at an exact half; result must remain an exact safe integer",
      bitOperations: "F32_BITS is the little-endian IEEE binary32 u32 bit pattern; shifts, masks, &, | and xor-sign operate on exact u32 values",
      specialValues: "NaN, positive/negative Infinity and signed zero branches execute before finite arithmetic exactly as listed by each program",
    },
    constants,
    rempiTable: {
      storageDtype: "F32",
      byteOrder: "little-endian",
      entries: 416,
      payloadBase64: SLEEF_REMPITABSP_F32_LE_BASE64,
      payloadSha256: SLEEF_REMPITABSP_F32_LE_SHA256,
      indexProgram: [
        "raw_exponent=((F32_BITS(input)>>23)&255)-127-25",
        "scale_exponent=raw_exponent>65 ? -64 : 0; scaled_input=F32(input*F32(2**scale_exponent))",
        "table_index=max(0,raw_exponent)*4; require table_index+3<416",
        "consume table[table_index+0..table_index+3] in ascending address order through REMPI_F32",
      ],
    },
    sharedSubprograms: sharedSubprograms(),
    programs: {
      SLEEF_EXP_F32: {
        kernel: "Sleef_expf4_u10advsimd",
        input: "input:F32",
        output: "result:F32",
        specialCases: ["isNaN(input) => NaN", "input < F32(-104) => F32(0)", "input > F32(100) => +Infinity"],
        scalarAssignments: [
          "exponent=ROUND_TIES_EVEN(F32_MUL(input,INV_LN2))",
          "reduced=F32_FMA(F32(exponent),NEG_LN2_HI,input)",
          "reduced=F32_FMA(F32(exponent),NEG_LN2_LO,reduced)",
          "polynomial=EXP_C5",
          "polynomial=F32_FMA(polynomial,reduced,EXP_C4)",
          "polynomial=F32_FMA(polynomial,reduced,EXP_C3)",
          "polynomial=F32_FMA(polynomial,reduced,EXP_C2)",
          "polynomial=F32_FMA(polynomial,reduced,EXP_C1)",
          "polynomial=F32_FMA(polynomial,reduced,EXP_C0)",
          "result=F32_SCALE_POW2(F32_ADD(F32(1),F32_FMA(F32_MUL(reduced,reduced),polynomial,reduced)),exponent)",
        ],
      },
      SLEEF_SIN_F32: {
        kernel: "Sleef_sinf4_u10advsimd",
        input: "input:F32",
        output: "result:F32",
        specialCases: ["!isFinite(input) => NaN", "input is -0 => -0"],
        scalarAssignments: [
          "if abs(input)>=F32(125): reduction=REMPI_F32(input); execute SIN_LARGE_F32(reduction)",
          "else quadrant_float=F32(ROUND_TIES_EVEN(F32_MUL(input,INV_PI))); quadrant=trunc(quadrant_float)",
          "reduced=PAIR_ADD_FLOAT_FLOAT2(F32_FMA(quadrant_float,NEG_PI_HI,input),F32_MUL(quadrant_float,NEG_PI_LO))",
          "reduced=PAIR_ADD_FLOAT(reduced,F32_MUL(quadrant_float,NEG_PI_TINY))",
          "result=SIN_REDUCED_F32(reduced); if quadrant&1 result=-result",
        ],
      },
      SLEEF_COS_F32: {
        kernel: "Sleef_cosf4_u10advsimd",
        input: "input:F32",
        output: "result:F32",
        specialCases: ["!isFinite(input) => NaN"],
        scalarAssignments: [
          "if abs(input)>=F32(125): reduction=REMPI_F32(input); execute COS_LARGE_F32(reduction)",
          "else rounded=F32(ROUND_TIES_EVEN(F32_FMA(input,INV_PI,F32(-0.5))))",
          "quadrant_float=F32_FMA(rounded,F32(2),F32(1)); quadrant=trunc(quadrant_float)",
          "reduced=PAIR_ADD_FLOAT_FLOAT2(input,F32_MUL(quadrant_float,F32(NEG_PI_HI*F32(0.5))))",
          "reduced=PAIR_ADD_FLOAT2(reduced,F32_MUL(quadrant_float,F32(NEG_PI_LO*F32(0.5))))",
          "reduced=PAIR_ADD_FLOAT2(reduced,F32_MUL(quadrant_float,F32(NEG_PI_TINY*F32(0.5))))",
          "result=SIN_REDUCED_F32(reduced); if (quadrant&2)==0 result=-result",
        ],
      },
      SLEEF_TANH_F32: {
        kernel: "Sleef_tanhf4_u10advsimd",
        input: "input:F32",
        output: "result:F32",
        specialCases: ["isNaN(input) => NaN", "abs(input)>TANH_SATURATION => signbit(input) ? F32(-1) : F32(1)", "input is -0 => -0"],
        scalarAssignments: [
          "magnitude=abs(input); exponential=EXP_PAIR(PAIR(magnitude,F32(0)))",
          "reciprocal=PAIR_RECIPROCAL(exponential)",
          "quotient=PAIR_DIVIDE(PAIR_SUBTRACT(exponential,reciprocal),PAIR_ADD(exponential,reciprocal))",
          "result=F32_ADD(quotient.x,quotient.y); if isNaN(result) result=F32(1)",
          "if signbit(input) result=-result",
        ],
      },
    },
  };
}

export function validateGemma4LiteralTranscendentalPrograms(programs: Gemma4LiteralTranscendentalPrograms): void {
  if (!isDeepStrictEqual(programs, buildGemma4LiteralTranscendentalPrograms())) {
    throw new Error("Programa literal Gemma 4 possui transcrição transcendental F32 ausente ou divergente.");
  }
  const bytes = Buffer.from(programs.rempiTable.payloadBase64, "base64");
  if (bytes.length !== programs.rempiTable.entries * 4 || createHash("sha256").update(bytes).digest("hex") !== programs.rempiTable.payloadSha256) {
    throw new Error("Programa literal Gemma 4 possui tabela rempi F32 inválida.");
  }
  const declaredCalls = new Set([
    ...programs.sharedSubprograms.map((program) => program.name),
    ...Object.keys(programs.programs),
    "F32", "F32_ADD", "F32_SUB", "F32_MUL", "F32_DIV", "F32_FMA", "F32_BITS",
    "PAIR", "ROUND_TIES_EVEN", "STRUCT",
  ]);
  const statements = [
    ...programs.rempiTable.indexProgram,
    ...programs.sharedSubprograms.flatMap((program) => program.scalarAssignments),
    ...Object.values(programs.programs).flatMap((program) => [...program.specialCases, ...program.scalarAssignments]),
  ];
  for (const statement of statements) for (const call of statement.matchAll(/\b([A-Z][A-Z0-9_]+)\s*\(/g)) {
    if (!declaredCalls.has(call[1]!)) throw new Error(`Programa transcendental Gemma 4 referencia subprograma não declarado: ${call[1]}.`);
  }
}

/** Every named SLEEF intrinsic in artifact formulas must resolve to one embedded program. */
export function validateGemma4LiteralTranscendentalCoverage(
  programs: Gemma4LiteralTranscendentalPrograms,
  formulas: readonly string[],
): void {
  validateGemma4LiteralTranscendentalPrograms(programs);
  for (const formula of formulas) for (const match of formula.matchAll(/\b(SLEEF_[A-Z0-9_]+)\b/g)) {
    if (!(match[1]! in programs.programs)) {
      throw new Error(`Fórmula Gemma 4 referencia transcendental sem programa incorporado: ${match[1]}.`);
    }
  }
}

/** Executes only a validated artifact contract; altered program data cannot silently fall back to host math. */
export function executeGemma4LiteralTranscendentalProgram(
  programs: Gemma4LiteralTranscendentalPrograms,
  intrinsic: Gemma4LiteralF32Transcendental,
  input: number,
): number {
  validateGemma4LiteralTranscendentalPrograms(programs);
  switch (intrinsic) {
    case "SLEEF_EXP_F32": return sleefExpF32(input);
    case "SLEEF_SIN_F32": return sleefSinF32(input);
    case "SLEEF_COS_F32": return sleefCosF32(input);
    case "SLEEF_TANH_F32": return sleefTanhF32(input);
  }
}

function sharedSubprograms(): Gemma4LiteralTranscendentalPrograms["sharedSubprograms"] {
  return [
    { name: "PAIR_ADD", scalarAssignments: ["sum=F32_ADD(left.x,right.x)", "result=PAIR(sum,F32_ADD(F32_ADD(F32_SUB(left.x,sum),right.x),F32_ADD(left.y,right.y)))"] },
    { name: "PAIR_ADD2", scalarAssignments: ["sum=F32_ADD(left.x,right.x); virtual=F32_SUB(sum,left.x)", "error=F32_ADD(F32_SUB(left.x,F32_SUB(sum,virtual)),F32_SUB(right.x,virtual))", "result=PAIR(sum,F32_ADD(error,F32_ADD(left.y,right.y)))"] },
    { name: "PAIR_SUBTRACT", scalarAssignments: ["difference=F32_SUB(left.x,right.x); error=F32_SUB(F32_SUB(left.x,difference),right.x); error=F32_ADD(error,left.y)", "result=PAIR(difference,F32_SUB(error,right.y))"] },
    { name: "PAIR_ADD_FLOAT", scalarAssignments: ["sum=F32_ADD(left.x,right)", "result=PAIR(sum,F32_ADD(F32_ADD(F32_SUB(left.x,sum),right),left.y))"] },
    { name: "PAIR_ADD_FLOAT2", scalarAssignments: ["sum=F32_ADD(left.x,right); virtual=F32_SUB(sum,left.x)", "error=F32_ADD(F32_SUB(left.x,F32_SUB(sum,virtual)),F32_SUB(right,virtual))", "result=PAIR(sum,F32_ADD(error,left.y))"] },
    { name: "PAIR_ADD_FLOAT_FLOAT2", scalarAssignments: ["sum=F32_ADD(left,right); virtual=F32_SUB(sum,left)", "result=PAIR(sum,F32_ADD(F32_SUB(left,F32_SUB(sum,virtual)),F32_SUB(right,virtual)))"] },
    { name: "PAIR_ADD_FLOAT_FLOAT", scalarAssignments: ["sum=F32_ADD(left,right)", "result=PAIR(sum,F32_ADD(F32_SUB(left,sum),right))"] },
    { name: "PAIR_ADD_FLOAT_PAIR", scalarAssignments: ["sum=F32_ADD(left,right.x)", "result=PAIR(sum,F32_ADD(F32_ADD(F32_SUB(left,sum),right.x),right.y))"] },
    { name: "PAIR_ADD_FLOAT_PAIR2", scalarAssignments: ["sum=F32_ADD(left,right.x); virtual=F32_SUB(sum,left)", "result=PAIR(sum,F32_ADD(F32_ADD(F32_SUB(left,F32_SUB(sum,virtual)),F32_SUB(right.x,virtual)),right.y))"] },
    { name: "PAIR_MULTIPLY_FLOAT", scalarAssignments: ["product=F32_MUL(left.x,right)", "result=PAIR(product,F32_FMA(left.y,right,F32_FMA(left.x,right,-product)))"] },
    { name: "PAIR_MULTIPLY_FLOAT_FLOAT", scalarAssignments: ["product=F32_MUL(left,right)", "result=PAIR(product,F32_FMA(left,right,-product))"] },
    { name: "PAIR_MULTIPLY", scalarAssignments: ["product=F32_MUL(left.x,right.x)", "result=PAIR(product,F32_FMA(left.x,right.y,F32_FMA(left.y,right.x,F32_FMA(left.x,right.x,-product))))"] },
    { name: "PAIR_SQUARE", scalarAssignments: ["square=F32_MUL(value.x,value.x)", "result=PAIR(square,F32_FMA(F32_ADD(value.x,value.x),value.y,F32_FMA(value.x,value.x,-square)))"] },
    { name: "PAIR_NORMALIZE", scalarAssignments: ["sum=F32_ADD(value.x,value.y)", "result=PAIR(sum,F32_ADD(F32_SUB(value.x,sum),value.y))"] },
    { name: "PAIR_RECIPROCAL", scalarAssignments: ["reciprocal=F32_DIV(F32(1),value.x)", "result=PAIR(reciprocal,F32_MUL(reciprocal,F32_FMA(-value.y,reciprocal,F32_FMA(-value.x,reciprocal,F32(1)))))"] },
    { name: "PAIR_DIVIDE", scalarAssignments: ["reciprocal=F32_DIV(F32(1),denominator.x); quotient=F32_MUL(numerator.x,reciprocal)", "numerator_error=F32_FMA(reciprocal,numerator.x,-quotient); denominator_error=F32_FMA(-denominator.y,reciprocal,F32_FMA(-denominator.x,reciprocal,F32(1)))", "result=PAIR(quotient,F32_FMA(quotient,denominator_error,F32_FMA(numerator.y,reciprocal,numerator_error)))"] },
    { name: "F32_SCALE_POW2", scalarAssignments: ["half=exponent>>1", "result=F32_MUL(F32_MUL(value,F32(2**half)),F32(2**(exponent-half)))"] },
    { name: "XOR_SIGN", scalarAssignments: ["result=signbit(sign_source) ? -value : value"] },
    { name: "SIN_REDUCED_F32", scalarAssignments: ["squared=PAIR_SQUARE(reduced); polynomial=SIN_C3", "polynomial=F32_FMA(polynomial,squared.x,SIN_C2); polynomial=F32_FMA(polynomial,squared.x,SIN_C1)", "coefficient=PAIR_ADD_FLOAT_FLOAT(SIN_C0,F32_MUL(polynomial,squared.x))", "correction=PAIR_MULTIPLY(coefficient,squared); expansion=PAIR_ADD_FLOAT_PAIR(F32(1),correction)", "result=F32_FMA(reduced.x,expansion.x,F32_FMA(reduced.y,expansion.x,F32_MUL(reduced.x,expansion.y)))"] },
    { name: "EXP_PAIR", scalarAssignments: ["exponent=ROUND_TIES_EVEN(F32_MUL(F32_ADD(input.x,input.y),INV_LN2))", "reduced=PAIR_ADD_FLOAT2(input,F32_MUL(F32(exponent),NEG_LN2_HI)); reduced=PAIR_ADD_FLOAT2(reduced,F32_MUL(F32(exponent),NEG_LN2_LO))", "polynomial=EXP_PAIR_C5; polynomial=F32_FMA(polynomial,reduced.x,EXP_PAIR_C4); polynomial=F32_FMA(polynomial,reduced.x,EXP_PAIR_C3); polynomial=F32_FMA(polynomial,reduced.x,EXP_PAIR_C2)", "correction=PAIR_ADD_FLOAT2(PAIR_MULTIPLY_FLOAT(reduced,polynomial),EXP_PAIR_C1); correction=PAIR_ADD_FLOAT2(PAIR_MULTIPLY(reduced,correction),EXP_PAIR_C0)", "correction=PAIR_ADD2(reduced,PAIR_MULTIPLY(PAIR_SQUARE(reduced),correction)); correction=PAIR_ADD_FLOAT_PAIR2(F32(1),correction)", "result=input.x<F32(-104) ? PAIR(F32(0),F32(0)) : PAIR(F32_SCALE_POW2(correction.x,exponent),F32_SCALE_POW2(correction.y,exponent))"] },
    { name: "REMPI_F32", scalarAssignments: ["execute rempiTable.indexProgram; reduced=PAIR_MULTIPLY_FLOAT_FLOAT(scaled_input,table[index]); sub=REMPI_SUB(reduced.x); quadrant=sub.quadrant; reduced=PAIR_NORMALIZE(PAIR(sub.remainder,reduced.y))", "reduced=PAIR_ADD2(reduced,PAIR_MULTIPLY_FLOAT_FLOAT(scaled_input,table[index+1])); sub=REMPI_SUB(reduced.x); quadrant+=sub.quadrant; reduced=PAIR_NORMALIZE(PAIR(sub.remainder,reduced.y))", "reduced=PAIR_ADD2(reduced,PAIR_MULTIPLY_FLOAT(PAIR(table[index+2],table[index+3]),scaled_input)); reduced=PAIR_NORMALIZE(reduced)", "reduced=PAIR_MULTIPLY(reduced,PAIR(TWO_PI_HI,TWO_PI_LO)); result=STRUCT(reduced,quadrant)"] },
    { name: "REMPI_SUB", scalarAssignments: ["rounded_four=F32(ROUND_TIES_EVEN(F32_MUL(input,F32(4)))); rounded_one=F32(ROUND_TIES_EVEN(input))", "result=STRUCT(remainder=F32_SUB(input,F32_MUL(rounded_four,F32(0.25))),quadrant=trunc(F32_SUB(rounded_four,F32_MUL(rounded_one,F32(4)))))"] },
    { name: "SIN_LARGE_F32", scalarAssignments: ["quadrant=reduction.quadrant&3; quadrant=((quadrant+quadrant)+(reduction.reduced.x>0?2:1))>>2", "if reduction.quadrant&1: reduction.reduced=PAIR_ADD2(reduction.reduced,PAIR(XOR_SIGN(NEG_HALF_PI_HI,reduction.reduced.x),XOR_SIGN(HALF_PI_TINY,reduction.reduced.x)))", "result=SIN_REDUCED_F32(PAIR_NORMALIZE(reduction.reduced)); if quadrant&1 result=XOR_SIGN(result,F32(-1))"] },
    { name: "COS_LARGE_F32", scalarAssignments: ["quadrant=reduction.quadrant&3; quadrant=((quadrant+quadrant)+(reduction.reduced.x>0?8:7))>>1", "if !(reduction.quadrant&1): sign=reduction.reduced.x>0?F32(0):F32(-1); reduction.reduced=PAIR_ADD2(reduction.reduced,PAIR(XOR_SIGN(NEG_HALF_PI_HI,sign),XOR_SIGN(HALF_PI_TINY,sign)))", "result=SIN_REDUCED_F32(PAIR_NORMALIZE(reduction.reduced)); if (quadrant&2)==0 result=XOR_SIGN(result,F32(-1))"] },
  ];
}
