import type { Gemma4LiteralScalarCalculation } from "./gemma4-literal-scalar-calculations.js";
import type { Gemma4LiteralScalarExpression, Gemma4LiteralScalarStatementProgram } from "./gemma4-literal-scalar-statement-programs.js";
import type { Gemma4RealExpressionBuilder } from "./gemma4-real-expression.js";
import {
  evaluateGemma4RealIntegerExpression,
  lowerGemma4ScalarExpressionToExactReal,
  type Gemma4RealStatementLoweringContext,
} from "./gemma4-real-statement-lowering.js";

export interface Gemma4RealScalarProgramCompilationContext {
  builder: Gemma4RealExpressionBuilder;
  integerBindings?: Readonly<Record<string, number>>;
  resolveTensorElement: (name: string, coordinates: readonly number[]) => string;
  resolveTensorAxis?: (name: string, axis: number) => string;
  resolveLearnedElement: (role: string, coordinates: readonly number[], reductionBindings: Readonly<Record<string, number>>) => string;
  resolveInteger?: Gemma4RealStatementLoweringContext["resolveInteger"];
}

export interface Gemma4RealScalarProgramCompilation {
  root: string;
  localValues: ReadonlyMap<string, string>;
  preconditions: string[];
}

/** Executes the finite statement control structure while lowering arithmetic to exact-real nodes. */
export function compileGemma4ScalarCalculationToExactReal(
  calculation: Gemma4LiteralScalarCalculation,
  outputCoordinate: readonly number[],
  context: Gemma4RealScalarProgramCompilationContext,
): Gemma4RealScalarProgramCompilation {
  if (outputCoordinate.length !== calculation.outputCoordinates.length) {
    throw new Error(`${calculation.output}: coordenada real possui rank ${outputCoordinate.length}; esperado ${calculation.outputCoordinates.length}.`);
  }
  const baseBindings = {
    ...(context.integerBindings ?? {}),
    ...Object.fromEntries(calculation.outputCoordinates.map((name, index) => [name, outputCoordinate[index]!])),
  };
  const locals = new Map<string, string>();
  const scalarLocals: Record<string, string> = {};
  const preconditions: string[] = [];
  let root: string | undefined;
  const compileExpression = (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>): string =>
    lowerGemma4ScalarExpressionToExactReal(expression, {
      builder: context.builder,
      integerBindings: bindings,
      valueBindings: scalarLocals,
      resolveTensorElement: (name, coordinates) => locals.get(localKey(name, coordinates)) ?? context.resolveTensorElement(name, coordinates),
      ...(context.resolveTensorAxis ? { resolveTensorAxis: context.resolveTensorAxis } : {}),
      resolveLearnedElement: (role, coordinates) => context.resolveLearnedElement(role, coordinates, bindings),
      ...(context.resolveInteger ? { resolveInteger: context.resolveInteger } : {}),
    });
  for (const statement of calculation.statementPrograms) {
    if (statement.kind === "require") {
      // Preconditions constrain the legal input domain but do not contribute
      // an arithmetic value to the output root. Keep their artifact rendering
      // as metadata; compiling both sides would eagerly pull inactive modal
      // branches into a scalar closure.
      preconditions.push(statement.source);
      continue;
    }
    if (statement.expression.kind === "ordered-loop") {
      compileOrderedLoop(statement, statement.expression, baseBindings, compileExpression, locals, scalarLocals, context);
      continue;
    }
    const value = compileExpression(statement.expression, baseBindings);
    assignTargets(statement, value, baseBindings, locals, scalarLocals, context);
    if (statement.targets.some((target) => target.role === "output")) root = value;
  }
  if (!root) throw new Error(`${calculation.output}: programa real não produziu raiz de output.`);
  return { root, localValues: locals, preconditions };
}

function compileOrderedLoop(
  statement: Gemma4LiteralScalarStatementProgram,
  loop: Extract<Gemma4LiteralScalarExpression, { kind: "ordered-loop" }>,
  baseBindings: Readonly<Record<string, number>>,
  compileExpression: (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, number>>) => string,
  locals: Map<string, string>,
  scalarLocals: Record<string, string>,
  context: Gemma4RealScalarProgramCompilationContext,
): void {
  if (loop.order !== "ascending" || loop.domain.kind !== "range-inclusive") throw new Error("Loop real Gemma 4 requer range ascendente finito.");
  const integerContext: Gemma4RealStatementLoweringContext = {
    builder: context.builder,
    integerBindings: baseBindings,
    valueBindings: scalarLocals,
    resolveTensorElement: context.resolveTensorElement,
    ...(context.resolveTensorAxis ? { resolveTensorAxis: context.resolveTensorAxis } : {}),
    resolveLearnedElement: (role, coordinates) => context.resolveLearnedElement(role, coordinates, baseBindings),
    ...(context.resolveInteger ? { resolveInteger: context.resolveInteger } : {}),
  };
  const start = evaluateGemma4RealIntegerExpression(loop.domain.start, baseBindings, integerContext);
  const end = evaluateGemma4RealIntegerExpression(loop.domain.end, baseBindings, integerContext);
  if (end < start) throw new Error(`${loop.index}: loop real Gemma 4 possui range invertido.`);
  for (let index = start; index <= end; index += 1) {
    const bindings = { ...baseBindings, [loop.index]: index };
    const value = compileExpression(loop.body, bindings);
    assignTargets(statement, value, bindings, locals, scalarLocals, context);
  }
}

function assignTargets(
  statement: Gemma4LiteralScalarStatementProgram,
  value: string,
  bindings: Readonly<Record<string, number>>,
  locals: Map<string, string>,
  scalarLocals: Record<string, string>,
  context: Gemma4RealScalarProgramCompilationContext,
): void {
  if (statement.targets.length !== 1) throw new Error(`${statement.source}: múltiplos targets estruturados ainda requerem lowering dedicado.`);
  const target = statement.targets[0]!;
  const integerContext: Gemma4RealStatementLoweringContext = {
    builder: context.builder,
    integerBindings: bindings,
    valueBindings: scalarLocals,
    resolveTensorElement: context.resolveTensorElement,
    ...(context.resolveTensorAxis ? { resolveTensorAxis: context.resolveTensorAxis } : {}),
    resolveLearnedElement: (role, coordinates) => context.resolveLearnedElement(role, coordinates, bindings),
    ...(context.resolveInteger ? { resolveInteger: context.resolveInteger } : {}),
  };
  const coordinates = target.coordinates.map((coordinate) => evaluateGemma4RealIntegerExpression(coordinate, bindings, integerContext));
  locals.set(localKey(target.name, coordinates), value);
  if (coordinates.length === 0) scalarLocals[target.name] = value;
}

function localKey(name: string, coordinates: readonly number[]): string {
  return `${name}[${coordinates.join(",")}]`;
}
