import type { Gemma4LiteralScalarCalculation } from "./gemma4-literal-scalar-calculations.js";
import type { Gemma4LiteralScalarExpression, Gemma4LiteralScalarStatementProgram } from "./gemma4-literal-scalar-statement-programs.js";
import { Gemma4ParametricRealBuilder } from "./gemma4-parametric-real-expression.js";
import {
  lowerGemma4IntegerExpressionToParametricReal,
  lowerGemma4ScalarExpressionToParametricReal,
} from "./gemma4-parametric-real-lowering.js";

export interface Gemma4ParametricScalarCompilationContext {
  builder: Gemma4ParametricRealBuilder;
  scope: string;
  integerBindings?: Readonly<Record<string, string>>;
  resolveTensorElement: (name: string, coordinates: readonly string[]) => string;
  resolveLearnedElement: (role: string, coordinates: readonly string[]) => string;
  resolveTensorAxis: (name: string, axis: number) => string;
}

export interface Gemma4ParametricScalarCompilation {
  root: string;
  compiledLocals: number;
  preconditions: string[];
}

/** Lazily substitutes local statement producers for one symbolic output coordinate. */
export function compileGemma4ScalarCalculationToParametricReal(
  calculation: Gemma4LiteralScalarCalculation,
  outputCoordinates: readonly string[],
  context: Gemma4ParametricScalarCompilationContext,
): Gemma4ParametricScalarCompilation {
  if (outputCoordinates.length !== calculation.outputCoordinates.length) {
    throw new Error(`${context.scope}: output paramétrico possui rank ${outputCoordinates.length}; esperado ${calculation.outputCoordinates.length}.`);
  }
  const baseBindings: Record<string, string> = {
    ...(context.integerBindings ?? {}),
    ...Object.fromEntries(calculation.outputCoordinates.map((name, index) => [name, outputCoordinates[index]!])),
  };
  const localProducers = new Map<string, Gemma4LiteralScalarStatementProgram>();
  const outputPrograms: Gemma4LiteralScalarStatementProgram[] = [];
  const preconditions = calculation.statementPrograms.filter((program) => program.kind === "require").map((program) => program.source);
  for (const program of calculation.statementPrograms) for (const target of program.targets) {
    if (target.role === "output") outputPrograms.push(program);
    else {
      if (localProducers.has(target.name)) {
        const previous = localProducers.get(target.name)!;
        if (previous !== program) throw new Error(`${context.scope}: local paramétrico ${target.name} possui produtores distintos.`);
      }
      localProducers.set(target.name, program);
    }
  }
  const output = outputPrograms.at(-1);
  if (!output || output.kind !== "assignment") throw new Error(`${context.scope}: output paramétrico terminal ausente.`);
  const memo = new Map<string, string>(), active = new Set<string>();
  const compileExpression = (expression: Gemma4LiteralScalarExpression, bindings: Readonly<Record<string, string>>, scope: string): string =>
    lowerGemma4ScalarExpressionToParametricReal(expression, {
      builder: context.builder,
      scope,
      integerBindings: bindings,
      resolveScalar: (name, currentBindings) => {
        if (name === "Infinity") return context.builder.positiveInfinity();
        return compileLocal(name, [], currentBindings);
      },
      resolveTensorElement: (name, coordinates) => localProducers.has(name)
        ? compileLocal(name, coordinates, bindings)
        : context.resolveTensorElement(name, coordinates),
      resolveLearnedElement: context.resolveLearnedElement,
      resolveTensorAxis: context.resolveTensorAxis,
    });
  const compileLocal = (name: string, coordinates: readonly string[], outerBindings: Readonly<Record<string, string>>): string => {
    const key = `${name}[${coordinates.join(",")}]@${Object.entries(outerBindings).sort().map(([binding, value]) => `${binding}=${value}`).join(";")}`;
    const cached = memo.get(key);
    if (cached) return cached;
    if (active.has(key)) throw new Error(`${context.scope}: ciclo local paramétrico em ${key}.`);
    const producer = localProducers.get(name);
    if (!producer) throw new Error(`${context.scope}: identificador paramétrico livre ${name}.`);
    const target = producer.targets.find((candidate) => candidate.name === name);
    if (!target || target.coordinates.length !== coordinates.length || producer.targets.length !== 1) {
      throw new Error(`${context.scope}: target local paramétrico ${name} requer lowering estruturado.`);
    }
    const bindings: Record<string, string> = { ...outerBindings };
    target.coordinates.forEach((coordinate, index) => bindTargetCoordinate(coordinate, coordinates[index]!, bindings, context, `${context.scope}/${name}`));
    active.add(key);
    const expression = producer.expression.kind === "ordered-loop" ? producer.expression.body : producer.expression;
    if (producer.expression.kind === "ordered-loop" && !bindings[producer.expression.index]) {
      throw new Error(`${context.scope}: loop local ${producer.expression.index} não foi ligado pelo target.`);
    }
    const result = compileExpression(expression, bindings, `${context.scope}/local-${name}`);
    active.delete(key);
    memo.set(key, result);
    return result;
  };
  const root = compileExpression(output.expression, baseBindings, `${context.scope}/output`);
  return { root, compiledLocals: memo.size, preconditions };
}

function bindTargetCoordinate(
  expression: Gemma4LiteralScalarExpression,
  value: string,
  bindings: Record<string, string>,
  context: Gemma4ParametricScalarCompilationContext,
  scope: string,
): void {
  if (expression.kind === "identifier") {
    // A local indexed producer is a lambda over its target coordinates. Its
    // formal index deliberately shadows any same-named reduction in the
    // consumer while receiving the consumer's concrete/symbolic argument.
    bindings[expression.name] = value;
    return;
  }
  const actual = lowerGemma4IntegerExpressionToParametricReal(expression, {
    builder: context.builder, scope, integerBindings: bindings, resolveTensorAxis: context.resolveTensorAxis,
  });
  if (actual !== value) throw new Error(`${scope}: coordenada local paramétrica não coincide com o acesso consumidor.`);
}
