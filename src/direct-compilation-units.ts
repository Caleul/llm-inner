import { createHash } from 'node:crypto';
import { DirectBranchDomain } from './direct-branch-domain.js';
import { FlatConditions } from './direct-flat-substitution.js';
import { exactNumberRational, finiteIeeeValue } from './direct-round-preimage.js';
import type { DirectRustStream } from './direct-rust-stream.js';

export interface DirectInputPartition { coordinate: number; first: number; last: number }
export interface DirectCompilationUnit {
  position: number;
  fullVectorSoftmax: boolean;
  partition?: DirectInputPartition;
}
// Sorted finite numerical F16 values. Both signed zeros belong to index 31743
// and must travel together; comparisons cannot separate their signs.
export const finiteF16Count = 63487;
export function* directCompilationUnits(context: number, width: number, partitions = 1,
  coordinate = 0): Generator<DirectCompilationUnit> {
  if (!Number.isSafeInteger(context) || context < 1 || !Number.isSafeInteger(width) || width < 1 ||
      !Number.isSafeInteger(partitions) || partitions < 1 || partitions > finiteF16Count ||
      !Number.isSafeInteger(coordinate) || coordinate < 0 || coordinate >= width)
    throw new RangeError('Invalid compilation geometry or partitions');
  for (let position = 0; position < context; position++) {
    const variants = position === 2 && context >= 4 ? [false, true] : [position >= 3];
    for (const fullVectorSoftmax of variants) for (let shard = 0; shard < partitions; shard++) {
      const unit: DirectCompilationUnit = { position, fullVectorSoftmax };
      if (partitions > 1) unit.partition = { coordinate,
        first: Math.floor(shard * finiteF16Count / partitions),
        last: Math.floor((shard + 1) * finiteF16Count / partitions) - 1 };
      yield unit;
    }
  }
}
export function validateDirectUnit(unit: DirectCompilationUnit, context: number, width: number): void {
  if (!Number.isSafeInteger(unit.position) || unit.position < 0 || unit.position >= context ||
      typeof unit.fullVectorSoftmax !== 'boolean' ||
      (unit.position !== 2 || context < 4) && unit.fullVectorSoftmax !== (unit.position >= 3))
    throw new RangeError('Invalid compilation unit');
  const p = unit.partition;
  if (p && (!Number.isSafeInteger(p.coordinate) || p.coordinate < 0 || p.coordinate >= width ||
      !Number.isSafeInteger(p.first) || !Number.isSafeInteger(p.last) ||
      p.first < 0 || p.last < p.first || p.last >= finiteF16Count))
    throw new RangeError('Invalid F16 input partition');
}
export function directUnitConditions(unit: DirectCompilationUnit, context: number,
  width: number, stream: DirectRustStream): FlatConditions {
  validateDirectUnit(unit, context, width);
  const guards = [() => stream.write(`t==${unit.position}`)];
  if (unit.position === 2 && context >= 4)
    guards.push(() => stream.write(unit.fullVectorSoftmax ? 'n>=4' : 'n==3'));
  const base = new FlatConditions(new DirectBranchDomain(), [], guards);
  if (!unit.partition) return base;
  const p = unit.partition, source = `input_tokens[${unit.position}][${p.coordinate}]`;
  const key = createHash('sha256').update(source).digest('hex');
  const narrowed = base.refine(key, () => stream.write(source), {
    lower: { value: exactNumberRational(finiteIeeeValue('f16', p.first)), inclusive: true },
    upper: { value: exactNumberRational(finiteIeeeValue('f16', p.last)), inclusive: true },
  }, true);
  if (!narrowed) throw new Error('Empty F16 partition');
  return narrowed;
}
export function directRustHeader(width: number, context: number): string {
  return `// Input: finite F16 embedding matrix, widened exactly to f64.\n// Policy: PyTorch CPU arm64; scalar substitution, flat path conditions.\n#![recursion_limit="65536"]\npub fn compiled_dimension(input_tokens:&[[f64;${width}]],t:usize)->f64 {let n=input_tokens.len();assert!(n>0 && n<=${context} && t<n);'result:{`;
}
export const directRustFooter = 'panic!("outside declared embedding domain")}}\n';
