import { createHash } from 'node:crypto';

/** Cache only pure lexical fingerprints, never ranges, paths, producers or
 * source expressions. Keys are tuples of bounded operator/operand identities;
 * domain-dependent inference must never use this cache.
 */
export class DirectSourceIdentities {
  private readonly entries = new Map<string, string>();
  hits = 0;
  misses = 0;
  constructor(readonly capacity = 8192) {
    if (!Number.isSafeInteger(capacity) || capacity < 0) throw new RangeError('Invalid identity cache capacity');
  }
  get size(): number { return this.entries.size; }
  key(...values: string[]): string {
    const text = values.join('|');
    const found = this.entries.get(text);
    if (found !== undefined) { this.hits++; return found; }
    this.misses++;
    const result = createHash('sha256').update(text).digest('hex');
    // Oversized source names are deliberately not retained.
    if (this.capacity && text.length <= 512) {
      if (this.entries.size >= this.capacity) this.entries.delete(this.entries.keys().next().value!);
      this.entries.set(text, result);
    }
    return result;
  }
}
const identities = new DirectSourceIdentities();
export function directSourceIdentity(...values: string[]): string { return identities.key(...values); }
