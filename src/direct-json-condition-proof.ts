import type {JsonFloatRange} from './direct-json-range.js';
/** Decisions over certified finite scalar intervals, before producer lowering.
 * Unknown/nonfinite domains never justify removing a reachable branch. */
export function proveJsonFiniteComparison(operation:string,a?:JsonFloatRange,b?:JsonFloatRange):boolean|undefined {
  if(!a||!b||![a.minimum,a.maximum,b.minimum,b.maximum].every(Number.isFinite)||
    a.minimum>a.maximum||b.minimum>b.maximum)return undefined;
  if(operation==='lt'){
    if(a.maximum<b.minimum)return true;
    if(a.minimum>=b.maximum)return false;
  }else if(operation==='le'){
    if(a.maximum<=b.minimum)return true;
    if(a.minimum>b.maximum)return false;
  }else if(operation==='eq'){
    if(a.maximum<b.minimum||b.maximum<a.minimum)return false;
    if(a.minimum===a.maximum&&b.minimum===b.maximum&&a.minimum===b.minimum)return true;
  }
  return undefined;
}
