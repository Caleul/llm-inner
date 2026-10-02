/** Diagnostic evaluator only. Never used by a generated model. */
import {decodeIeeeF16ToF32} from './utils.js';
import {readFileSync} from 'node:fs';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from './fixed-f16-projection.js';
import {foldDeclaredCpuF32Exponential} from './direct-rust-numeric.js';
import {jsonInteger,jsonWidths,validateJsonNode,type JsonDtype,type JsonExpression} from './direct-json-expression.js';
export type JsonValue=number|bigint|boolean;
let siluProfile:Buffer|undefined;
function halfBits(value:number):number {
  if(Object.is(value,-0))return 0x8000;
  const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,value);
  return roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0)));
}
export function jsonConstantValue(node: JsonExpression): JsonValue {
  if(node[0]!=='constant')throw new TypeError('Constant required');
  validateJsonNode(node);
  const type=node[1],raw=node[2];
  if(type==='bool')return raw==='1';
  const bits=BigInt(raw);
  if(jsonInteger(type))return bits;
  if(type==='f16')return decodeIeeeF16ToF32(Number(bits));
  const data=new DataView(new ArrayBuffer(8));
  if(type==='f32'){data.setUint32(0,Number(bits));return data.getFloat32(0);}
  data.setBigUint64(0,bits);return data.getFloat64(0);
}
function normalize(type:JsonDtype,value:JsonValue):JsonValue {
  if(type==='bool')return Boolean(value);
  if(jsonInteger(type))return BigInt.asUintN(jsonWidths[type],BigInt(value));
  return type==='f32'?Math.fround(Number(value)):Number(value);
}
function reinterpret(from:JsonDtype,to:JsonDtype,value:JsonValue):JsonValue {
  if(from===to)return value;
  const data=new DataView(new ArrayBuffer(8)),width=jsonWidths[from];
  if(jsonInteger(from)){
    const bits=BigInt(value);
    if(to==='f16')return decodeIeeeF16ToF32(Number(bits));
    if(width===32){data.setUint32(0,Number(bits));return data.getFloat32(0);}
    if(width===64){data.setBigUint64(0,bits);return data.getFloat64(0);}
  }else if(jsonInteger(to)){
    if(from==='f16')return BigInt(halfBits(Number(value)));
    if(from==='f32'){data.setFloat32(0,Number(value));return BigInt(data.getUint32(0));}
    if(from==='f64'){data.setFloat64(0,Number(value));return data.getBigUint64(0);}
  }
  throw new TypeError('Unsupported diagnostic bit reinterpretation');
}
export function evaluateJsonExpression(node:JsonExpression,inputs:Readonly<Record<string,JsonValue>>={},
  options:{allowPendingPrimitives?:boolean}={}):JsonValue {
  // Diagnostic cache only, scoped to one input. Sharing compiler syntax does
  // not require repeated harness evaluation; this cache is never serialized.
  const values=new WeakMap<object,JsonValue>();
  function evaluate(node:JsonExpression):JsonValue {
    const hit=values.get(node);if(hit!==undefined)return hit;
    const value=calculate(node);values.set(node,value);return value;
  }
  function calculate(node:JsonExpression):JsonValue {
  validateJsonNode(node);
  if(node[0]==='constant')return jsonConstantValue(node);
  if(node[0]==='input'){
    const value=inputs[node[2]];
    if(value===undefined)throw new Error(`Missing input ${node[2]}`);
    if(node[1]==='bool'&&typeof value!=='boolean'||jsonInteger(node[1])&&typeof value!=='bigint'||
      node[1].startsWith('f')&&typeof value!=='number')throw new TypeError('Input dtype mismatch');
    if(jsonInteger(node[1])&&BigInt.asUintN(jsonWidths[node[1]],value as bigint)!==value)throw new RangeError('Integer input outside dtype');
    if(node[1]==='f32'&&!Object.is(Math.fround(value as number),value))throw new RangeError('F32 input must be exactly representable');
    if(node[1]==='f16'&&(!Number.isFinite(value as number)||!Object.is(decodeIeeeF16ToF32(halfBits(value as number)),value)))
      throw new RangeError('Finite F16 input must be exactly representable');
    return value;
  }
  const args=node.slice(2) as JsonExpression[],op=node[0],type=node[1];
  if(options.allowPendingPrimitives===false&&(op.startsWith('pending-')||op==='widen'||
    type==='f32'&&['add','sub','mul','div'].includes(op)))throw new Error('Diagnostic closed evaluation rejects pending primitives and implicit F32 rounding');
  if(op==='if')return evaluate(args[evaluate(args[0]!)?1:2]!);
  const a=evaluate(args[0]!);
  if(op==='not')return !a;
  if(op==='reinterpret')return reinterpret(args[0]![1],type,a);
  if(op==='convert')return normalize(type,a);
  if(op==='widen')return normalize(type,a);
  if(op==='pending-sqrt')return Math.fround(Math.sqrt(a as number));
  if(op==='pending-exp')return foldDeclaredCpuF32Exponential(a as number);
  if(op==='pending-silu'){
    siluProfile??=readFileSync(new URL('../../numeric-profiles/pytorch-2.12.1-cpu-f16-silu.bin',import.meta.url));
    if(siluProfile.length!==131072)throw new Error('Invalid diagnostic SiLU profile');
    return decodeIeeeF16ToF32(siluProfile.readUInt16LE(2*halfBits(a as number)));
  }
  const b=evaluate(args[1]!);
  if(op==='eq')return a===b;
  if(op==='lt')return a<b;
  if(op==='le')return a<=b;
  if(jsonInteger(type)){
    const x=a as bigint,y=b as bigint;
    if((op==='shl'||op==='shr')&&(y<0n||y>=BigInt(jsonWidths[type])))throw new RangeError('Undefined shift count');
    return normalize(type,op==='add'?x+y:op==='sub'?x-y:op==='mul'?x*y:op==='div'?x/y:
      op==='and'?x&y:op==='or'?x|y:op==='xor'?x^y:op==='shl'?x<<y:x>>y);
  }
  const x=a as number,y=b as number;
  return normalize(type,op==='add'?x+y:op==='sub'?x-y:op==='mul'?x*y:x/y);
  }
  return evaluate(node);
}
