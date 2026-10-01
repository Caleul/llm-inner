/** Diagnostic evaluator only. Never used by a generated model. */
import {decodeIeeeF16ToF32} from './utils.js';
import {jsonInteger,jsonWidths,validateJsonNode,type JsonDtype,type JsonExpression} from './direct-json-expression.js';
export type JsonValue=number|bigint|boolean;
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
    if(from==='f32'){data.setFloat32(0,Number(value));return BigInt(data.getUint32(0));}
    if(from==='f64'){data.setFloat64(0,Number(value));return data.getBigUint64(0);}
  }
  throw new TypeError('Unsupported diagnostic bit reinterpretation');
}
export function evaluateJsonExpression(node:JsonExpression,inputs:Readonly<Record<string,JsonValue>>={}):JsonValue {
  validateJsonNode(node);
  if(node[0]==='constant')return jsonConstantValue(node);
  if(node[0]==='input'){
    const value=inputs[node[2]];
    if(value===undefined)throw new Error(`Missing input ${node[2]}`);
    if(node[1]==='bool'&&typeof value!=='boolean'||jsonInteger(node[1])&&typeof value!=='bigint'||
      node[1].startsWith('f')&&typeof value!=='number')throw new TypeError('Input dtype mismatch');
    if(jsonInteger(node[1])&&BigInt.asUintN(jsonWidths[node[1]],value as bigint)!==value)throw new RangeError('Integer input outside dtype');
    if(node[1]==='f32'&&!Object.is(Math.fround(value as number),value))throw new RangeError('F32 input must be exactly representable');
    return value;
  }
  const args=node.slice(2) as JsonExpression[],op=node[0],type=node[1];
  if(op==='if')return evaluateJsonExpression(args[evaluateJsonExpression(args[0]!,inputs)?1:2]!,inputs);
  const a=evaluateJsonExpression(args[0]!,inputs);
  if(op==='not')return !a;
  if(op==='reinterpret')return reinterpret(args[0]![1],type,a);
  if(op==='convert')return normalize(type,a);
  const b=evaluateJsonExpression(args[1]!,inputs);
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
