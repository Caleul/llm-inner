/** Compilation-only scalar syntax. No stage, tensor, weight reference or executor
 * is serialized. Each operation carries its actual result type. Float constants
 * are bit strings so JSON never normalizes -0 or rounds checkpoint values. */
export type JsonDtype = 'bool' | 'u16' | 'u32' | 'u64' | 'f16' | 'f32' | 'f64';
export type JsonOperator = 'add' | 'sub' | 'mul' | 'div' | 'and' | 'or' | 'xor' |
  'shl' | 'shr' | 'eq' | 'lt' | 'le' | 'not' | 'if' | 'reinterpret' | 'convert' |
  'widen' | 'pending-sqrt' | 'pending-exp' | 'pending-silu';
export type JsonExpression = readonly ['constant', JsonDtype, string] |
  readonly ['input', JsonDtype, string] |
  readonly [JsonOperator, JsonDtype, ...JsonExpression[]];
export const jsonWidths: Record<JsonDtype, number> = {bool:1,u16:16,u32:32,u64:64,f16:16,f32:32,f64:64};
export const jsonInteger = (type: JsonDtype) => type.startsWith('u');
export function jsonConstant(type: JsonDtype, value: number | bigint | boolean): JsonExpression {
  if(type==='bool') {
    if(typeof value!=='boolean')throw new TypeError('Boolean constant required');
    return ['constant',type,value?'1':'0'];
  }
  const width=jsonWidths[type];
  let bits:bigint;
  if(jsonInteger(type)) {
    if(typeof value==='number'&&!Number.isSafeInteger(value))throw new TypeError('Exact integer constant required');
    if(typeof value==='boolean')throw new TypeError('Integer constant required');
    bits=BigInt.asUintN(width,BigInt(value));
  }else{
    if(typeof value!=='number')throw new TypeError('Floating constant required');
    if(type==='f16')throw new TypeError('Use exact F16 bits, not an implicit conversion');
    const data=new DataView(new ArrayBuffer(8));
    if(type==='f32'){data.setFloat32(0,value);bits=BigInt(data.getUint32(0));}
    else{data.setFloat64(0,value);bits=data.getBigUint64(0);}
  }
  return ['constant',type,'0x'+bits.toString(16).padStart(width/4,'0')];
}
export function jsonInput(type: JsonDtype, name: string): JsonExpression {
  if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(name))throw new TypeError('Invalid input name');
  return ['input',type,name];
}
export function jsonOperation(op: JsonOperator,type: JsonDtype,...args: JsonExpression[]): JsonExpression {
  const result:JsonExpression=[op,type,...args];validateJsonNode(result);return result;
}
const arities:Record<JsonOperator,number>={add:2,sub:2,mul:2,div:2,and:2,or:2,xor:2,
  shl:2,shr:2,eq:2,lt:2,le:2,not:1,if:3,reinterpret:1,convert:1,
  widen:1,'pending-sqrt':1,'pending-exp':1,'pending-silu':1};
export function validateJsonNode(node: JsonExpression): void {
  if(!Array.isArray(node)||!(node[1] in jsonWidths))throw new TypeError('Invalid scalar JSON node');
  const [op,type]=node;
  if(op==='constant'){
    const bits=node[2];
    if(node.length!==3||typeof bits!=='string'||!(type==='bool'?/^[01]$/.test(bits):
      new RegExp('^0x[0-9a-f]{'+jsonWidths[type]/4+'}$').test(bits)))throw new TypeError('Invalid exact constant bits');
    return;
  }
  if(op==='input'){
    if(node.length!==3||typeof node[2]!=='string'||!/^[A-Za-z][A-Za-z0-9_]*$/.test(node[2]))throw new TypeError('Invalid fundamental input');
    return;
  }
  if(!(op in arities)||node.length!==arities[op]+2)throw new TypeError('Unknown operator or invalid arity');
  const args=node.slice(2) as JsonExpression[];
  if(args.some(x=>!Array.isArray(x)||!(x[1] in jsonWidths)))throw new TypeError('Invalid operation operand');
  if(op==='if'){
    if(args[0]![1]!=='bool'||args[1]![1]!==type||args[2]![1]!==type)throw new TypeError('Invalid conditional types');
  }else if(op==='not'){
    if(type!=='bool'||args[0]![1]!=='bool')throw new TypeError('Invalid Boolean negation');
  }else if(op==='eq'||op==='lt'||op==='le'){
    if(type!=='bool'||args[0]![1]!==args[1]![1])throw new TypeError('Invalid comparison types');
  }else if(op==='reinterpret'){
    if(jsonWidths[type]!==jsonWidths[args[0]![1]])throw new TypeError('Bit reinterpretation changes width');
    if(type!==args[0]![1]&&jsonInteger(type)===jsonInteger(args[0]![1]))throw new TypeError('Bit reinterpretation requires matching float and integer widths');
  }else if(op==='convert'){
    // Only unsigned integer width conversion. Floating conversions must be lowered.
    if(!jsonInteger(type)||!jsonInteger(args[0]![1]))throw new TypeError('Unlowered floating conversion');
  }else if(op==='widen'){
    if(!type.startsWith('f')||!args[0]![1].startsWith('f')||jsonWidths[type]<=jsonWidths[args[0]![1]])
      throw new TypeError('Invalid exact floating widening');
  }else if(op.startsWith('pending-')){
    const expected=op==='pending-silu'?'f16':'f32';
    if(type!==expected||args[0]![1]!==expected)throw new TypeError('Invalid pending primitive dtype');
  }else{
    if(args.some(x=>x[1]!==type)||type==='bool')throw new TypeError('Invalid arithmetic types');
    if(['and','or','xor','shl','shr'].includes(op)&&!jsonInteger(type))throw new TypeError('Bitwise operand must be unsigned');
    if(type==='f16')throw new TypeError('F16 arithmetic must expose accumulation and conversion');
  }
}
