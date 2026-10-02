import {jsonConstant as c,jsonOperation as o,jsonInput,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue,evaluateJsonExpression} from './direct-json-evaluator.js';
import {lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';
import {lowerJsonFiniteF16AsF64,lowerJsonNormalF32ThenF16AsF64} from './direct-json-half-value.js';
import {lowerJsonPositiveNormalSqrtAsF64} from './direct-json-sqrt.js';
import {lowerJsonSmallNonpositiveExpAsF64} from './direct-json-exp.js';
import {certifyJsonSmallSilu,lowerJsonSmallSiluAsF64} from './direct-json-silu.js';
import type {JsonModelLoweringFacts} from './direct-json-model.js';
import {decodeIeeeF16ToF32} from './utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from './fixed-f16-projection.js';
import type {JsonScalarHeader} from './direct-json-stream.js';
import type {JsonPrecisionFacts} from './direct-json-precision.js';
import {jsonModelRangeAnalysis} from './direct-json-range.js';

/** Runtime arguments were already finite F16 values exactly widened to F64.
 * Declare that input contract instead of encoding and decoding X at every use. */
export function loweredJsonHeader(header:JsonScalarHeader):JsonScalarHeader {
  return {...header,inputs:Object.fromEntries(Object.entries(header.inputs).map(([name,binding])=>
    [name,'source' in binding?binding:{...binding,dtype:'f64' as const,inputDtype:'f16' as const}]))};
}

/** Replace compiler numerical boundaries with closed scalar bit/arithmetic
 * expressions. Keep F16/F32 values exactly widened while composing consumers,
 * instead of encoding/decoding the same value at every adjacent operation.
 * The builder's finite-domain certificate and width/context limits establish
 * that nonzero F32 arithmetic results are normal, not subnormal. */
export function lowerJsonModelExpression(root:JsonExpression,facts:JsonModelLoweringFacts,
  precision:JsonPrecisionFacts=new WeakMap()):JsonExpression {
  const memo=new WeakMap<object,JsonExpression>();
  const f32Sources=new WeakMap<JsonExpression,JsonExpression>();
  const range=jsonModelRangeAnalysis(facts);
  const value=(node:JsonExpression)=>node[0]==='constant'?Number(jsonConstantValue(node)):undefined;
  const halfConstant=(node:JsonExpression)=>{
    if(node[0]!=='constant'||node[1]!=='f32')return false;
    const x=value(node)!;if(!Number.isFinite(x))return false;
    if(Object.is(x,-0))return true;
    const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,x);
    return Object.is(decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0)))),x);
  };
  const halfOperand=(node:JsonExpression)=>node[0]==='widen'&&node[2]![1]==='f16'||halfConstant(node);
  function visit(node:JsonExpression):JsonExpression {
    const hit=memo.get(node);if(hit)return hit;
    const result=calculate(node);
    // Every completed F16/F32 producer is exactly widened. These are compiler
    // proofs, not retained conversion operators or runtime metadata.
    if(node[1]==='f16')precision.set(result,42);
    else if(node[1]==='f32')precision.set(result,29);
    memo.set(node,result);return result;
  }
  function calculate(node:JsonExpression):JsonExpression {
    const halfSource=facts.halfSources.get(node);
    if(halfSource){
      if(halfSource[0]==='widen'&&halfSource[2]![1]==='f16')return visit(halfSource[2]!);
      if(halfConstant(halfSource))return c('f64',value(halfSource)!);
      if(halfSource[0]==='mul'){
        const a=halfSource[2]!,b=halfSource[3]!;
        if(halfOperand(a)&&value(b)===1)return visit(a);
        if(halfOperand(b)&&value(a)===1)return visit(b);
      }
      const source=visit(halfSource);
      if(source[0]==='constant'){
        const x=Number(jsonConstantValue(source));
        const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,x);
        const bits=Object.is(x,-0)?0x8000:roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0)));
        return c('f64',decodeIeeeF16ToF32(bits));
      }
      const interval=range(halfSource),raw=f32Sources.get(source);
      if(raw&&interval&&(interval.minimum>=2**-14||interval.maximum<=-(2**-14))&&
        Math.max(Math.abs(interval.minimum),Math.abs(interval.maximum))<65520)
        return lowerJsonNormalF32ThenF16AsF64(raw);
      return lowerJsonFiniteF16AsF64(source,interval);
    }
    if(node[0]==='constant')return node[1].startsWith('f')?c('f64',Number(jsonConstantValue(node))):node;
    if(node[0]==='input')return node[1]==='f16'?jsonInput('f64',node[2]):node;
    const args=node.slice(2) as JsonExpression[];
    if(node[0]==='widen')return visit(args[0]!);
    if(node[0]==='pending-sqrt'){
      if(!facts.positiveNormalRoots.has(node))throw new Error('Missing positive normal root certificate');
      const expression=lowerJsonPositiveNormalSqrtAsF64(visit(args[0]!));
      return args[0]![0]==='constant'?c('f64',Number(evaluateJsonExpression(expression))):expression;
    }
    if(node[0]==='pending-exp'){
      const bound=facts.exponentialBounds.get(node);
      if(bound===undefined||bound>.34)throw new Error('Small exponential domain is not proved for this checkpoint');
      const input=visit(args[0]!);
      const expression=lowerJsonSmallNonpositiveExpAsF64(input);
      return input[0]==='constant'?c('f64',Number(evaluateJsonExpression(expression))):expression;
    }
    if(node[0]==='pending-silu'){
      const bound=facts.activationBounds.get(node);
      if(bound===undefined)throw new Error('Missing checkpoint activation range');
      const certificate=certifyJsonSmallSilu(bound),input=visit(args[0]!);
      const expression=lowerJsonSmallSiluAsF64(input,certificate.polynomialDegree);
      return input[0]==='constant'?c('f64',Number(evaluateJsonExpression(expression))):expression;
    }
    if(node[0]==='if')return o('if',node[1].startsWith('f')?'f64':node[1],...args.map(visit));
    if(node[1]==='bool')return o(node[0],'bool',...args.map(visit));
    if(node[1]==='f32'&&['add','sub','mul','div'].includes(node[0])){
      const a=args[0]!,b=args[1]!,expression=o(node[0],'f64',visit(a),visit(b));
      if(node[0]==='sub'&&expression[2]===expression[3])return c('f64',0);
      if(expression[2]![0]==='constant'&&expression[3]![0]==='constant')
        return c('f64',Math.fround(Number(evaluateJsonExpression(expression))));
      // Products of two finite F16 values have <=22 significant bits and are
      // exact normal F32 (or zero). Zero/unit operations stay on the F32 grid;
      // retain the actual F64 operation so signed-zero rules remain visible.
      const exact=node[0]==='mul'&&(halfOperand(a)&&halfOperand(b)||value(a)===0||value(b)===0||
        Math.abs(value(a)??NaN)===1||Math.abs(value(b)??NaN)===1)||
        (node[0]==='add'||node[0]==='sub')&&(value(a)===0||value(b)===0)||
        node[0]==='div'&&Math.abs(value(b)??NaN)===1;
      if(exact)return expression;
      const rounded=lowerJsonRoundNormalF32AsF64(expression);
      f32Sources.set(rounded,expression);return rounded;
    }
    return o(node[0],node[1],...args.map(visit));
  }
  return visit(root);
}
