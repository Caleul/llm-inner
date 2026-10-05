import {jsonConstant as c,jsonOperation as o,jsonInput,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue,evaluateJsonExpression} from './direct-json-evaluator.js';
import {lowerJsonRoundNormalF32AsF64} from './direct-json-f16.js';
import {lowerJsonFiniteF16AsF64,lowerJsonFiniteF32ThenF16AsF64} from './direct-json-half-value.js';
import {lowerJsonRationalPositiveNormalSqrtAsF64,lowerJsonRationalPositiveNormalInverseSqrtAsF64} from './direct-json-rational-sqrt.js';
import {lowerJsonSmallNonpositiveExpAsF64,certifyJsonHalfDifferenceExp} from './direct-json-exp.js';
import {certifyJsonSmallSilu,lowerJsonSmallSiluAsF64} from './direct-json-silu.js';
import type {JsonModelLoweringFacts} from './direct-json-model.js';
import {decodeIeeeF16ToF32} from './utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from './fixed-f16-projection.js';
import type {JsonScalarHeader} from './direct-json-stream.js';
import type {JsonPrecisionFacts} from './direct-json-precision.js';
import {jsonModelRangeAnalysis} from './direct-json-range.js';
import {simplifyJsonAffine,type JsonAffineDomains} from './direct-json-affine.js';
import {proveJsonFiniteComparison} from './direct-json-condition-proof.js';
import {constantJsonF32Cell} from './direct-json-rounding-cell.js';
import {maximumF32MagnitudeForHalfBound} from './direct-json-half-preimage.js';
import {jsonModelMagnitudeAnalysis} from './direct-json-magnitude.js';
import {jsonResidualCellThreshold} from './direct-json-residual-cell.js';
import {createJsonModelSignProof} from './direct-json-sign.js';
import {simplifyJsonFixedPoint,sameJsonExpression,JsonSimplificationSession} from './direct-json-simplify.js';
import {simplifyJsonBitPrecision} from './direct-json-precision.js';
import {shareJsonExpression} from './direct-json-share.js';
import {measureJsonExpression,type JsonExpressionMeasure} from './direct-json-measure.js';

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
  return createJsonModelLowerer(facts,precision).lower(root);
}

export interface JsonSubstitutionProgress {
  ordinal:number;operation:string;before:JsonExpressionMeasure;after:JsonExpressionMeasure;passes:number;
  simplificationCacheHits:number;simplificationCacheMisses:number;
}
/** One session per output coordinate. Completed producers are simplified before
 * a consumer substitutes them. Memoization retains compiler syntax, never
 * forward values, and no identifiers or aliases enter the literal artifact. */
export function createJsonModelLowerer(facts:JsonModelLoweringFacts,
  precision:JsonPrecisionFacts=new WeakMap(),
  options:{incremental?:boolean;onSubstitution?:(event:JsonSubstitutionProgress)=>void}={}):{
    lower:(root:JsonExpression)=>JsonExpression
  } {
  const memo=new WeakMap<object,JsonExpression>(),simplification=new JsonSimplificationSession();let ordinal=0;
  const f32Sources=new WeakMap<JsonExpression,JsonExpression>();
  const affineDomains:JsonAffineDomains=new WeakMap(),workingAffineDomains:JsonAffineDomains=new WeakMap();
  function retainAffineDomain(map:JsonAffineDomains,node:JsonExpression,domain:NonNullable<ReturnType<JsonAffineDomains['get']>>):void {
    const previous=map.get(node);
    const merged=previous?{minimum:Math.max(previous.minimum,domain.minimum),maximum:Math.min(previous.maximum,domain.maximum),
      quantumExponent:Math.max(previous.quantumExponent,domain.quantumExponent),
      excludesNegativeZero:previous.excludesNegativeZero||domain.excludesNegativeZero}:domain;
    if(merged.minimum>merged.maximum)throw new Error('Contradictory affine producer certificates');
    map.set(node,merged);
  }
  const magnitude=facts.inputMagnitudeBounds?jsonModelMagnitudeAnalysis(facts):undefined;
  const range=jsonModelRangeAnalysis(facts),sourceSign=createJsonModelSignProof(facts,visit,range);
  const value=(node:JsonExpression)=>node[0]==='constant'?Number(jsonConstantValue(node)):undefined;
  const halfConstant=(node:JsonExpression)=>{
    if(node[0]!=='constant'||node[1]!=='f32')return false;
    const x=value(node)!;if(!Number.isFinite(x))return false;
    if(Object.is(x,-0))return true;
    const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,x);
    return Object.is(decodeIeeeF16ToF32(roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0)))),x);
  };
  const halfOperand=(node:JsonExpression)=>node[0]==='widen'&&node[2]![1]==='f16'||halfConstant(node);
  const zeroSignMemo=new WeakMap<JsonExpression,boolean>();
  // Within the admitted finite domain, addition/subtraction of F32 operands
  // cannot underflow to -0: their exact sum is a multiple of 2^-149. A zero
  // addition is negative only when both operands are -0. Half conversions
  // remain unknown because a negative nonzero source can underflow to -0.
  function excludesNegativeZero(node:JsonExpression):boolean {
    const hit=zeroSignMemo.get(node);if(hit!==undefined)return hit;
    let result=false;
    if(node[0]==='constant'&&node[1].startsWith('f')){
      const x=value(node)!;result=Number.isFinite(x)&&!Object.is(x,-0);
    }else if(node[0]==='widen')result=excludesNegativeZero(node[2]!);
    else if(node[0]==='if'&&node[1].startsWith('f'))
      result=excludesNegativeZero(node[3]!)&&excludesNegativeZero(node[4]!);
    else if(node[1]==='f32'&&node[0]==='add')
      result=excludesNegativeZero(node[2]!)||excludesNegativeZero(node[3]!);
    else if(node[1]==='f32'&&node[0]==='sub')result=excludesNegativeZero(node[2]!);
    else if(node[0]==='pending-sqrt')result=facts.positiveNormalRoots.has(node);
    else if(node[0]==='pending-exp')result=facts.exponentialBounds.has(node);
    zeroSignMemo.set(node,result);return result;
  }
  function visit(node:JsonExpression):JsonExpression {
    const hit=memo.get(node);if(hit)return hit;
    const produced=calculate(node);
    let result=produced;
    const sourceRange=range(node);
    if(produced[1]==='f64'&&sourceRange&&(node[1]==='f16'||node[1]==='f32')){
      const minimum=sourceRange.minimum>0?sourceRange.minimum:sourceRange.maximum<0?-sourceRange.maximum:0;
      const quantumExponent=node[1]==='f16'?-24:minimum>0?Math.max(-149,Math.floor(Math.log2(minimum))-23):-149;
      const domain={...sourceRange,quantumExponent:node[0]==='widen'&&node[2]![1]==='f16'?-24:quantumExponent,excludesNegativeZero:excludesNegativeZero(node)||minimum>0};
      retainAffineDomain(affineDomains,produced,domain);retainAffineDomain(workingAffineDomains,node,domain);
    }
    if(options.incremental!==false){
      // Children have already reached their own fixed points. Stabilize the
      // newly substituted producer before making it available to consumers.
      let passes=0,stable=false;
      for(let round=0;round<32;round++){
        const stabilized=simplifyJsonFixedPoint(shareJsonExpression(simplifyJsonAffine(simplifyJsonBitPrecision(result,precision),affineDomains)).expression,32,1_000_000,simplification);
        passes+=stabilized.passes;
        if(sameJsonExpression(result,stabilized.expression)){stable=true;break;}
        result=stabilized.expression;
      }
      if(!stable)throw new Error('JSON substitution rules have not reached a joint fixed point');
      const raw=f32Sources.get(produced);if(raw)f32Sources.set(result,raw);
      const domain=affineDomains.get(produced);if(domain)retainAffineDomain(affineDomains,result,domain);
      if(options.onSubstitution)options.onSubstitution({ordinal:++ordinal,operation:node[0],
        before:measureJsonExpression(produced),after:measureJsonExpression(result),passes,
        simplificationCacheHits:simplification.stats.hits,simplificationCacheMisses:simplification.stats.misses});
    }
    // Every completed F16/F32 producer is exactly widened. These are compiler
    // proofs, not retained conversion operators or runtime metadata.
    // An identity or widening can return an already-completed half producer.
    // Its stronger proof remains valid; the consumer cannot weaken it.
    if(node[1]==='f16'||node[1]==='f32')
      precision.set(result,Math.max(precision.get(result)??0,node[1]==='f16'?42:29));
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
      // For finite half operands, R16(R32(a ± b)) == R16(a ± b).
      // With exponent gap <=12 the exact sum has <=24 significant bits.
      // With larger gaps the smaller operand stays strictly inside the larger
      // operand's half cell, even after F32 rounding (also at binade edges).
      // This includes signed zeros and half overflow. It does not extend to
      // sums of products: their operands can have 22 significant bits.
      if((halfSource[0]==='add'||halfSource[0]==='sub')&&
        halfOperand(halfSource[2]!)&&halfOperand(halfSource[3]!)){
        if(magnitude){
          const a=magnitude(halfSource[2]!),b=magnitude(halfSource[3]!);
          const threshold=b?jsonResidualCellThreshold(b.maximum):undefined;
          if(a&&threshold!==undefined&&a.minimum>=threshold)return visit(halfSource[2]!);
        }
        const raw=o(halfSource[0],'f64',visit(halfSource[2]!),visit(halfSource[3]!));
        return lowerJsonFiniteF16AsF64(raw,range(halfSource),sourceSign(halfSource,raw),magnitude?.(halfSource)?.minimum??0);
      }
      const source=visit(halfSource);
      if(source[0]==='constant'){
        const x=Number(jsonConstantValue(source));
        const data=new DataView(new ArrayBuffer(4));data.setFloat32(0,x);
        const bits=Object.is(x,-0)?0x8000:roundDyadicToF16IfElse(f32BitsToDyadic(data.getUint32(0)));
        return c('f64',decodeIeeeF16ToF32(bits));
      }
      let interval=range(halfSource);
      // A certified finite half output also bounds its F32 producer through
      // the exact rounding preimage. Retain this relation when closing the
      // conversion; an independent numerator/denominator bound loses RMS
      // correlation and incorrectly leaves overflow and double rounding live.
      const outputRange=facts.ranges?.get(node);
      if(outputRange&&halfSource[1]==='f32'){
        const maximum=maximumF32MagnitudeForHalfBound(Math.max(Math.abs(outputRange.minimum),Math.abs(outputRange.maximum)));
        if(maximum!==undefined)interval=interval?{minimum:Math.max(interval.minimum,-maximum),maximum:Math.min(interval.maximum,maximum)}:
          {minimum:-maximum,maximum};
      }
      const raw=f32Sources.get(source);
      if(raw&&interval&&Math.max(Math.abs(interval.minimum),Math.abs(interval.maximum))<65520)
        return lowerJsonFiniteF32ThenF16AsF64(raw,interval,sourceSign(halfSource,source),magnitude?.(halfSource)?.minimum??0);
      return lowerJsonFiniteF16AsF64(source,interval,sourceSign(halfSource,source),magnitude?.(halfSource)?.minimum??0);
    }
    if(node[0]==='constant')return node[1].startsWith('f')?c('f64',Number(jsonConstantValue(node))):node;
    if(node[0]==='input')return node[1]==='f16'?jsonInput('f64',node[2]):node;
    const args=node.slice(2) as JsonExpression[];
    if(node[0]==='widen')return visit(args[0]!);
    if(node[0]==='pending-sqrt'){
      if(!facts.positiveNormalRoots.has(node))throw new Error('Missing positive normal root certificate');
      const input=visit(args[0]!);
      const expression=lowerJsonRationalPositiveNormalSqrtAsF64(input);
      return input[0]==='constant'?c('f64',Number(evaluateJsonExpression(expression))):expression;
    }
    if(node[0]==='pending-exp'){
      const bound=facts.exponentialBounds.get(node);
      if(bound===undefined||bound>.34)throw new Error('Small exponential domain is not proved for this checkpoint');
      const input=visit(args[0]!);
      const scoreBound=facts.exponentialScoreBounds?.get(node);
      const certificate=scoreBound!==undefined&&scoreBound<=.17?certifyJsonHalfDifferenceExp(scoreBound):undefined;
      const expression=lowerJsonSmallNonpositiveExpAsF64(input,certificate);
      return input[0]==='constant'?c('f64',Number(evaluateJsonExpression(expression))):expression;
    }
    if(node[0]==='pending-silu'){
      const bound=facts.activationBounds.get(node);
      if(bound===undefined)throw new Error('Missing checkpoint activation range');
      const certificate=certifyJsonSmallSilu(bound),input=visit(args[0]!);
      const expression=lowerJsonSmallSiluAsF64(input,certificate.polynomialDegree);
      return input[0]==='constant'?c('f64',Number(evaluateJsonExpression(expression))):expression;
    }
    if(node[0]==='if'){
      // Stabilize the condition before touching either branch's producers.
      const condition=simplifyJsonFixedPoint(visit(args[0]!)).expression;
      if(condition[0]==='constant')return visit(args[jsonConstantValue(condition)?1:2]!);
      return o('if',node[1].startsWith('f')?'f64':node[1],condition,visit(args[1]!),visit(args[2]!));
    }
    if(node[1]==='bool'){
      const proof=args.length===2?proveJsonFiniteComparison(node[0],range(args[0]!),range(args[1]!)):undefined;
      if(proof!==undefined)return c('bool',proof);
      return o(node[0],'bool',...args.map(visit));
    }
    if(node[1]==='f32'&&['add','sub','mul','div'].includes(node[0])){
      const a=args[0]!,b=args[1]!;
      if(node[0]==='div'&&value(a)===1&&b[0]==='pending-sqrt'&&facts.positiveNormalRoots.has(b)){
        const input=visit(b[2]!);
        const closed=lowerJsonRationalPositiveNormalInverseSqrtAsF64(input);
        return input[0]==='constant'?c('f64',Number(evaluateJsonExpression(closed))):closed;
      }
      const cell=constantJsonF32Cell(node[0],range(a),range(b));
      if(cell!==undefined)return c('f64',cell);
      // Complete operands before composing their affine source forms. The
      // root-only pass keeps all opaque producer identities and their facts.
      const leftProducer=visit(a),rightProducer=visit(b);
      const affineStats={visited:0,rewrites:0,barriers:0,exactRoot:false};
      const composed=simplifyJsonAffine(node,workingAffineDomains,affineStats,true);
      if(!sameJsonExpression(composed,node))return visit(composed);
      const expression=o(node[0],'f64',leftProducer,rightProducer);
      if(node[0]==='sub'&&sameJsonExpression(expression[2] as JsonExpression,expression[3] as JsonExpression))return c('f64',0);
      const left=expression[2] as JsonExpression,right=expression[3] as JsonExpression;
      const leftValue=value(left),rightValue=value(right);
      // Returning the already-rounded producer also preserves its fusion
      // provenance. Keep +0 additions when they canonicalize a possible -0.
      if(node[0]==='add'){
        if(rightValue===0&&(Object.is(rightValue,-0)||excludesNegativeZero(a)))return left;
        if(leftValue===0&&(Object.is(leftValue,-0)||excludesNegativeZero(b)))return right;
      }
      if(node[0]==='sub'&&rightValue===0&&(!Object.is(rightValue,-0)||excludesNegativeZero(a)))return left;
      if(facts.exactRmsMeans?.has(node)||affineStats.exactRoot)return expression;
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
      // Every finite half is an integer multiple of 2^-24. A sum/difference
      // proved within [-1,1] has at most 24 significant integer bits (the
      // endpoints are exact powers of two), hence needs no F32 rounding.
      // This is a domain proof for half operands, never for products of them.
      if((node[0]==='add'||node[0]==='sub')&&halfOperand(a)&&halfOperand(b)){
        const ar=range(a),br=range(b);
        if(ar&&br&&Math.max(Math.abs(ar.minimum),Math.abs(ar.maximum))+
          Math.max(Math.abs(br.minimum),Math.abs(br.maximum))<=1)return expression;
      }
      const rounded=lowerJsonRoundNormalF32AsF64(expression);
      f32Sources.set(rounded,expression);return rounded;
    }
    return o(node[0],node[1],...args.map(visit));
  }
  return {lower:visit};
}
