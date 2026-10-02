import {jsonConstant as c,jsonOperation as o,type JsonExpression} from './direct-json-expression.js';
import {jsonConstantValue} from './direct-json-evaluator.js';
import {addDyadic,multiplyDyadic,type Dyadic} from './fixed-f16-projection.js';
import {sameJsonExpression} from './direct-json-simplify.js';
import {measureJsonExpression} from './direct-json-measure.js';

/** Compiler-only finite dyadic lattice. Never serialized into the artifact. */
export interface JsonAffineDomain {minimum:number;maximum:number;quantumExponent:number;excludesNegativeZero:boolean}
export type JsonAffineDomains=WeakMap<JsonExpression,JsonAffineDomain>;
export interface JsonAffineStats {visited:number;rewrites:number;barriers:number;exactRoot?:boolean}
function normalized(v:Dyadic):Dyadic {
  if(v.coefficient===0n)return {coefficient:0n,exponent:0};
  let {coefficient,exponent}=v;while((coefficient&1n)===0n){coefficient>>=1n;exponent++;}return {coefficient,exponent};
}
function dyadic(value:number):Dyadic {
  const word=new DataView(new ArrayBuffer(8));word.setFloat64(0,value);const bits=word.getBigUint64(0);
  const e=Number((bits>>52n)&2047n),fraction=bits&0xfffffffffffffn;
  return normalized({coefficient:(bits>>63n?-1n:1n)*(e===0?fraction:fraction|0x10000000000000n),exponent:e===0?-1074:e-1075});
}
function ceilUnits(value:number,exponent:number):bigint {
  const d=dyadic(Math.abs(value)),shift=d.exponent-exponent;
  return shift>=0?d.coefficient<<BigInt(shift):(d.coefficient+(1n<<BigInt(-shift))-1n)>>BigInt(-shift);
}
interface Term {atom:JsonExpression;coefficient:Dyadic}
interface Form {constant:Dyadic;terms:Term[]}
/** Visit every kind of node, including guards, casts and both reachable branches.
 * Collect/distribute only islands whose original and replacement F32/F64 operations
 * are exact on their certified lattices. Other operations remain explicit atoms.
 * A branch owns its refined bounds; proofs cannot leak to sibling branches. */
export function simplifyJsonAffine(root:JsonExpression,domains:JsonAffineDomains=new WeakMap(),
  stats:JsonAffineStats={visited:0,rewrites:0,barriers:0},rootOnly=false):JsonExpression {
  function session(overrides:Map<JsonExpression,JsonAffineDomain>) {
    const forms=new WeakMap<JsonExpression,Form|undefined>();
    const memo=new WeakMap<JsonExpression,JsonExpression>(),proofs=new WeakMap<JsonExpression,JsonAffineDomain|undefined>();
    function proof(node:JsonExpression):JsonAffineDomain|undefined {
      if(proofs.has(node))return proofs.get(node);
      let result=overrides.get(node)??domains.get(node);
      if(result&&(!Number.isFinite(result.minimum)||!Number.isFinite(result.maximum)||result.minimum>result.maximum||
        !Number.isSafeInteger(result.quantumExponent)||result.quantumExponent<-1074||result.quantumExponent>1023||typeof result.excludesNegativeZero!=='boolean'))
        throw new RangeError('Invalid certified affine domain');
      if(node[0]==='constant'&&(node[1]==='f64'||node[1]==='f32')){
        const x=Number(jsonConstantValue(node));if(Number.isFinite(x))result={minimum:x,maximum:x,quantumExponent:dyadic(x).exponent,excludesNegativeZero:!Object.is(x,-0)};
      }else if((node[1]==='f64'||node[1]==='f32')&&['add','sub','mul'].includes(node[0])){
        const a=proof((node[2] as JsonExpression)),b=proof((node[3] as JsonExpression));
        if(a&&b){
          const q=node[0]==='mul'?a.quantumExponent+b.quantumExponent:Math.min(a.quantumExponent,b.quantumExponent);
          const endpoints=node[0]==='add'?[a.minimum+b.minimum,a.maximum+b.maximum]:node[0]==='sub'?
            [a.minimum-b.maximum,a.maximum-b.minimum]:[a.minimum,a.maximum].flatMap(x=>[b.minimum,b.maximum].map(y=>x*y));
          // Bounds are rounded outward in lattice units, so F64 endpoint
          // arithmetic cannot understate a real-operation magnitude.
          const peak=Math.max(...endpoints.map(Math.abs));
          const bound=peak===0?0n:ceilUnits(peak,q)+1n;
          if(q>=(node[1]==='f32'?-149:-1074)&&q<=(node[1]==='f32'?127:1023)&&bound<=1n<<BigInt(node[1]==='f32'?24:53)){
            const maximum=Number(bound)*2**q;
            if(Number.isFinite(maximum)&&maximum<=(node[1]==='f32'?3.4028234663852886e38:Number.MAX_VALUE))result={minimum:-maximum,maximum,quantumExponent:q,
              excludesNegativeZero:result?.excludesNegativeZero??(node[0]==='add'?
                a.excludesNegativeZero||b.excludesNegativeZero:node[0]==='sub'?a.excludesNegativeZero:
                (a.minimum>0||a.maximum<0)&&(b.minimum>0||b.maximum<0))};
          }
        }
      }
      proofs.set(node,result);return result;
    }
    function form(node:JsonExpression):Form|undefined {
      if(forms.has(node))return forms.get(node);
      const result=calculateForm(node);forms.set(node,result);return result;
    }
    function calculateForm(node:JsonExpression):Form|undefined {
      if(node[1]!=='f64'&&node[1]!=='f32')return undefined;
      if(node[0]==='constant'){
        const x=Number(jsonConstantValue(node));return Number.isFinite(x)?{constant:dyadic(x),terms:[]}:undefined;
      }
      const d=proof(node);if(!d)return undefined;
      if(['add','sub','mul'].includes(node[0])&&exactOperation(node)){
        const a=form((node[2] as JsonExpression)),b=form((node[3] as JsonExpression));if(!a||!b)return undefined;
        if(node[0]==='mul'){
          const scalar=a.terms.length===0?a:b.terms.length===0?b:undefined;
          const expression=scalar===a?b:a;if(!scalar)return {constant:{coefficient:0n,exponent:0},terms:[{atom:node,coefficient:{coefficient:1n,exponent:0}}]};
          return {constant:normalized(multiplyDyadic(expression.constant,scalar.constant)),terms:expression.terms.map(t=>({atom:t.atom,coefficient:normalized(multiplyDyadic(t.coefficient,scalar.constant))}))};
        }
        const sign=node[0]==='sub'?-1n:1n,terms=a.terms.map(t=>({...t}));
        for(const term of b.terms){
          const coefficient={...term.coefficient,coefficient:term.coefficient.coefficient*sign};
          const found=terms.find(t=>sameJsonExpression(t.atom,term.atom));
          if(found)found.coefficient=normalized(addDyadic(found.coefficient,coefficient));else terms.push({atom:term.atom,coefficient});
        }
        return {constant:normalized(addDyadic(a.constant,{...b.constant,coefficient:b.constant.coefficient*sign})),terms:terms.filter(t=>t.coefficient.coefficient!==0n)};
      }
      return {constant:{coefficient:0n,exponent:0},terms:[{atom:node,coefficient:{coefficient:1n,exponent:0}}]};
    }
    function exactOperation(node:JsonExpression):boolean {
      const a=proof((node[2] as JsonExpression)),b=proof((node[3] as JsonExpression));if(!a||!b)return false;
      // An output-range certificate alone does not prove its operation exact.
      const q=node[0]==='mul'?a.quantumExponent+b.quantumExponent:Math.min(a.quantumExponent,b.quantumExponent);
      const maximum=node[0]==='mul'?Math.max(Math.abs(a.minimum),Math.abs(a.maximum))*Math.max(Math.abs(b.minimum),Math.abs(b.maximum)):
        Math.max(Math.abs(a.minimum),Math.abs(a.maximum))+Math.max(Math.abs(b.minimum),Math.abs(b.maximum));
      return q>=(node[1]==='f32'?-149:-1074)&&q<=(node[1]==='f32'?127:1023)&&Number.isFinite(maximum)&&maximum<=(node[1]==='f32'?3.4028234663852886e38:Number.MAX_VALUE)&&ceilUnits(maximum,q)+1n<=1n<<BigInt(node[1]==='f32'?24:53);
    }
    function emit(f:Form,type:'f32'|'f64'):JsonExpression|undefined {
      const asConstant=(d:Dyadic)=>{
        const value=Number(d.coefficient)*2**d.exponent;
        return Number.isFinite(value)&&normalized(addDyadic(d,{...dyadic(value),coefficient:-dyadic(value).coefficient})).coefficient===0n&&Object.is(Number(jsonConstantValue(c(type,value))),value)?c(type,value):undefined;
      };
      let result=asConstant(f.constant);if(!result)return undefined;
      for(const term of f.terms){
        const coefficient=asConstant(term.coefficient);if(!coefficient)return undefined;
        const product=term.coefficient.coefficient===1n&&term.coefficient.exponent===0?term.atom:o('mul',type,term.atom,coefficient);
        if(product[0]==='mul'&&!exactOperation(product))return undefined;
        const sum=o('add',type,result,product);if(!exactOperation(sum))return undefined;result=sum;
      }
      return result;
    }
    function refine(condition:JsonExpression,truth:boolean):Map<JsonExpression,JsonAffineDomain> {
      const next=overrides;if(!['lt','le','eq'].includes(condition[0]))return next;
      const x=condition[2] as JsonExpression,constant=condition[3] as JsonExpression;if(constant[0]!=='constant'||constant[1]!==x[1]||!constant[1].startsWith('f'))return next;
      const d=proof(x),value=Number(jsonConstantValue(constant));if(!d||!Number.isFinite(value))return next;
      let {minimum,maximum}=d;
      if(condition[0]==='eq'){if(!truth)return next;minimum=Math.max(minimum,value);maximum=Math.min(maximum,value);}
      else if(truth)maximum=Math.min(maximum,value);else minimum=Math.max(minimum,value);
      if(minimum>maximum||minimum===d.minimum&&maximum===d.maximum)return next;
      const refined=new Map(overrides);refined.set(x,{...d,minimum,maximum,excludesNegativeZero:d.excludesNegativeZero||minimum>0||maximum<0});
      return refined;
    }
    function visit(node:JsonExpression):JsonExpression {
      const hit=memo.get(node);if(hit)return hit;stats.visited++;
      if(node[0]==='constant'||node[0]==='input')return node;
      const args=node.slice(2) as JsonExpression[];let result:JsonExpression;
      if(node[0]==='if'){
        const condition=visit(args[0]!);
        if(condition[0]==='constant')result=visit(args[jsonConstantValue(condition)?1:2]!);
        else {const yesFacts=refine(condition,true),noFacts=refine(condition,false);
          const yes=(yesFacts===overrides?visit:session(yesFacts))(args[1]!),no=(noFacts===overrides?visit:session(noFacts))(args[2]!);
          result=condition===args[0]&&yes===args[1]&&no===args[2]?node:o('if',node[1],condition,yes,no);}
      }else{
        const children=rootOnly?args:args.map(visit);result=children.every((x,i)=>x===args[i])?node:o(node[0],node[1],...children);
        const inherited=domains.get(node);if(inherited&&result!==node)proofs.set(result,inherited);
        if((result[1]==='f64'||result[1]==='f32')&&['add','sub','mul'].includes(result[0])){
          if(node===root&&exactOperation(result))stats.exactRoot=true;
          const d=proof(result),f=d?.excludesNegativeZero&&exactOperation(result)?form(result):undefined;
          const candidate=f?emit(f,result[1] as 'f32'|'f64'):undefined;
          if(candidate&&measureJsonExpression(candidate).serializedBytes<measureJsonExpression(result).serializedBytes){result=candidate;stats.rewrites++;}
          else stats.barriers++;
        }
      }
      memo.set(node,result);return result;
    }
    return visit;
  }
  return session(new Map())(root);
}
