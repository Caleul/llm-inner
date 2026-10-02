import {fixedF16RopeLiteral} from './fixed-f16-rope-branches.js';
import {decodeIeeeF16ToF32} from './utils.js';
interface Projection {weight:string;shape:[number,number]}
export interface JsonHeadScoreGeometry {
  width:number;context:number;heads:number;kvHeads:number;headDim:number;ropeTheta:number;scaling:number;
  normalization:{weight:string;epsilon:number};q:Projection;k:Projection;
}
export interface JsonHeadScoreBoundCertificate {
  readonly normalizationNorm:number;readonly rotationNorm:number;
  readonly scoreBounds:readonly number[];readonly ropePoints:number;
}
/** Compiler proof over the already admitted finite-normalization domain.
 * The existing RMS certificate supplies ||normalized||2 <=
 * 1.25*sqrt(width)*max|gamma| + sqrt(width)*2^-23.
 * Cauchy/Frobenius bounds each projected head, rather than treating every
 * coordinate independently. The paired RoPE block [[c,-s],[s,c]] has norm
 * sqrt(c*c+s*s); the actual half coefficients are scanned, not assumed to be
 * a perfect real rotation. Half-product, F32-sum and final-half errors fit
 * within 1.01*rho*||head||2 + sqrt(headDim)*2^-23.
 * No weights, activation arrays or execution graph are retained.
 */
export async function proveJsonHeadScoreBounds(g:JsonHeadScoreGeometry,
  weight:(name:string,index:number)=>Promise<number>,maxRopePoints=100_000):Promise<JsonHeadScoreBoundCertificate|undefined>{
  if(!Number.isSafeInteger(maxRopePoints)||maxRopePoints<1)throw new RangeError('Invalid RoPE proof budget');
  const dimensions=[g.width,g.context,g.heads,g.kvHeads,g.headDim];
  if(dimensions.some(x=>!Number.isSafeInteger(x)||x<1||x>1_000_000)||g.headDim%2||g.heads%g.kvHeads||
    g.q.shape[0]!==g.heads*g.headDim||g.k.shape[0]!==g.kvHeads*g.headDim||
    g.q.shape[1]!==g.width||g.k.shape[1]!==g.width||!Number.isFinite(g.scaling)||
    !Number.isFinite(g.ropeTheta)||g.ropeTheta<1||!(Math.fround(g.normalization.epsilon)>=2**-126&&
    Number.isFinite(Math.fround(g.normalization.epsilon))))return undefined;
  const ropePoints=g.context*(g.headDim/2);
  if(ropePoints>maxRopePoints)return undefined;
  let gamma=0;
  for(let i=0;i<g.width;i++){
    const value=await weight(g.normalization.weight,i);if(!Number.isFinite(value))return undefined;
    gamma=Math.max(gamma,Math.abs(value));
  }
  const normalizationNorm=gamma===0?0:1.25*Math.sqrt(g.width)*gamma+Math.sqrt(g.width)*2**-23;
  if(!(normalizationNorm<65504))return undefined;
  let rotationNorm=0;
  for(let position=0;position<g.context;position++)for(let pair=0;pair<g.headDim/2;pair++){
    const c=decodeIeeeF16ToF32(fixedF16RopeLiteral(position,pair,g.headDim,g.ropeTheta,0));
    const s=decodeIeeeF16ToF32(fixedF16RopeLiteral(position,pair,g.headDim,g.ropeTheta,1));
    if(!Number.isFinite(c)||!Number.isFinite(s)||Math.abs(c)>1||Math.abs(s)>1)return undefined;
    // Half coefficients have magnitude <=1; their squared sum is exact F64.
    // Padding also makes the numerical square-root bound outward.
    rotationNorm=Math.max(rotationNorm,Math.sqrt(c*c+s*s)*1.001);
  }
  async function projectedHead(p:Projection,head:number):Promise<number>{
    let squared=0;
    for(let row=head*g.headDim;row<(head+1)*g.headDim;row++)for(let c=0;c<g.width;c++){
      const w=await weight(p.weight,row*g.width+c);if(!Number.isFinite(w))return Infinity;
      squared+=w*w;
    }
    // <=1e12 positive F64 terms lose <0.00012 relatively; 1.001 covers
    // that accumulation and sqrt error. The 1.125 factor covers the ordered
    // F32 dot and relative half rounding; the absolute half term is separate.
    return 1.125*normalizationNorm*Math.sqrt(squared)*1.001+Math.sqrt(g.headDim)*2**-24;
  }
  const scoreBounds:number[]=[];
  for(let head=0;head<g.heads;head++){
    const q=await projectedHead(g.q,head),k=await projectedHead(g.k,Math.floor(head/(g.heads/g.kvHeads)));
    if(!(q<65504&&k<65504))return undefined;
    const rotatedQ=1.01*rotationNorm*q+Math.sqrt(g.headDim)*2**-23;
    const rotatedK=1.01*rotationNorm*k+Math.sqrt(g.headDim)*2**-23;
    if(!(rotatedQ<65504&&rotatedK<65504))return undefined;
    const dot=1.125*rotatedQ*rotatedK+2**-24;
    if(!(dot<65504))return undefined;
    const bound=dot*Math.abs(Math.fround(g.scaling))*1.01+2**-24;
    if(!(bound<65504))return undefined;scoreBounds.push(bound);
  }
  return Object.freeze({normalizationNorm,rotationNorm,scoreBounds:Object.freeze(scoreBounds),ropePoints});
}
