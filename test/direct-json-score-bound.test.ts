import test from 'node:test';
import assert from 'node:assert/strict';
import {proveJsonHeadScoreBounds,type JsonHeadScoreGeometry} from '../src/direct-json-score-bound.js';
import {fixedF16RopeLiteral} from '../src/fixed-f16-rope-branches.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
const geometry=(heads=1,kvHeads=1):JsonHeadScoreGeometry=>({width:2,context:8,heads,kvHeads,headDim:2,
  ropeTheta:10000,scaling:Math.SQRT1_2,normalization:{weight:'gamma',epsilon:1e-6},
  q:{weight:'q',shape:[2*heads,2]},k:{weight:'k',shape:[2*kvHeads,2]}});
test('JSON correlated score proof reads weights progressively and covers every paired RoPE coefficient',async()=>{
  const g=geometry(2,1);let reads=0;
  const weight=async(name:string,index:number)=>{reads++;return name==='gamma'?1:
    name==='q'&&index>=4?1/512:1/256;};
  const proof=await proveJsonHeadScoreBounds(g,weight);
  assert.ok(proof);assert.equal(proof.scoreBounds.length,2);assert.equal(proof.ropePoints,8);
  assert.ok(proof.scoreBounds[1]!<proof.scoreBounds[0]!);assert.ok(reads>=2+8+4);
  assert.ok(Object.isFrozen(proof));assert.ok(Object.isFrozen(proof.scoreBounds));
  for(let row=0;row<g.context;row++){
    const c=decodeIeeeF16ToF32(fixedF16RopeLiteral(row,0,2,g.ropeTheta,0));
    const s=decodeIeeeF16ToF32(fixedF16RopeLiteral(row,0,2,g.ropeTheta,1));
    assert.ok(Math.sqrt(c*c+s*s)<=proof.rotationNorm);
    // The ideal rotation preserves the pair correlation; independent maxima
    // would spuriously allow both input coordinates at their L2 budget.
    for(const [x,y] of [[1,0],[0,1],[.6,.8],[-.6,.8],[-1,0]])
      assert.ok(Math.hypot(c*x!-s*y!,s*x!+c*y!)<=proof.rotationNorm*Math.hypot(x!,y!));
  }
});
test('JSON correlated score proof refuses unsupported geometry, unavailable coefficient budgets and nonfinite projections',async()=>{
  const g=geometry(),weight=async(name:string,_index:number)=>name==='gamma'?1:1/256;
  assert.equal(await proveJsonHeadScoreBounds(g,weight,1),undefined);
  assert.equal(await proveJsonHeadScoreBounds({...g,headDim:3},weight),undefined);
  assert.equal(await proveJsonHeadScoreBounds({...g,normalization:{weight:'gamma',epsilon:0}},weight),undefined);
  assert.equal(await proveJsonHeadScoreBounds(g,async()=>Infinity),undefined);
  assert.equal(await proveJsonHeadScoreBounds(g,async()=>65504),undefined);
  assert.equal(await proveJsonHeadScoreBounds({...g,scaling:65504},async()=>1),undefined);
  await assert.rejects(proveJsonHeadScoreBounds(g,weight,0),/budget/);
});
