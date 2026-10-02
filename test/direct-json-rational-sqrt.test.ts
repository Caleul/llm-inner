import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,writeFile,rm} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {jsonInput,type JsonExpression} from '../src/direct-json-expression.js';
import {lowerJsonRationalPositiveNormalSqrtAsF64} from '../src/direct-json-rational-sqrt.js';
import {lowerJsonPositiveNormalSqrtAsF64} from '../src/direct-json-sqrt.js';
import {evaluateJsonExpression as evaluate} from '../src/direct-json-evaluator.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {auditJsonExpression} from '../src/direct-json-stream.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';

// Test oracle only: compile the actual literal JSON arithmetic into C++ with
// IEEE contraction disabled. No handwritten second implementation of the seed.
function cpp(node:JsonExpression):string {
  if(node[0]==='input')return node[2];
  if(node[0]==='constant')return node[1]==='u64'?`uint64_t{${node[2]}ULL}`:
    node[1]==='f64'?`word<double>(uint64_t{${node[2]}ULL})`:(()=>{throw new Error('Unexpected certificate constant');})();
  const args=node.slice(2) as JsonExpression[];
  if(node[0]==='reinterpret')return `word<${node[1]==='u64'?'uint64_t':'double'}>(${cpp(args[0]!)})`;
  const ops:Record<string,string>={add:'+',sub:'-',mul:'*',div:'/',and:'&',or:'|',xor:'^',shl:'<<',shr:'>>'};
  if(!ops[node[0]])throw new Error(`Unexpected certificate operation ${node[0]}`);
  return `(${cpp(args[0]!)} ${ops[node[0]]} ${cpp(args[1]!)})`;
}

test('rational root literal JSON exhausts both normalized F32 exponent parities against native IEEE sqrt',async()=>{
  const root=lowerJsonRationalPositiveNormalSqrtAsF64(jsonInput('f64','X1'));
  const measure=measureJsonExpression(root),previous=measureJsonExpression(lowerJsonPositiveNormalSqrtAsF64(jsonInput('f64','X1')));
  assert.equal(measure.inputReferences,26n);assert.equal(measure.uniqueDecisions,0);
  assert.ok(measure.inputReferences<previous.inputReferences);
  auditJsonExpression(root,{X1:{dtype:'f64',tokenPosition:0,coordinate:0}});
  const dir=await mkdtemp(join(tmpdir(),'json-rational-root-')),run=promisify(execFile);
  try{
    const source=join(dir,'proof.cpp'),binary=join(dir,'proof');
    await writeFile(source,`#include <cmath>
#include <cfenv>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
template<class T,class U> T word(U x){static_assert(sizeof(T)==sizeof(U));T y;std::memcpy(&y,&x,sizeof y);return y;}
double candidate(double X1){return ${cpp(root)};}
int main(){
 static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
 static_assert(FLT_EVAL_METHOD==0);
 if(std::fegetround()!=FE_TONEAREST)return 2;
 uint64_t tested=0,mismatches=0;
 for(uint32_t e=127;e<=128;e++)for(uint32_t f=0;f<0x800000;f++){
  float x=word<float>((e<<23)|f);double got=candidate(x),expected=std::sqrt(x);
  tested++;if(word<uint64_t>(got)!=word<uint64_t>(expected))mismatches++;
 }
 std::printf("%llu %llu\\n",(unsigned long long)tested,(unsigned long long)mismatches);
 return mismatches?1:0;
}`);
    await run('clang++',['-O3','-ffp-contract=off','-std=c++17',source,'-o',binary]);
    const {stdout}=await run(binary,[],{timeout:120_000});
    assert.equal(stdout.trim(),'16777216 0');
  }finally{await rm(dir,{recursive:true,force:true});}
});

test('rational root JSON preserves exact widening across normal exponent boundaries and all positive finite half values',()=>{
  const root=lowerJsonRationalPositiveNormalSqrtAsF64(jsonInput('f64','X1'));
  const check=(X1:number)=>assert.equal(evaluate(root,{X1},{allowPendingPrimitives:false}),Math.fround(Math.sqrt(X1)));
  for(let bits=1;bits<0x7c00;bits++)check(decodeIeeeF16ToF32(bits));
  const word=new DataView(new ArrayBuffer(4));
  for(let e=1;e<255;e++)for(const f of [0,1,2,3,0x3fffff,0x7ffffb,0x7ffffd,0x7fffff]){
    word.setUint32(0,(e<<23)|f);check(word.getFloat32(0));
  }
  let state=0x12345678;
  for(let i=0;i<20000;i++){
    state=(Math.imul(state,1664525)+1013904223)>>>0;
    word.setUint32(0,((state%254+1)<<23)|(state&0x7fffff));check(word.getFloat32(0));
  }
  assert.throws(()=>lowerJsonRationalPositiveNormalSqrtAsF64(jsonInput('f32','X1')),TypeError);
});
