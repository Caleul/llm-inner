import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {jsonInput as input,jsonConstant as c,jsonOperation as o} from '../src/direct-json-expression.js';
import {createJsonModelLowerer} from '../src/direct-json-lower-model.js';
import {simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {simplifyJsonBitPrecision} from '../src/direct-json-precision.js';
import {evaluateJsonExpression} from '../src/direct-json-evaluator.js';
import {writeJsonScalarUnits,type JsonScalarHeader} from '../src/direct-json-stream.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';

test('unchanged structural and precision fixed points retain completed producer identity',()=>{
  const x=input('f64','X1'),tree=o('mul','f64',x,c('f64',3));
  assert.equal(simplifyJsonFixedPoint(tree).expression,tree);
  assert.equal(simplifyJsonBitPrecision(tree),tree);
});

test('incremental substitution stabilizes children before consumers and removes equivalent producer subtraction',()=>{
  const facts={halfSources:new WeakMap(),positiveNormalRoots:new WeakSet(),exponentialBounds:new WeakMap(),activationBounds:new WeakMap()};
  const events:string[]=[],lowerer=createJsonModelLowerer(facts,new WeakMap(),{
    incremental:true,onSubstitution:event=>events.push(event.operation)});
  const x=input('f16','X1'),a=o('widen','f32',x),b=o('widen','f32',x);
  const result=lowerer.lower(o('sub','f32',a,b));
  assert.deepEqual(result,c('f64',0));
  assert.deepEqual(events,['input','widen','widen','sub']);
  const count=events.length;assert.equal(lowerer.lower(a),lowerer.lower(b));assert.equal(events.length,count);
  for(let bits=0;bits<65536;bits++)if((bits&0x7c00)!==0x7c00)
    assert.ok(Object.is(evaluateJsonExpression(result,{X1:decodeIeeeF16ToF32(bits)},{allowPendingPrimitives:false}),0));
});

test('a selected coordinate is emitted as a complete scalar artifact without pretending to cover the vector',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-json-coordinate-')),path=join(dir,'coordinate.jsonl');
  const header:JsonScalarHeader={schema:'direct-scalar-json-v1',inputWidth:1,context:2,outputWidth:3,
    inputs:{X1:{dtype:'f64',inputDtype:'f16',tokenPosition:0,coordinate:0}}};
  try{
    async function* units(){yield {position:1,dimension:2,expression:input('f64','X1')};}
    const result=await writeJsonScalarUnits(path,header,units(),{coordinate:{position:1,dimension:2}});
    assert.equal(result.units,1);
    const records=(await readFile(path,'utf8')).trim().split('\n').map(row=>JSON.parse(row));
    assert.equal(records[0].kind,'coordinate-header');assert.equal(records[2].kind,'coordinate-end');
    assert.equal(records[2].completeVector,false);assert.equal(records[2].finalParity,false);
    assert.deepEqual(records[1].expression,input('f64','X1'));
    await assert.rejects(writeJsonScalarUnits(join(dir,'bad.jsonl'),header,units(),{
      coordinate:{position:0,dimension:0}}),/unordered JSON scalar coordinate/);
  }finally{await rm(dir,{recursive:true,force:true});}
});

const python=process.env.LLM_INNER_DIRECT_PYTHON,checkpoint=process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT;
test('one complete source-discovered coordinate preserves live PyTorch bits after each dependency is stabilized',
  {skip:!python||!checkpoint},async()=>{
  const {openJsonModelBuilder}=await import('../src/direct-json-model.js');
  const {simplifyJsonSharedConditionsParallel}=await import('../src/direct-json-cofactor-parallel.js');
  const {execFile}=await import('node:child_process');
  const {promisify}=await import('node:util');
  const dir=await mkdtemp(join(tmpdir(),'direct-json-incremental-parity-'));
  let builder:Awaited<ReturnType<typeof openJsonModelBuilder>>|undefined;
  try{
    const reference=join(dir,'reference.json');
    await promisify(execFile)(python!,[new URL('../../helpers/capture_direct_json_reference.py',import.meta.url).pathname,
      checkpoint!,reference],{maxBuffer:1024*1024});
    const corpus=JSON.parse(await readFile(reference,'utf8')) as {width:number;vocab:number;
      cases:{inputBits:number[][];logitF64Bits:string[][]}[]};
    builder=await openJsonModelBuilder(checkpoint!,python!);
    const position=0,dimension=Math.min(2,corpus.vocab-1),precision=new WeakMap();
    let lowering:ReturnType<typeof createJsonModelLowerer>|undefined,dependencies=0;const dependencyKeys:string[]=[];
    const {expression,stats}=await builder.build(position,dimension,{onDependency:(key,node,facts)=>{
      dependencyKeys.push(key);
      lowering??=createJsonModelLowerer(facts,precision,{incremental:true});
      lowering.lower(node);dependencies++;
    }});
    assert.equal(dependencies,stats.dependencies+1);
    assert.ok(dependencyKeys.every(key=>!key.startsWith('score:')&&!/^projection:\d+:[qk]:/.test(key)));
    assert.equal(stats.exponentials,0);
    const {expression:closed}=await simplifyJsonSharedConditionsParallel(lowering!.lower(expression));
    const word=new DataView(new ArrayBuffer(8));let compared=0;
    for(const row of corpus.cases){
      const values:Record<string,number|bigint>={N:BigInt(row.inputBits.length)};
      for(let i=0;i<row.inputBits.length;i++)for(let j=0;j<corpus.width;j++)
        values[`X${i*corpus.width+j+1}`]=decodeIeeeF16ToF32(row.inputBits[i]![j]!);
      word.setFloat64(0,Number(evaluateJsonExpression(closed,values,{allowPendingPrimitives:false})));
      assert.equal('0x'+word.getBigUint64(0).toString(16).padStart(16,'0'),row.logitF64Bits[position]![dimension]);compared++;
    }
    assert.ok(compared>1);
    console.log(`Structural coordinate parity: position=${position} dimension=${dimension} cases=${compared} tokenLengths=${[...new Set(corpus.cases.map(row=>row.inputBits.length))].sort((a,b)=>a-b).join(',')} mismatches=0 artifactEmitted=false`);
  }finally{await builder?.close();await rm(dir,{recursive:true,force:true});}
});
