import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {openJsonModelBuilder} from '../src/direct-json-model.js';
import {evaluateJsonExpression,type JsonValue} from '../src/direct-json-evaluator.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {auditJsonExpression} from '../src/direct-json-stream.js';
import type {JsonExpression} from '../src/direct-json-expression.js';
import {lowerJsonModelExpression} from '../src/direct-json-lower-model.js';
import {measureJsonExpression} from '../src/direct-json-measure.js';
import {simplifyJsonBitPrecision,type JsonPrecisionFacts} from '../src/direct-json-precision.js';
import {simplifyJsonFixedPoint} from '../src/direct-json-simplify.js';
import {simplifyJsonSharedConditions} from '../src/direct-json-cofactor.js';
import {writeDirectJsonModel} from '../src/direct-json-compile.js';
const python=process.env.LLM_INNER_DIRECT_PYTHON,checkpoint=process.env.LLM_INNER_DIRECT_JSON_CHECKPOINT;

test('JSON compiler pipeline reports prepared coordinates without publishing an over-budget checkpoint vector',
  {skip:!python||!checkpoint},async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-json-compiler-')),path=join(dir,'model.jsonl');
  let prepared=0;
  try{
    await assert.rejects(writeDirectJsonModel(checkpoint!,python!,path,{maxBytes:4096,onPrepared:unit=>{
      prepared=unit.preparedUnits;assert.ok(unit.totalUnits>0);
      assert.ok(unit.measure.serializedBytes>4096n);
    }}),/before expression emission/);
    assert.equal(prepared,1);
    await assert.rejects(readFile(path),{code:'ENOENT'});
    const status=JSON.parse(await readFile(path+'.status.json','utf8'));
    assert.equal(status.units,0);assert.equal(status.finalParity,false);
    assert.equal(status.rejectedCoordinate.position,0);assert.equal(status.rejectedCoordinate.dimension,0);
  }finally{await rm(dir,{recursive:true,force:true});}
});

test('source-discovered JSON working expressions reproduce complete checkpoint logits before primitive lowering',
  {skip:!python||!checkpoint},async()=>{
  const dir=await mkdtemp(join(tmpdir(),'direct-json-model-'));
  const run=promisify(execFile);
  let builder:Awaited<ReturnType<typeof openJsonModelBuilder>>|undefined;
  try{
    const corpusPath=join(dir,'reference.json');
    await run(python!,[new URL('../../helpers/capture_direct_json_reference.py',import.meta.url).pathname,checkpoint!,corpusPath],{maxBuffer:1024*1024});
    const corpus=JSON.parse(await readFile(corpusPath,'utf8')) as {width:number;vocab:number;context:number;
      cases:{label:string;inputBits:number[][];logitF64Bits:string[][]}[]};
    builder=await openJsonModelBuilder(checkpoint!,python!);
    assert.equal(builder.header.inputWidth,corpus.width);assert.equal(builder.header.outputWidth,corpus.vocab);
    const expressions:JsonExpression[][]=[];
    const lowered:JsonExpression[][]=[];
    for(let row=0;row<corpus.context;row++){
      expressions[row]=[];
      lowered[row]=[];
      for(let dimension=0;dimension<corpus.vocab;dimension++){
        const {expression,stats,facts}=await builder.build(row,dimension);
        assert.ok(stats.dependencies>0);assert.ok(stats.squareRoots>0);assert.ok(stats.activations>0);
        assert.throws(()=>auditJsonExpression(expression,builder!.header.inputs),/Unlowered|Implicit/);
        expressions[row]!.push(expression);
        const precision:JsonPrecisionFacts=new WeakMap();
        const expanded=lowerJsonModelExpression(expression,facts,precision);
        const fixed=simplifyJsonFixedPoint(simplifyJsonBitPrecision(expanded,precision)).expression;
        const {expression:closed,stats:cofactor}=simplifyJsonSharedConditions(fixed);
        assert.ok(cofactor.afterBytes<=cofactor.beforeBytes);
        const measure=measureJsonExpression(closed);
        assert.ok(measure.uniqueNodes<100_000);lowered[row]!.push(closed);
      }
    }
    const bits=new DataView(new ArrayBuffer(8));
    for(const item of corpus.cases){
      const inputs:Record<string,JsonValue>={N:BigInt(item.inputBits.length)};
      for(let row=0;row<item.inputBits.length;row++)for(let column=0;column<corpus.width;column++)
        inputs[`X${row*corpus.width+column+1}`]=decodeIeeeF16ToF32(item.inputBits[row]![column]!);
      for(let row=0;row<item.inputBits.length;row++)for(let dimension=0;dimension<corpus.vocab;dimension++){
        const value=evaluateJsonExpression(expressions[row]![dimension]!,inputs);
        bits.setFloat64(0,Number(value));const actual='0x'+bits.getBigUint64(0).toString(16).padStart(16,'0');
        assert.equal(actual,item.logitF64Bits[row]![dimension],`${item.label}, n=${item.inputBits.length}, position=${row}, logit=${dimension}`);
        const closedValue=evaluateJsonExpression(lowered[row]![dimension]!,inputs,{allowPendingPrimitives:false});
        bits.setFloat64(0,Number(closedValue));const closedActual='0x'+bits.getBigUint64(0).toString(16).padStart(16,'0');
        assert.equal(closedActual,item.logitF64Bits[row]![dimension],`closed ${item.label}, n=${item.inputBits.length}, position=${row}, logit=${dimension}`);
      }
    }
  }finally{await builder?.close();await rm(dir,{recursive:true,force:true});}
});
