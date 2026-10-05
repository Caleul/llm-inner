import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {composeJsonNextToken} from '../src/direct-json-next-token.js';
import {jsonInput,jsonConstant as c,jsonOperation as o,type JsonExpression} from '../src/direct-json-expression.js';
import {evaluateJsonExpression} from '../src/direct-json-evaluator.js';
import {writeJsonScalarUnits,type JsonScalarHeader,type JsonScalarUnit} from '../src/direct-json-stream.js';
import {writeDirectJsonModel} from '../src/direct-json-compile.js';

test('Invalid compiler output scopes fail before opening any checkpoint',async()=>{
  await assert.rejects(writeDirectJsonModel('absent','absent','absent',{
    outputScope:'invalid' as 'next-token'}),/Unknown compiler output scope/);
});

test('Next-token literal vector contains four logits and reads only the selected causal prefix for lengths 1..8',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'next-token-json-')),path=join(dir,'next.jsonl');
  const header:JsonScalarHeader={schema:'direct-scalar-json-v1',context:8,inputWidth:1,outputWidth:4,
    outputSelection:'last-position',inputs:{N:{dtype:'u32',source:'inputLength'}}};
  for(let position=0;position<8;position++)header.inputs[`X${position+1}`]={dtype:'f64',inputDtype:'f16',tokenPosition:position,coordinate:0};
  const expressions:JsonExpression[]=[],requests:number[][]=[];
  try{
    for(let dimension=0;dimension<4;dimension++){
      requests[dimension]=[];
      expressions.push(await composeJsonNextToken(8,'N',async position=>{
        requests[dimension]!.push(position);
        // A small ordered causal contribution tests prefix preservation;
        // the model builder supplies the actual attention implementation.
        let result=c('f64',dimension);
        for(let row=0;row<=position;row++)result=o('add','f64',result,jsonInput('f64',`X${row+1}`));
        return result;
      }));
      assert.deepEqual(requests[dimension],[7,6,5,4,3,2,1,0]);
    }
    async function* units():AsyncGenerator<JsonScalarUnit>{
      for(let dimension=0;dimension<4;dimension++)yield {position:0,dimension,expression:expressions[dimension]!};
    }
    const result=await writeJsonScalarUnits(path,header,units());assert.equal(result.units,4);
    const lines=(await readFile(path,'utf8')).trim().split('\n').map(line=>JSON.parse(line));
    assert.equal(lines.length,6);assert.equal(lines[0].outputSelection,'last-position');
    const bits=new DataView(new ArrayBuffer(8)),word=(x:number)=>{bits.setFloat64(0,x);return bits.getBigUint64(0);};
    for(let length=1;length<=8;length++){
      const inputs:Record<string,number|bigint>={N:BigInt(length)};
      for(let row=0;row<length;row++)inputs[`X${row+1}`]=row%2?-(row+1):row===0?-0:row+1;
      for(let dimension=0;dimension<4;dimension++){
        let expected=dimension;for(let row=0;row<length;row++)expected+=Number(inputs[`X${row+1}`]);
        const actual=Number(evaluateJsonExpression(lines[dimension+1].expression,inputs,{allowPendingPrimitives:false}));
        assert.equal(word(actual),word(expected),`length=${length}, dimension=${dimension}`);
      }
    }
    // A matrix-shaped generator cannot be accidentally accepted as a vector.
    async function* matrix():AsyncGenerator<JsonScalarUnit>{for await(const unit of units())yield {...unit,position:1};}
    const previous=await readFile(path);
    await assert.rejects(writeJsonScalarUnits(path,header,matrix()),/unordered/);
    assert.deepEqual(await readFile(path),previous);
    async function* incomplete():AsyncGenerator<JsonScalarUnit>{yield {position:0,dimension:0,expression:expressions[0]!};}
    await assert.rejects(writeJsonScalarUnits(path,header,incomplete()),/Incomplete/);
    assert.deepEqual(await readFile(path),previous);
  }finally{await rm(dir,{recursive:true,force:true});}
});
