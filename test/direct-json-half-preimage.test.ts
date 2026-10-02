import test from 'node:test';
import assert from 'node:assert/strict';
import {maximumF32MagnitudeForHalfBound} from '../src/direct-json-half-preimage.js';
import {decodeIeeeF16ToF32} from '../src/utils.js';
import {f32BitsToDyadic,roundDyadicToF16IfElse} from '../src/fixed-f16-projection.js';

test('backward half-output bounds certify every finite half cell with exact F32 tie ownership',()=>{
  const word=new DataView(new ArrayBuffer(4));
  for(let code=0;code<=0x7bff;code++){
    const value=decodeIeeeF16ToF32(code),maximum=maximumF32MagnitudeForHalfBound(value)!;
    word.setFloat32(0,maximum);const source=word.getUint32(0);
    for(const sign of [0,0x80000000]){
      assert.equal(roundDyadicToF16IfElse(f32BitsToDyadic((source|sign)>>>0)),code|(sign?0x8000:0));
      assert.equal(roundDyadicToF16IfElse(f32BitsToDyadic(((source+1)|sign)>>>0)),(code+1)|(sign?0x8000:0));
    }
    const next=code===0x7bff?65536:decodeIeeeF16ToF32(code+1);
    assert.equal(maximumF32MagnitudeForHalfBound((value+next)/2),maximum);
  }
  for(const bound of [-1,Infinity,NaN])assert.equal(maximumF32MagnitudeForHalfBound(bound),undefined);
});
