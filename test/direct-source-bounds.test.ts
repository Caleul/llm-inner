import assert from "node:assert/strict";
import { test } from "node:test";
import { DirectSourceBounds } from "../src/direct-source-bounds.js";

function fold(source:string,values:Record<string,number>={}){
  const parser=new DirectSourceBounds(name=>{
    const value=values[name];if(value===undefined)throw new Error(`Unknown ${name}`);
    return {minimum:value,maximum:value,key:name};
  });
  for(const char of source)parser.accept(char);
  return parser.finish();
}
test("streaming scalar reduction preserves binary64 grouping and signed zeros",()=>{
  const cases:[string,number][]=[
    ["(10000000000000000.0+1.0)-10000000000000000.0",0],
    ["10000000000000000.0+(1.0-10000000000000000.0)",0],
    ["(4503599627370496.0+1.5)-4503599627370496.0",2],
    ["-0.0_f64",-0],["(-0.0_f64)*1.0_f64",-0],["0.0+(-0.0)",0],
    ["((1.0e-20_f64/2.0e-10_f64)*-3.0_f64)",(1e-20/2e-10)*-3],
  ];
  for(const [source,expected] of cases){const value=fold(source);
    assert.ok(Object.is(value.minimum,expected),source);assert.ok(Object.is(value.maximum,expected),source);}
});
test("matrix reads and interval correlations are reduced without storing expressions",()=>{
  assert.equal(fold("input_tokens[12][5]+input_tokens[2][8]",{"input_tokens[12][5]":3,"input_tokens[2][8]":7}).minimum,10);
  const parser=new DirectSourceBounds(name=>({minimum:-2,maximum:3,key:name}));
  parser.accept("input_tokens[0][0]*input_tokens[0][0]");
  const square=parser.finish();assert.equal(square.minimum,0);assert.equal(square.maximum,9);
});
