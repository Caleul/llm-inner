import {jsonInput,jsonConstant,jsonOperation,type JsonExpression} from './direct-json-expression.js';
import {simplifyJsonFixedPoint} from './direct-json-simplify.js';

/** Compile only each supported length's last position. Lazy conditions retain
 * all its causal attention dependencies without evaluating absent tokens.
 * Producer interfaces disappear: the result contains literal expressions. */
export async function composeJsonNextToken(context:number,lengthInput:string,
  buildLastPosition:(position:number)=>Promise<JsonExpression>):Promise<JsonExpression> {
  if(!Number.isSafeInteger(context)||context<1)throw new RangeError('Invalid next-token context');
  const length=jsonInput('u32',lengthInput);
  let result:JsonExpression|undefined;
  for(let position=context-1;position>=0;position--){
    const current=await buildLastPosition(position);
    result=result?jsonOperation('if',current[1],jsonOperation('eq','bool',length,jsonConstant('u32',position+1)),current,result):current;
    result=simplifyJsonFixedPoint(result).expression;
  }
  return result!;
}
