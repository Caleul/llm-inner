import {createHash} from 'node:crypto';
import {mkdir,open,rename,writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';
import {withDirectCompilationLease} from './direct-compilation-lease.js';
import {validateJsonNode,type JsonDtype,type JsonExpression} from './direct-json-expression.js';

export type JsonInputBinding = {dtype:JsonDtype;tokenPosition:number;coordinate:number} |
  {dtype:'u32';source:'inputLength'};
export interface JsonScalarHeader {
  schema:'direct-scalar-json-v1';
  inputWidth:number;
  context:number;
  outputWidth:number;
  inputs:Record<string,JsonInputBinding>;
}
export interface JsonScalarUnit {position:number;dimension:number;expression:JsonExpression}
export interface JsonExpressionAudit {occurrences:number;uniqueNodes:number;decisions:number;uniqueDecisions:number;inputReferences:number}
/** Audit the fully substituted syntax before emission. Budget is for expanded
 * occurrences, not unique compiler nodes: sharing cannot conceal huge output. */
export function auditJsonExpression(root:JsonExpression,inputs:Record<string,JsonInputBinding>,maxOccurrences=1_000_000):JsonExpressionAudit {
  if(!Number.isSafeInteger(maxOccurrences)||maxOccurrences<1)throw new RangeError('Invalid expression budget');
  const result:JsonExpressionAudit={occurrences:0,uniqueNodes:0,decisions:0,uniqueDecisions:0,inputReferences:0};
  const seen=new WeakSet<object>(),active=new WeakSet<object>();
  const stack:{node:JsonExpression;leave:boolean;depth:number}[]=[{node:root,leave:false,depth:0}];
  while(stack.length){
    const {node,leave,depth}=stack.pop()!;
    if(leave){active.delete(node);continue;}
    validateJsonNode(node);
    if(active.has(node))throw new Error('Cyclic compilation expression');
    if(depth>512)throw new RangeError('Expression depth budget exceeded');
    if(++result.occurrences>maxOccurrences)throw new RangeError('Expanded expression budget exceeded');
    const unique=!seen.has(node);seen.add(node);if(unique)result.uniqueNodes++;
    if(node[0]==='input'){
      const binding=inputs[node[2]];
      if(!binding||binding.dtype!==node[1])throw new Error('Undeclared or mistyped fundamental input');
      result.inputReferences++;
    }else if(node[0]!=='constant'){
      if(node[1]==='f32'&&['add','sub','mul','div'].includes(node[0]))
        throw new Error('Implicit F32 arithmetic rounding must be lowered before emission');
      if(node[0]==='if'){result.decisions++;if(unique)result.uniqueDecisions++;}
      active.add(node);stack.push({node,leave:true,depth});
      for(const child of (node.slice(2) as JsonExpression[]).reverse())stack.push({node:child,leave:false,depth:depth+1});
    }
  }
  return result;
}
/** Writes primitive syntax iteratively, without rendering a whole expression as
 * a string. One scalar unit is resident; checkpoint-sized ASTs are not required. */
function* expressionTokens(root:JsonExpression):Generator<string> {
  const stack:(JsonExpression|string)[]=[root];
  while(stack.length){
    const next=stack.pop()!;
    if(typeof next==='string'){yield next;continue;}
    yield '['+JSON.stringify(next[0])+','+JSON.stringify(next[1])+',';
    if(next[0]==='constant'||next[0]==='input'){yield JSON.stringify(next[2])+']';continue;}
    stack.push(']');
    for(let i=next.length-1;i>=2;i--){stack.push(next[i] as JsonExpression);if(i>2)stack.push(',');}
  }
}
export async function writeJsonScalarUnits(path:string,header:JsonScalarHeader,units:AsyncIterable<JsonScalarUnit>,
  options:{maxBytes?:number;maxOccurrences?:number}={}):Promise<{units:number;bytes:number;sha256:string;decisions:number}> {
  for(const size of [header.inputWidth,header.context,header.outputWidth])if(!Number.isSafeInteger(size)||size<1)throw new RangeError('Invalid discovered JSON geometry');
  if(!Number.isSafeInteger(header.context*header.outputWidth))throw new RangeError('JSON coordinate count is not exactly representable');
  if(header.schema!=='direct-scalar-json-v1')throw new TypeError('Unknown JSON schema');
  for(const [name,b] of Object.entries(header.inputs)){
    if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(name))throw new TypeError('Invalid fundamental input name');
    if('source' in b){if(b.source!=='inputLength'||b.dtype!=='u32')throw new TypeError('Invalid structural input binding');continue;}
    if(b.dtype!=='f16'||!Number.isSafeInteger(b.tokenPosition)||
      b.tokenPosition<0||b.tokenPosition>=header.context||!Number.isSafeInteger(b.coordinate)||
      b.coordinate<0||b.coordinate>=header.inputWidth)throw new TypeError('Invalid embedding input binding');
  }
  const maxBytes=options.maxBytes??64*1024*1024;
  if(!Number.isSafeInteger(maxBytes)||maxBytes<1)throw new RangeError('Invalid JSON byte budget');
  return withDirectCompilationLease(path,async()=>{
    await mkdir(dirname(path),{recursive:true});
    const draft=path+'.draft',file=await open(draft,'w'),hash=createHash('sha256');
    let bytes=0,count=0,decisions=0,buffer='';
    const flush=async()=>{if(buffer){await file.writeFile(buffer);buffer='';}};
    const emit=async(token:string)=>{
      bytes+=Buffer.byteLength(token);
      if(bytes>maxBytes)throw new RangeError('JSON output budget exceeded');
      hash.update(token);buffer+=token;if(Buffer.byteLength(buffer)>=65536)await flush();
    };
    try{
      await emit(JSON.stringify({kind:'header',...header})+'\n');
      for await(const unit of units){
        const {position,dimension,expression}=unit;
        if(!Number.isSafeInteger(position)||position<0||position>=header.context||
          !Number.isSafeInteger(dimension)||dimension<0||dimension>=header.outputWidth)throw new RangeError('Invalid JSON scalar coordinate');
        // Canonical order proves coverage/uniqueness without retaining a set
        // proportional to checkpoint context multiplied by vocabulary size.
        if(position!==Math.floor(count/header.outputWidth)||dimension!==count%header.outputWidth)
          throw new Error('Missing, duplicate or unordered JSON scalar coordinate');
        const audit=auditJsonExpression(expression,header.inputs,options.maxOccurrences);
        decisions+=audit.decisions;
        await emit(`{"kind":"scalar","position":${position},"dimension":${dimension},"expression":`);
        for(const token of expressionTokens(expression))await emit(token);
        await emit(',"audit":'+JSON.stringify(audit)+'}\n');count++;
      }
      // A complete vector contains every dimension at every supported position.
      if(count!==header.context*header.outputWidth)throw new Error('Incomplete JSON output vector');
      await emit(JSON.stringify({kind:'end',units:count,decisions,finalParity:false})+'\n');
      await flush();await file.sync();await file.close();
      const result={units:count,bytes,sha256:hash.digest('hex'),decisions};
      await rename(draft,path);
      await writeFile(path+'.status.json',JSON.stringify({status:'emitted',...result,finalParity:false},null,2)+'\n');
      return result;
    }catch(error){
      await file.close().catch(()=>{});
      await writeFile(path+'.status.json',JSON.stringify({status:'pending',units:count,bytes,finalParity:false,
        reason:error instanceof Error?error.message:String(error)},null,2)+'\n');
      throw error;
    }
  });
}
