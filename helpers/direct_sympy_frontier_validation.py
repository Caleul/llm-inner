"""Evaluate the actual saved producer strings; NOT final coordinate parity.

Exact earlier literal subtrees may be reused solely by this verifier.
Source hashes, expression hashes and absence of runtime aliases are checked.
"""
import ast,hashlib,json,struct,sys,time,re
from unittest.mock import patch
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parent))
if len(sys.argv)!=4:raise ValueError('Usage: frontier-validation state checkpoint inputs')
from direct_sympy_strings import StringCompiler,syntax
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_savepoints import canonical,digest_file,ProducerSavepoints
original_search=re.search
def fast_search(pattern,text,*args,**kwargs):
 if pattern==r'\bCASStableRegion[0-9]+\b' and 'CASStableRegion' not in text:return None
 return original_search(pattern,text,*args,**kwargs)
for text in ('Bits64(X1)','CASStableRegion1()','otherCASStableRegion1()'):
 assert bool(fast_search(r'\bCASStableRegion[0-9]+\b',text))==bool(original_search(r'\bCASStableRegion[0-9]+\b',text))
started=time.monotonic();state=Path(sys.argv[1]);envelope=json.loads((state/'frontier.json').read_text());manifest=envelope['payload']
assert hashlib.sha256(canonical(manifest)).hexdigest()==envelope['integrity'],'Manifest integrity mismatch'
builder=CheckpointStrings(sys.argv[2],StringCompiler(max_characters=max([1048576,*[r['characters'] for r in manifest['records']]])));compiler=builder.compiler
with ProducerSavepoints(state,builder,manifest['identity']['dimension']) as store:
 assert manifest['identity']==store.identity,'Incompatible saved semantics, checkpoint, domain or reference platform'
class LazyBranches(ast.NodeTransformer):
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if node.func.id!='Piecewise':return node
  result=ast.Constant(value=0)
  for pair in reversed(node.args):result=ast.IfExp(test=pair.elts[1],body=pair.elts[0],orelse=result)
  return result
programs=[]
for record in manifest['records']:
 path=state/'objects'/(record['digest']+'.expr');assert digest_file(path)==record['digest'];text=path.read_text();assert len(text)==record['characters']
 with patch('direct_sympy_strings.re.search',side_effect=fast_search):
  compact,regions=compiler.compact_regions(text,compiler.context(builder.domains),validate_context=False)
 assert not any(x in text for x in ('CASBoundary','CASNumericRegion','CASStableRegion','R16(','R32(','Silu16(','sqrt('))
 node=syntax(compact);program=compile(ast.fix_missing_locations(ast.Expression(LazyBranches().visit(syntax(compact)))),str(path),'eval')
 programs.append((record['name'],text,program,regions))
 with patch('direct_sympy_strings.re.search',side_effect=fast_search):
  compiler.register_completed_region(text,builder.domains,node,word_closed=True)
 print(json.dumps({'producer':record['name'],'characters':len(text),'validationEnvelopeCharacters':len(compact),'sha256':record['digest']}),flush=True)
MASK=2**64-1
functions={'F64FromU64':lambda x:float(x),'U64FromF64':lambda x:int(x),'Bits64':lambda x:struct.unpack('Q',struct.pack('d',float(x)))[0],'Float64':lambda x:struct.unpack('d',struct.pack('Q',x))[0],'U64Add':lambda a,b:(a+b)&MASK,'U64Mul':lambda a,b:(a*b)&MASK,'U64And':lambda a,b:a&b,'U64Or':lambda a,b:a|b,'U64Shr':lambda a,b:a>>b,'And':lambda *x:all(x),'Or':lambda *x:any(x),'Not':lambda x:not x}
torch.set_num_threads(1);model=AutoModelForCausalLM.from_pretrained(sys.argv[2],dtype=torch.float16,attn_implementation='eager').eval();observed={}
handles=[]
for label,module in [('post',model.model.layers[0].post_attention_layernorm),('gate',model.model.layers[0].mlp.gate_proj),('up',model.model.layers[0].mlp.up_proj),('activation',model.model.layers[0].mlp.act_fn)]:
 handles.append(module.register_forward_hook(lambda module,args,out,label=label:observed.__setitem__(label,out.detach().clone())))
cases=json.loads(Path(sys.argv[3]).read_text())['cases'];comparisons=0
for row in cases:
 matrix=torch.tensor([[struct.unpack('e',struct.pack('H',v))[0] for v in values] for values in row['inputBits']],dtype=torch.float16);observed.clear()
 with torch.inference_mode():model(inputs_embeds=matrix.unsqueeze(0),use_cache=False)
 fundamental={f'X{i+1}':float(x) for i,x in enumerate(matrix[0])};values={};named={}
 for name,text,program,regions in programs:
  actual=eval(program,{'__builtins__':{},**functions},{**fundamental,**{token:values[source] for token,source in regions.items()}});values[text]=actual;named[name]=actual
 for i in range(2):
  expected=float(observed['post'][0,0,i]);actual=named[f'model.layers.0.post:{i}'];assert struct.pack('d',actual)==struct.pack('d',expected),(row['label'],i,actual,expected);comparisons+=1
 for label in ('gate','activation','up'):
  name=f'model.layers.0.{label}:0'
  if name in named:
   expected=float(observed[label][0,0,0]);actual=named[name];assert struct.pack('d',actual)==struct.pack('d',expected),(row['label'],label,actual,expected);comparisons+=1
for handle in handles:handle.remove()
builder.__exit__(None,None,None)
print(json.dumps({'cases':len(cases),'comparisons':comparisons,'mismatches':0,'producerFilesEvaluated':len(programs),'scope':'Full saved posterior normalization and completed gate/activation/up expressions, position 0; verification-only reuse of exact immutable earlier subtrees, not the final artifact or model coordinate parity','elapsedSeconds':time.monotonic()-started}))
