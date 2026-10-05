"""Capture compact selected recipes before allocating their full expressions."""
import json,re,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_cover_regions import live_identity
from direct_sympy_conversions import ConversionSession
import ast
original=ConversionSession.no_odd_f32_ties
observed=[]
def observe(self,node):
 result=original(self,node)
 if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add) and ('e-07' in ast.unparse(node) or 'e-06' in ast.unparse(node)):
  b=self.bounds(node);a=self.bounds(node.left)
  observed.append({'expression':ast.unparse(node),'bound':None if b is None else vars(b),'leftBound':None if a is None else vars(a),'leftKind':self.value_kind(node.left),'noOddTies':result})
 return result
ConversionSession.no_odd_f32_ties=observe
root=Path(__file__).resolve().parent;checkpoint='docs/evidence/direct-sympy-test-checkpoint'
identity=live_identity(checkpoint,2);start=time.monotonic()
with CheckpointStrings(checkpoint,StringCompiler(max_characters=96*1024**2))as m:
 with streaming_literals(m)as registry:
  expression=m.coordinate(2);paths=CoherentPaths(registry,max_paths=128)
  for i,arm in enumerate(paths.arms(expression)):
   size=arm.view.size(arm.expression)
   if size>96*1024**2:
    definition_keys={ast.dump(ast.parse(text,mode='eval').body):index for index,text in enumerate(registry.definitions)}
    matches=[]
    for g in arm.guards:
     for n in ast.walk(ast.parse(g.expression,mode='eval').body):
      if isinstance(n,ast.Call)and n.func.id=='Bits64' and len(n.args)==1:
       k=ast.dump(n.args[0]);matches.append({'guardTruth':g.truth,'matchedDefinition':definition_keys.get(k),'source':ast.unparse(n.args[0])[:500]})
    report={'literalGuardMatches':matches,'compilerIdentity':identity,'seconds':time.monotonic()-start,'candidate':i+1,'bodyCharacters':size,
     'expression':arm.expression,'definitions':[{'index':index,'text':text,'recipe':registry.definition_recipes.get(index),'expandedCharacters':arm.view.size(f'CompileValue{index}()'),
      'references':{key:len(re.findall(r'\b'+key+r'\(\)',text))for key in set(re.findall(r'CompileValue[0-9]+',text))}}
      for index,text in enumerate(arm.view.definitions)],
     'observedMeanConversions':observed,'guards':[{'expression':g.expression,'truth':g.truth,'expandedCharacters':g.view.size(g.expression)}for g in arm.guards],'stats':paths.stats}
    (root/'candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k]for k in ('candidate','seconds','bodyCharacters')}));break
assert identity==live_identity(checkpoint,2)
