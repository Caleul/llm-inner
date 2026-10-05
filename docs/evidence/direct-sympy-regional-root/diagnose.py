import json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
import direct_sympy_sqrt as roots
from direct_sympy_strings import StringCompiler,syntax
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
folder=Path(__file__).resolve().parent;original=roots.expand;counts={}
def expand(source,session):
 b=session.bounds(syntax(source));first=math.frexp(b.minimum)[1]-1;last=math.frexp(b.maximum)[1]-1
 k=(first,last,math.ldexp(b.minimum,-first),math.ldexp(b.maximum,-first))
 counts[k]=counts.get(k,0)+1
 return original(source,session)
roots.expand=expand
start=time.monotonic()
with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=96*1024**2))as model:
 with streaming_literals(model)as registry:
  expression=model.coordinate(2);plan=CoherentPaths(registry,max_paths=128)
  try:plan.write('/private/tmp/llm-inner-regional-root-diagnostic.expr',expression,max_characters=96*1024**2)
  except ValueError as e:
   if 'budget'not in str(e).lower():raise
  report={'seconds':time.monotonic()-start,'stats':plan.stats,'rootEnclosures':[{'first':k[0],'last':k[1],'normalizedMinimum':k[2],'normalizedMaximum':k[3],'uses':v}for k,v in counts.items()]}
(folder/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'seconds':report['seconds'],'enclosures':len(counts),'singleBinadeUses':sum(v for k,v in counts.items()if k[0]==k[1])}))
