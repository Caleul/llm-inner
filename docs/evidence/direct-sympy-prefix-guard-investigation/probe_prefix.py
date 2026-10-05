"""Bounded inspection of selected dependencies and frozen guard prefixes."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
class Inspect(CoherentPaths):
 def literal(self,root,facts,domains):
  self.last_facts=facts
  return super().literal(root,facts,domains)
with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=96*1024**2))as m:
 with streaming_literals(m)as r:
  expression=m.coordinate(2);plan=Inspect(r,max_paths=128)
  for i,arm in enumerate(plan.arms(expression)):
   print('ARM',i,'BODY',arm.view.size(arm.expression),'GUARDS',sum(g.view.size(g.expression)for g in arm.guards),flush=True)
   if i in (0,5,7,8):
    print('BOUNDS',repr(plan.rms_bounds(plan.last_facts)),flush=True)
    for j,g in enumerate(arm.guards):print('GUARD',j,g.truth,g.expression[:350],g.view.size(g.expression),flush=True)
    for k in (1,2,8,11,12,14,27,29,31,34,35,37,39):
     print('PRODUCER',k,'RECIPE',r.definition_recipes.get(k,'?')[:500],'SELECTED',arm.view.definitions[k][:500],flush=True)
   if i>=8:break
