"""Inspect selected RMS source shapes without claiming an inverse proof."""
import re,sys
sys.path.insert(0,str(__import__('pathlib').Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=32*1024**2))as m:
 with streaming_literals(m)as r:
  expression=m.coordinate(2);plan=CoherentPaths(r);arm=next(plan.arms(expression))
  print('First path decisions:',len(arm.guards))
  for key,vector in m.norm_vectors.items():
   print('RMS:',key,'sources:',vector.get('sources'))
   for source in vector.get('sources',()):
    match=re.fullmatch(r'CompileValue([0-9]+)\(\)',source)
    if match:
     i=int(match[1]);print('Selected source',i,':',arm.view.definitions[i])
  for i,text in enumerate(arm.view.definitions):
   if i<14:print('Selected producer',i,':',text)
