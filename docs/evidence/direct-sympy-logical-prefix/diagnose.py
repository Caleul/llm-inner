"""Measure selected dependency growth; never admit a partial expression."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_cover_regions import live_identity
root=Path(__file__).resolve().parent;checkpoint='docs/evidence/direct-sympy-test-checkpoint'
class Capture(CoherentPaths):
 def arms(self,expression):
  for arm in super().arms(expression):
   if 'firstArm' not in self.stats:
    self.stats['firstArm']={'bodyCharacters':arm.view.size(arm.expression),'guardCharacters':sum(g.view.size(g.expression)for g in arm.guards),'decisions':len(arm.guards)}
   yield arm
for mib in (32,96):
 identity=live_identity(checkpoint,2);started=time.monotonic();report={'artifactBudgetMiB':mib}
 with CheckpointStrings(checkpoint,StringCompiler(max_characters=mib*1024**2))as m:
  with streaming_literals(m)as r:
   expression=m.coordinate(2);p=Capture(r,max_paths=128)
   try:report['artifact']=p.write(root/(f'full-{mib}.expr'),expression,max_characters=mib*1024**2)
   except ValueError as error:
    if 'budget' not in str(error).lower():raise
    report['stop']=str(error)
   report['stats']=p.stats
 assert identity==live_identity(checkpoint,2);report.update(seconds=time.monotonic()-started,compilerIdentity=identity)
 (root/f'diagnostic-{mib}.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'budgetMiB':mib,'seconds':report['seconds'],'completedPaths':p.stats['completedPaths'],'firstArm':p.stats['firstArm'],'lastArm':{k:v for k,v in p.stats['lastArm'].items()if k!='dependencyGrowth'},'stop':report.get('stop')}))
