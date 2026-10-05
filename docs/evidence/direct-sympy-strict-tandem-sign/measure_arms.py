"""Measure stabilized arms, never publish or count them as a final coordinate."""
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_coherent_paths import CoherentPaths,OrderedGuardPruner
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals
folder=Path(__file__).resolve().parent;start=time.monotonic();records=[]
with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=512*1024**2))as model:
 with streaming_literals(model)as registry:
  expression=model.coordinate(2);plan=CoherentPaths(registry,max_paths=128);pruner=OrderedGuardPruner(plan.stats)
  for index,arm in enumerate(plan.arms(expression),1):
   arm=pruner.prune(arm)
   # Guard skeleton identity tests the same ordered branch decisions; root
   # numerical payloads differ only under the independently tested certificate.
   skeleton=[(g.expression,g.truth)for g in arm.guards]
   records.append({'candidate':index,'bodyCharacters':arm.view.size(arm.expression),'guardCharacters':sum(g.view.size(g.expression)for g in arm.guards),'guardSkeletonSHA256':hashlib.sha256(json.dumps(skeleton,separators=(',',':')).encode()).hexdigest()})
   if index==27:break
  report={'seconds':time.monotonic()-start,'candidates':records,'stats':plan.stats,'completeCoordinate':False,'tandemSourceSHA256':hashlib.sha256(Path(sys.argv[1],'direct_sympy_tandem.py').read_bytes()).hexdigest()}
(folder/(sys.argv[2]+'-arms.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'label':sys.argv[2],'seconds':report['seconds'],'measuredCandidates':len(records)}))
