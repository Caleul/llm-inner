from pathlib import Path
import hashlib,json,sys,time,resource
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
start=time.monotonic();report={'coordinateCompositionComplete':False,'finalArtifactEmitted':False,'finalParity':False,'stateRestored':False,'position':0,'dimension':2,'expressionFormat':'Mathematical string','producerGrowth':[]}
try:
 with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8388608)) as model:
  with streaming_literals(model) as registry:
   def record(current):
    name=list(current.memo)[-1];text=current.memo[name]
    report['producerGrowth'].append({'producer':name,'storedCharacters':len(text),'expandedCharacters':registry.size(text),'definitions':len(registry.definitions)})
   expression=model.coordinate(2)
   report['producerGrowth']=[{'producer':name,'storedCharacters':stored,'expandedCharacters':expanded,'definitions':definitions} for name,stored,expanded,definitions in registry.events]
   report.update(coordinateCompositionComplete=True,completedProducers=len(model.memo),compilerDefinitions=len(registry.definitions),storedCharacters=sum(map(len,registry.definitions)),logicalCharacters=registry.size(expression),compositionSeconds=time.monotonic()-start,arithmeticEliminated=model.conversions.arithmetic_eliminated,rmsConstantComponents=model.rms_constant_components)
   plan=CoherentPaths(registry,max_paths=4)
   try:
    report['artifact']=plan.write('artifacts/direct-sympy-input-partitions/partial-fractions-full-coordinate.expr',expression,max_characters=1048576)
    report['finalArtifactEmitted']=True
   except ValueError as e:report['stop']=str(e)
   report['paths']=plan.stats
except BaseException as e:report['stop']=str(e);raise
finally:
 report.update(seconds=time.monotonic()-start,peakRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sourceSHA256={name:hashlib.sha256((Path('helpers')/name).read_bytes()).hexdigest() for name in ('direct_sympy_checkpoint.py','direct_sympy_strings.py','direct_sympy_conversions.py','direct_sympy_arithmetic.py','direct_sympy_sqrt.py','direct_sympy_coherent_paths.py','direct_sympy_streaming_literals.py')})
 Path('docs/evidence/direct-sympy-partial-fractions/coordinate-run.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report),flush=True)
