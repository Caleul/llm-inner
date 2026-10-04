"""Compare the preceding and tightened proof on the same input region."""
import hashlib,importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
import direct_sympy_layer_bounds as bounds
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_input_partitions import interval
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler

root=Path(__file__).resolve().parent
domains={'X1':interval(0.125,1),'X2':interval(-65504,-64)}
old_source=subprocess.check_output(['git','show','a8ecd24:helpers/direct_sympy_layer_bounds.py'])
rows=[]
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'old_bounds.py';path.write_bytes(old_source)
    spec=importlib.util.spec_from_file_location('old_bounds',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for name,implementation in [('previous',old.layer),('tightened',bounds.layer)]:
        started=time.monotonic()
        with patch.object(bounds,'layer',implementation):
            with CheckpointStrings('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8388608),input_domains=domains) as model:
                with streaming_literals(model) as registry:
                    expression=model.coordinate(2)
                    report={'mode':name,'producers':len(model.memo),'elidedUpdates':model.elided_updates,
                        'logicalCharacters':registry.size(expression),'storedCharacters':sum(map(len,registry.definitions)),
                        'producerGrowth':[{'producer':key,'storedCharacters':size,'expandedCharacters':expanded} for key,size,expanded,_ in registry.events]}
                    plan=CoherentPaths(registry,max_paths=128)
                    try:report['artifact']=plan.write(root/(name+'.expr'),expression,max_characters=1048576)
                    except ValueError as error:
                        if 'budget' not in str(error).lower():raise
                        report['stop']=str(error)
                    report['paths']=plan.stats
        report['seconds']=time.monotonic()-started;rows.append(report)
        print(json.dumps(report),flush=True)
if all('artifact' in row for row in rows) and rows[0]['artifact']['sha256']==rows[1]['artifact']['sha256']:
    (root/'previous.expr').unlink(missing_ok=True)
    rows[0]['artifact']['path']=rows[1]['artifact']['path']
(root/'diagnosis.json').write_text(json.dumps({'oldBoundsSHA256':hashlib.sha256(old_source).hexdigest(),
    'newBoundsSHA256':hashlib.sha256(Path('helpers/direct_sympy_layer_bounds.py').read_bytes()).hexdigest(),
    'domains':{'X1':[0.125,1],'X2':[-65504,-64]},'results':rows},indent=2)+'\n')
