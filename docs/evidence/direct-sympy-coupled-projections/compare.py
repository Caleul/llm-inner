"""Emit the same region with the preceding and current path simplifiers."""
import hashlib,importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
import torch
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_coherent_paths import CoherentPaths
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_strings import StringCompiler
from direct_sympy_input_partitions import interval
from direct_sympy_region_parity import verify_region
from direct_sympy_cover_regions import live_identity

checkpoint='docs/evidence/direct-sympy-test-checkpoint'
directory=Path('artifacts/direct-sympy-input-partitions/central-guard-diagnosis')
old=subprocess.check_output(['git','show','abf5d72:helpers/direct_sympy_coherent_paths.py'])
domains={'X1':interval(.0625,2),'X2':interval(.0625,1)}
torch.set_num_threads(1);rows=[]
with tempfile.TemporaryDirectory() as temporary:
    source=Path(temporary)/'previous_paths.py';source.write_bytes(old)
    spec=importlib.util.spec_from_file_location('previous_paths',source)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    for label,cls in [('previous',module.CoherentPaths),('coupled',CoherentPaths)]:
        started=time.monotonic();path=directory/(label+'.expr')
        with CheckpointStrings(checkpoint,StringCompiler(max_characters=16777216),input_domains=domains) as model:
            with streaming_literals(model) as registry:
                expression=model.coordinate(2);plan=cls(registry,max_paths=128)
                artifact=plan.write(path,expression,max_characters=16777216)
                row={'label':label,'producers':len(model.memo),'logicalCharacters':registry.size(expression),
                    'artifact':artifact,'pathStatistics':plan.stats,'compilationSeconds':time.monotonic()-started}
        row['parity']=verify_region(checkpoint,2,path,domains,random_cases=8192)
        assert row['parity']['mismatches']==0
        rows.append(row);print(json.dumps(row),flush=True)
        (directory/'comparison.json').write_text(json.dumps({'previousCommit':'abf5d72',
            'previousPathCompilerSHA256':hashlib.sha256(old).hexdigest(),
            'currentIdentity':live_identity(checkpoint,2),'scope':'One central region, position 0, one token, coordinate 2',
            'results':rows},indent=2)+'\n')
