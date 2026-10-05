"""Compare fresh old/new builders over the same coordinate and input domain."""
import hashlib,importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_input_partitions import interval
from direct_sympy_streaming_literals import streaming_literals
from direct_sympy_coherent_paths import CoherentPaths
old=subprocess.check_output(['git','show','1c2bb36:helpers/direct_sympy_checkpoint.py'])
root=Path(__file__).resolve().parent;rows=[]
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'old_checkpoint.py';p.write_bytes(old)
 spec=importlib.util.spec_from_file_location('old_checkpoint',p)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for label,cls in [('previous',module.CheckpointStrings),('early',CheckpointStrings)]:
  start=time.monotonic()
  with cls('docs/evidence/direct-sympy-test-checkpoint',StringCompiler(max_characters=8388608),
      input_domains={'X1':interval(-2**-16,2**-16),'X2':interval(1,65504)}) as m:
   with streaming_literals(m) as registry:
    expr=m.coordinate(2);plan=CoherentPaths(registry,max_paths=128)
    row={'label':label,'producers':len(m.memo),'constantProjections':getattr(m,'constant_projections',0),
     'logicalCharacters':registry.size(expr),'growth':registry.events}
    try:row['artifact']=plan.write(root/(label+'.expr'),expr,max_characters=1048576)
    except ValueError as e:
     if 'budget' not in str(e).lower():raise
     row['stop']=str(e)
    row['seconds']=time.monotonic()-start;rows.append(row);print(json.dumps(row),flush=True)
result={'oldCheckpointSHA256':hashlib.sha256(old).hexdigest(),
 'newCheckpointSHA256':hashlib.sha256(Path('helpers/direct_sympy_checkpoint.py').read_bytes()).hexdigest(),
 'newBoundsSHA256':hashlib.sha256(Path('helpers/direct_sympy_layer_bounds.py').read_bytes()).hexdigest(),
 'domains':{'X1':[-2**-16,2**-16],'X2':[1,65504]},'results':rows}
(root/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
