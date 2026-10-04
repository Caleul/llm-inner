from pathlib import Path
import json,sys
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_partition_run import _compile_region,decode
from direct_sympy_coherent_paths import CoherentPaths
root=Path(__file__).resolve().parent
original=CoherentPaths.write
stats={}
def observed(self,*args,**kwargs):
 try:return original(self,*args,**kwargs)
 finally:stats.update(self.stats)
with patch.object(CoherentPaths,'write',observed):
 report=_compile_region('docs/evidence/direct-sympy-test-checkpoint',2,decode({'X1':[-23807,-15872],'X2':[-23807,-15872]}),root/'coordinate.expr',max_characters=1048576,cas_characters=8388608,max_paths=64)
report['paths']=stats
if report['complete'] or 'budget' not in report.get('stop','').lower() or not stats['contradictions']:raise ValueError('Original numeric contradiction not eliminated')
(root/'region-run.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
