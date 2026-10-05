"""Attempt a complete finite-Half coordinate under explicit publication budgets."""
import json,resource,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_checkpoint import CheckpointStrings
from direct_sympy_strings import StringCompiler
from direct_sympy_partition_run import compile_region
from direct_sympy_region_parity import verify_region
root=Path(__file__).resolve().parent;checkpoint='docs/evidence/direct-sympy-test-checkpoint'
identity=live_identity(checkpoint,2)
with CheckpointStrings(checkpoint,StringCompiler())as m:domains=dict(m.domains)
r=compile_region(checkpoint,2,domains,root/'full.expr',max_characters=96*1024**2,cas_characters=96*1024**2,max_paths=128,max_seconds=240)
r['workerPeakRSSBytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
if r['complete']:
 r['parity']=verify_region(checkpoint,2,root/'full.expr',domains,random_cases=8192)
 assert r['parity']['mismatches']==0,r['parity']
assert identity==live_identity(checkpoint,2);r['compilerIdentity']=identity
(root/'full-run.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k]for k in ('complete','seconds','workerSeconds','stop','artifact','workerPeakRSSBytes')if k in r}))
