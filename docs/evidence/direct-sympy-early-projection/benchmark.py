"""Compare the same four bounded projection regions with one/two workers."""
import json,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
import torch
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_parallel import compile_wave
from direct_sympy_partition_run import encode,cardinality
from direct_sympy_input_partitions import interval
from direct_sympy_region_parity import verify_region
checkpoint='docs/evidence/direct-sympy-test-checkpoint';torch.set_num_threads(1)
domains=[]
for x in ((-2**-16,0),(2**-24,2**-16)):
 for y in ((-65504,-1),(1,65504)):
  domains.append({'X1':interval(*x),'X2':interval(*y)})
results=[]
with tempfile.TemporaryDirectory() as d:
 for workers in (1,2):
  start=time.monotonic();directory=Path(d)/str(workers);directory.mkdir()
  jobs=[(str(i),encode(domain),directory/(str(i)+'.expr')) for i,domain in enumerate(domains)]
  compiled,stats=compile_wave(checkpoint,2,jobs,workers=workers,memory_bytes=3*2**30,
   cas_characters=8388608,max_characters=1048576,max_paths=128,max_seconds=30)
  assert all(row['complete'] for row in compiled)
  parity=[verify_region(checkpoint,2,job[2],domain,random_cases=256) for job,domain in zip(jobs,domains)]
  assert not any(row['mismatches'] for row in parity)
  results.append({'workers':workers,'seconds':time.monotonic()-start,'memory':stats,
   'compiled':compiled,'parity':parity,'artifactHashes':[r['sha256'] for r in parity]})
assert results[0]['artifactHashes']==results[1]['artifactHashes']
report={'compilerIdentity':live_identity(checkpoint,2),'domains':[encode(d) for d in domains],
 'coveredPatterns':sum(cardinality(encode(d)) for d in domains),'sequential':results[0],'parallel':results[1],
 'speedRatio':results[0]['seconds']/results[1]['seconds'],'identicalDomainsAndArtifactHashes':True,
 'fullCoordinateParity':False,'scope':'Only the same four signed regions; not full-coordinate timing.'}
Path('artifacts/direct-sympy-input-partitions/early-projection-diagnosis/parallel-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
