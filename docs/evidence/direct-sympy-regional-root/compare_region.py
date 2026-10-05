"""Measure one compiler version in isolation and verify actual emitted bytes."""
import itertools,json,resource,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
from direct_sympy_partition_run import compile_region
from direct_sympy_input_partitions import interval,rank
from direct_sympy_region_parity import verify_region
from direct_sympy_cover_regions import live_identity
checkpoint=str(Path('docs/evidence/direct-sympy-test-checkpoint').resolve())
root=Path(__file__).resolve().parent;label=sys.argv[2]
identity=live_identity(checkpoint,2)
domains={name:interval(1/2048,5/8192)for name in ('X1','X2')}
result=compile_region(checkpoint,2,domains,root/(label+'.expr'),max_characters=32*1024**2,cas_characters=32*1024**2,max_paths=128,max_seconds=180)
result['workerPeakRSSBytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
assert sys.platform=='darwin','Peak RSS unit is macOS bytes'
if result['complete']:
 corpus=list(itertools.product(range(rank(1/2048),rank(5/8192)+1),repeat=2))
 with patch('direct_sympy_region_parity.sample_bits',return_value=(['X1','X2'],corpus)):
  parity=verify_region(checkpoint,2,root/(label+'.expr'),domains,random_cases=0)
 assert parity['cases']==66049 and parity['mismatches']==0,parity
 parity.update(exhaustiveRegion=True,inputPatterns=66049,compilerIdentity=identity)
 result['parity']=parity
assert identity==live_identity(checkpoint,2)
result['compilerIdentity']=identity
(root/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key]for key in ('complete','seconds','workerSeconds','workerPeakRSSBytes','artifact','parity')if key in result}))
