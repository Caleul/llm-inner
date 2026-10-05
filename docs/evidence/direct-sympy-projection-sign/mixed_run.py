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
domains={'X1':interval(-2**-24,2**-24),'X2':interval(2**-23,2**-18)}
result=compile_region(checkpoint,2,domains,root/(label+'.expr'),max_characters=32*1024**2,cas_characters=32*1024**2,max_paths=128,max_seconds=180)
result['workerPeakRSSBytes']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
assert sys.platform=='darwin','Peak RSS unit is macOS bytes'
if result['complete']:
 corpus=list(itertools.product((0,0x8000,1,0x8001),range(2,65)))
 with patch('direct_sympy_region_parity.sample_bits',return_value=(['X1','X2'],corpus)):
  parity=verify_region(checkpoint,2,root/(label+'.expr'),domains,random_cases=0)
 assert parity['cases']==252 and parity['mismatches']==0,parity
 parity.update(exhaustiveRegion=True,inputPatterns=252,compilerIdentity=identity)
 result['parity']=parity
assert identity==live_identity(checkpoint,2)
result['compilerIdentity']=identity
(root/(label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:result[key]for key in ('complete','seconds','workerSeconds','workerPeakRSSBytes','artifact','parity')if key in result}))
