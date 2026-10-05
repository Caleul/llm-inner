"""Fresh actual emission and exhaustive regional parity; no runtime lookup."""
import itertools,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_partition_run import compile_region
from direct_sympy_input_partitions import interval
from direct_sympy_region_parity import verify_region
from direct_sympy_cover_regions import live_identity
checkpoint='docs/evidence/direct-sympy-test-checkpoint'
root=Path(__file__).resolve().parent
identity=live_identity(checkpoint,2)
domains={name:interval(-2**-24,2**-24)for name in ('X1','X2')}
r=compile_region(checkpoint,2,domains,root/'small.expr',max_characters=32*1024**2,cas_characters=32*1024**2,max_paths=128,max_seconds=180)
assert r['complete'],r
corpus=list(itertools.product((0,0x8000,1,0x8001),repeat=2))
# Enumerate every IEEE input in this region, including all signed-zero pairs.
# This replaces only the verifier's sampling corpus, not generated arithmetic.
with patch('direct_sympy_region_parity.sample_bits',return_value=(['X1','X2'],corpus)):
 parity=verify_region(checkpoint,2,root/'small.expr',domains,random_cases=0)
assert parity['cases']==16 and parity['mismatches']==0,parity
assert identity==live_identity(checkpoint,2)
r['compilerIdentity']=identity
parity.update(compilerIdentity=identity,exhaustiveRegion=True,inputPatterns=16,inputBits=corpus)
(root/'emission.json').write_text(json.dumps(r,indent=2)+'\n')
(root/'parity.json').write_text(json.dumps(parity,indent=2)+'\n')
print(json.dumps({'characters':r['artifact']['characters'],'cases':parity['cases'],'mismatches':parity['mismatches'],'sha256':parity['sha256']}))
