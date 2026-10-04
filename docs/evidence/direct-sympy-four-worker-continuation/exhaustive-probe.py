"""Exhaustively compare the emitted probe region; never build response lookup."""
from pathlib import Path
import itertools,json,sys
from unittest.mock import patch
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_region_parity import verify_region
from direct_sympy_partition_run import decode
import torch
torch.set_num_threads(1)
root=Path(__file__).resolve().parent
domains=json.loads((root/'diagnosis.json').read_text())['inputDomains']
names=sorted(domains,key=lambda key:int(key[1:]))
# Both axes exclude zero; these are all distinct finite Half bit patterns.
assert all(not a<=0<=b for a,b in domains.values())
bits=[tuple(abs(v)|(0x8000 if v<0 else 0) for v in row) for row in itertools.product(*(range(a,b+1) for a,b in (domains[name] for name in names)))]
with patch('direct_sympy_region_parity.sample_bits',return_value=(names,bits)):
    result=verify_region('docs/evidence/direct-sympy-test-checkpoint',2,root/'probe.expr',decode(domains),random_cases=0)
result['corpus']='Every Half input-vector pattern in the recorded probe rectangle'
result['fullRegionEnumeration']=True
(root/'exhaustive-parity.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['cases'],result['mismatches'])
