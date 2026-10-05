"""Compile remaining central quadrants without narrowing the full root."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import cover
from direct_sympy_partition_run import encode
from direct_sympy_input_partitions import interval

checkpoint='docs/evidence/direct-sympy-test-checkpoint'
directory='artifacts/direct-sympy-input-partitions/coupled-projection-state'
for a,b in ((-1,-1),(-1,1),(1,-1)):
    domains={'X1':interval(-2,-.0625) if a<0 else interval(.0625,2),
        'X2':interval(-1,-.0625) if b<0 else interval(.0625,1)}
    result=cover(checkpoint,directory,encode(domains),max_seconds=120,
        max_characters=8388608,random_cases=8192)
    print(json.dumps(result),flush=True)
