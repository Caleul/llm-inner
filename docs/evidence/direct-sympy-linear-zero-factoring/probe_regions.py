"""Compile broad certified domains, then validate actual emitted bodies."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path('helpers').resolve()))
import torch
from direct_sympy_partition_run import compile_region,decode,cardinality
from direct_sympy_input_partitions import interval
from direct_sympy_region_parity import verify_region
from direct_sympy_savepoints import SOURCES,digest_file

torch.set_num_threads(1)
root=Path(__file__).resolve().parent
rows=[];proofs=[]
for i,(a,b) in enumerate([((-65504,-32),(-65504,-32)),((32,65504),(32,65504)),
    ((-65504,-32),(32,65504)),((32,65504),(-65504,-32))]):
    result=compile_region('docs/evidence/direct-sympy-test-checkpoint',2,
        {'X1':interval(*a),'X2':interval(*b)},root/f'large-region-{i}.expr',
        max_characters=1048576,cas_characters=8388608,max_paths=128,max_seconds=30)
    assert result['complete'];rows.append(result)
    proof=verify_region('docs/evidence/direct-sympy-test-checkpoint',2,
        result['artifact']['path'],decode(result['inputDomains']),random_cases=8192)
    assert proof['mismatches']==0;proofs.append(proof)
(root/'large-regions-run.json').write_text(json.dumps(rows,indent=2)+'\n')
report={'regions':proofs,'coveredInputPatterns':sum(cardinality(row['inputDomains']) for row in rows),
    'cases':sum(p['cases'] for p in proofs),'mismatches':sum(p['mismatches'] for p in proofs),
    'distinctArtifacts':len({p['sha256'] for p in proofs}),
    'sourceSHA256':{name:digest_file(Path('helpers')/name) for name in SOURCES},
    'fullCoordinateParity':False,'addsToPreviousTreeCoverage':False}
(root/'large-regions-parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
