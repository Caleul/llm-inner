"""Read only proven domain geometry; recompile/parity-check every expression."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_partition_run import audit_tree,cardinality
from direct_sympy_cover_regions import cover
from direct_sympy_savepoints import atomic
root=Path(__file__).resolve().parent
old=json.loads(Path('artifacts/direct-sympy-input-partitions/update-cell-state/frontier.json').read_text())
audit_tree(old['tree'],old['root'])
state=Path('artifacts/direct-sympy-input-partitions/dominant-square-state')
regions=sorted(((key,node['domains']) for key,node in old['tree'].items() if node['status']=='complete'),key=lambda item:(-cardinality(item[1]),item[0]))
reports=[]
for key,domains in regions:
    result=cover('docs/evidence/direct-sympy-test-checkpoint',state,domains,max_seconds=15,random_cases=256)
    reports.append({'geometrySourceKey':key,**result})
    atomic(root/'recompiled.json',json.dumps(reports,indent=2).encode())
    print(json.dumps({'sourceRegion':key,'added':result['addedInputPatterns'],'covered':result['coveredAfter'],'stop':result.get('stop'),'mismatches':result.get('parity',{}).get('mismatches')}),flush=True)
