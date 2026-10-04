from pathlib import Path
import json,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_region_parity import verify_region
from direct_sympy_partition_run import decode,audit_tree
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/pruned-paths-state')
saved=json.loads((state/'frontier.json').read_text())
old=json.loads((root/'initial-parity.json').read_text())
cache={Path(r['artifact']).name:r for r in old['regions']}
reports=[]
for node in saved['tree'].values():
 if node['status']!='complete':continue
 artifact=state/node['artifact']['file'];sha=digest_file(artifact)
 if sha!=node['artifact']['sha256']:raise ValueError('Artifact mismatch')
 report=cache.get(artifact.name)
 if report is not None:
  if report['sha256']!=sha or report['mismatches']:raise ValueError('Initial parity mismatch')
 else:report=verify_region('docs/evidence/direct-sympy-test-checkpoint',2,artifact,decode(node['domains']))
 reports.append(report)
covered,pending=audit_tree(saved['tree'],saved['root'])
result={'regions':reports,'completedRegions':len(reports),'coveredInputPatterns':covered,'unfinishedInputPatterns':pending,'cases':sum(r['cases'] for r in reports),'mismatches':sum(r['mismatches'] for r in reports),'sourceIdentity':saved['identity'],'fullCoordinateParity':False}
(root/'partition-parity.json').write_text(json.dumps(result,indent=2)+'\n')
if result['mismatches']:raise ValueError('Coordinate parity failed')
print(json.dumps({k:v for k,v in result.items() if k not in ('regions','sourceIdentity')}),flush=True)
