from pathlib import Path
import hashlib,json,shutil,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_partition_run import audit_tree
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/pruned-paths-state')
frontier=json.loads((state/'frontier.json').read_text())
covered,pending=audit_tree(frontier['tree'],frontier['root'])
previous=json.loads(Path('docs/evidence/direct-sympy-structural-word-comparison/previous-source-frontier-72.json').read_text())
rows=[];compared=[]
for key,node in frontier['tree'].items():
 if node['status']!='complete':continue
 source=state/node['artifact']['file'];payload=source.read_bytes();sha=hashlib.sha256(payload).hexdigest()
 if sha!=node['artifact']['sha256']:raise ValueError('Artifact integrity mismatch')
 file='region-'+sha+'.expr';(root/file).write_bytes(payload)
 rows.append({'region':key,'domains':node['domains'],'file':file,'sha256':sha,'characters':len(payload)})
 old=previous['tree'].get(key)
 if old and old['status']=='complete' and old['domains']==node['domains']:
  compared.append({'region':key,'previousCharacters':old['artifact']['characters'],'currentCharacters':len(payload)})
(root/'frontier.json').write_text(json.dumps(frontier,indent=2)+'\n')
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
comparison={'regions':compared,'matchedRegions':len(compared),'reducedRegions':sum(r['currentCharacters']<r['previousCharacters'] for r in compared),'increasedRegions':sum(r['currentCharacters']>r['previousCharacters'] for r in compared),'previousCharacters':sum(r['previousCharacters'] for r in compared),'currentCharacters':sum(r['currentCharacters'] for r in compared),'scope':'Matching input regions across previous and current source versions; not an isolated timing benchmark or full-domain artifact.'}
(root/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
shutil.copyfile('artifacts/direct-sympy-input-partitions/rms-components-continuation.log',root/'previous-continuation.log')
print(len(rows),covered,pending,len({r['sha256'] for r in rows}))
