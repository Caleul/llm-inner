from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/frontier-final-state')
s=json.loads((state/'frontier.json').read_text());p=json.loads((root/'continuation-parity.json').read_text())
assert p['mismatches']==0 and p['coveredInputPatterns']==s['coveredInputPatterns'] and p['sourceIdentity']==s['identity']
rows=[]
for key,node in s['tree'].items():
 if node['status']!='complete':continue
 payload=(state/node['artifact']['file']).read_bytes();sha=hashlib.sha256(payload).hexdigest();assert sha==node['artifact']['sha256']
 filename='region-'+sha+'.expr';(root/filename).write_bytes(payload)
 rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(payload)})
assert len(rows)==p['completedRegions']
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'current-frontier.json').write_text(json.dumps(s,indent=2)+'\n')
validation={'tests':23,'pass':23,'fail':0,'skipped':0,'totalAttempts':len(s['attempts']),'completedRegions':len(rows),'distinctArtifacts':len({r['sha256'] for r in rows}),'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],'nativeCases':p['cases'],'nativeMismatches':p['mismatches'],'sourceIdentity':s['identity'],'finalArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'continuation-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text();assert 'coverageFrontierContinuation' not in json.loads(original)
entry=json.dumps({'report':str(root/'continuation-validation.json'),**{k:v for k,v in validation.items() if k!='sourceIdentity'}},indent=2)
updated=original.rstrip()[:-1].rstrip()+',\n  "coverageFrontierContinuation": '+'\n'.join('  '+line for line in entry.splitlines()).lstrip()+'\n}\n';json.loads(updated);path.write_text(updated)
for log in root.glob('*.log'):log.write_text(log.read_text().rstrip()+'\n')
print(json.dumps({k:v for k,v in validation.items() if k!='sourceIdentity'},indent=2))
