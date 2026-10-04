"""Audit and snapshot X2 covers without modifying compiler source or geometry."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/update-cell-state')
s=json.loads((state/'frontier.json').read_text())
before=json.loads(Path('docs/evidence/direct-sympy-central-cover/frontier.json').read_text())
p=json.loads((root/'partition-parity.json').read_text())
proofs=[json.loads((root/(name+'.json')).read_text()) for name in ('x2-negative','x2-positive')]
assert s['identity']==before['identity']==p['compilerIdentity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert audit_tree(s['tree'],s['root'])==(s['coveredInputPatterns'],s['unfinishedInputPatterns'])
assert p['mismatches']==0 and p['coveredInputPatterns']==s['coveredInputPatterns']
old=[key for key,node in before['tree'].items() if node['status']=='complete']
assert all(s['tree'][key]==before['tree'][key] for key in old)
assert s['attempts']==before['attempts'] and proofs==s['coveragePromotions'][-2:]
assert sum(r['addedInputPatterns'] for r in proofs)==s['coveredInputPatterns']-before['coveredInputPatterns']
rows=[]
for proof in proofs:
    assert proof['parity']['mismatches']==0 and proof['compilation']['complete']
    assert proof['compilerIdentity']==s['identity']
    assert proof['toolSHA256']==hashlib.sha256(Path('helpers/direct_sympy_cover_regions.py').read_bytes()).hexdigest()
    artifact=proof['sharedArtifact'];payload=(state/artifact['file']).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==artifact['sha256']
    (root/artifact['file']).write_bytes(payload)
    rows.append({'domains':proof['inputDomains'],**artifact,'admittedRegions':proof['admittedRegions']})
for name in ('x2-negative-wide','x2-negative-medium'):
    failed=json.loads((root/(name+'.json')).read_text())
    assert not failed['compilation']['complete'] and failed['addedInputPatterns']==0
    assert failed['coveredBefore']==failed['coveredAfter']==s['coveredInputPatterns']
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
validation={'coveredBefore':before['coveredInputPatterns'],'addedInputPatterns':s['coveredInputPatterns']-before['coveredInputPatterns'],
 'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],
 'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],'completedRegions':p['completedRegions'],
 'oldCompletedLeavesPreserved':len(old),'newNativeCases':sum(r['parity']['cases'] for r in proofs),'newNativeMismatches':0,
 'allLeafNativeCases':p['cases'],'allLeafNativeMismatches':p['mismatches'],
 'compilerSourcesChanged':False,'lastIntegrationGate':'docs/evidence/direct-sympy-central-cover/test.log',
 'compilationSpeedupMeasured':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-x2-covers/validation.json',**validation}
if 'x2BroadCoverValidation' not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "x2BroadCoverValidation": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)['x2BroadCoverValidation']==entry
print(json.dumps(validation,indent=2))
