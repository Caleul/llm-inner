"""Snapshot fresh broad-region proofs and exact pending-only coverage."""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/update-cell-state')
s=json.loads((state/'frontier.json').read_text())
before=json.loads((root/'frontier-before.json').read_text())
p=json.loads((root/'partition-parity.json').read_text())
proofs=[json.loads((root/(name+'-cover.json')).read_text()) for name in ('negative','positive')]
assert s['identity']==before['identity']==p['compilerIdentity']==live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert audit_tree(s['tree'],s['root'])==(s['coveredInputPatterns'],s['unfinishedInputPatterns'])
assert p['mismatches']==0 and p['coveredInputPatterns']==s['coveredInputPatterns']
old=[key for key,node in before['tree'].items() if node['status']=='complete']
assert all(s['tree'][key]==before['tree'][key] for key in old)
assert s['attempts']==before['attempts']
assert sum(r['addedInputPatterns'] for r in proofs)==s['coveredInputPatterns']-before['coveredInputPatterns']
assert proofs==s['coveragePromotions']
for proof in proofs:
    assert proof['compilerIdentity']==s['identity']
    assert proof['toolSHA256']==hashlib.sha256(Path('helpers/direct_sympy_cover_regions.py').read_bytes()).hexdigest()
    assert proof['parity']['mismatches']==0 and proof['compilation']['complete']
    for key in proof['admittedRegions']:
        node=s['tree'][key]
        assert node['artifact']==proof['sharedArtifact'] and node['compiledDomain']==proof['inputDomains']
        assert all(proof['inputDomains'][name][0]<=a<=b<=proof['inputDomains'][name][1] for name,(a,b) in node['domains'].items())
log=(root/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',25),('pass',25),('fail',0),('skipped',0)))
rows=[]
for key,node in s['tree'].items():
    if node['status']!='complete':continue
    payload=(state/node['artifact']['file']).read_bytes()
    sha=hashlib.sha256(payload).hexdigest();assert sha==node['artifact']['sha256']
    filename='region-'+sha+'.expr';(root/filename).write_bytes(payload)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(payload),'compiledDomain':node.get('compiledDomain',node['domains'])})
assert len(rows)==p['completedRegions']
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
validation={'tests':25,'pass':25,'fail':0,'skipped':0,'build':True,'coverUnitTests':3,
 'oldCompletedLeavesPreserved':len(old),'ordinaryAttemptsUnchanged':len(s['attempts']),
 'freshCoveragePromotions':len(proofs),'completedRegions':len(rows),'distinctArtifacts':len({r['sha256'] for r in rows}),
 'coveredBefore':before['coveredInputPatterns'],'addedInputPatterns':s['coveredInputPatterns']-before['coveredInputPatterns'],
 'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],
 'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],
 'nativeCases':p['cases'],'nativeMismatches':p['mismatches'],
 'freshBroadRegionNativeCases':sum(r['parity']['cases'] for r in proofs),'freshBroadRegionMismatches':0,
 'allCompilerSourcesCompatible':True,'compiledAfreshBeforePromotion':True,
 'compilationSpeedupMeasured':False,'finalArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-central-cover/validation.json',**validation}
if 'centralCoverValidation' not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "centralCoverValidation": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)['centralCoverValidation']==entry
for logfile in root.glob('*.log'):
    logfile.write_text('\n'.join(line.rstrip() for line in logfile.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))
