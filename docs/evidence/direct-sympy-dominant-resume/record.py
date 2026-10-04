"""Audit all fresh numerical evidence and snapshot the compatible continuation."""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/dominant-square-state')
s=json.loads((state/'frontier.json').read_text())
p=json.loads((root/'partition-parity.json').read_text())
reports=json.loads((root/'recompiled.json').read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
assert s['identity']==p['compilerIdentity']==identity
assert audit_tree(s['tree'],s['root'])==(s['coveredInputPatterns'],s['unfinishedInputPatterns'])
assert p['coveredInputPatterns']==s['coveredInputPatterns'] and p['mismatches']==0
assert len(reports)==28
assert all(r['compilerIdentity']==identity and r.get('parity',{}).get('mismatches',0)==0 for r in reports)
assert all(r.get('compilation',{}).get('complete',False) or r['addedInputPatterns']==0 for r in reports)
rows=[]
for key,node in s['tree'].items():
    if node['status']!='complete':continue
    body=(state/node['artifact']['file']).read_bytes()
    sha=hashlib.sha256(body).hexdigest();assert sha==node['artifact']['sha256']
    filename='region-'+sha+'.expr';(root/filename).write_bytes(body)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'characters':len(body),'sha256':sha})
assert len(rows)==p['completedRegions']
previous=json.loads(Path('docs/evidence/direct-sympy-x2-covers/frontier.json').read_text())
assert previous['identity']!=identity and previous['identity']['checkpoint']==identity['checkpoint']
# Old completed boxes must be completely covered by newly proved leaves.
for old in previous['tree'].values():
    if old['status']!='complete':continue
    for node in s['tree'].values():
        if node['status']=='pending':
            assert any(node['domains'][axis][1]<low or node['domains'][axis][0]>high for axis,(low,high) in old['domains'].items())
prior_gate=json.loads(Path('docs/evidence/direct-sympy-dominant-square/compiler-identity.json').read_text())
assert prior_gate==identity
validation={'priorSourceCoverage':previous['coveredInputPatterns'],'coveredInputPatterns':s['coveredInputPatterns'],
 'additionalPatternsBeyondPriorSourceCoverage':s['coveredInputPatterns']-previous['coveredInputPatterns'],
 'unfinishedInputPatterns':s['unfinishedInputPatterns'],'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],
 'completedRegions':len(rows),'distinctArtifacts':len({row['sha256'] for row in rows}),
 'ordinaryAttempts':len(s['attempts']),'ordinaryCompletedAttempts':sum(r['complete'] for r in s['attempts']),
 'geometryRegionsRecompiled':len(reports),'recompiledRegionNativeCases':sum(r.get('parity',{}).get('cases',0) for r in reports),
 'allLeafNativeCases':p['cases'],'allLeafMismatches':p['mismatches'],
 'numericalResultsFromPreviousSourceReused':False,'allPreviousCompletedDomainsRecovered':True,
 'lastIntegrationGate':'docs/evidence/direct-sympy-dominant-square/test.log','integrationTests':25,
 'compilerSourcesChangedSinceGate':False,'compilationSpeedupMeasured':False,
 'fullCoordinateParity':False,'multipleTokenParity':False,'colabActiveSessions':0}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-dominant-resume/validation.json',**validation}
if 'dominantResumeValidation' not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "dominantResumeValidation": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)['dominantResumeValidation']==entry
for logfile in root.glob('*.log'):
    logfile.write_text('\n'.join(line.rstrip() for line in logfile.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))
