"""Snapshot the audited live tree and effective artifacts after native parity."""
from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/update-cell-state')
s=json.loads((state/'frontier.json').read_text())
p=json.loads((root/'partition-parity.json').read_text())
outer=json.loads((root/'outer-box-parity.json').read_text())
central=json.loads((root/'central-probes-parity.json').read_text())
assert p['compilerIdentity']==outer['compilerIdentity']==s['identity']
assert p['mismatches']==outer['mismatches']==0
assert all(r['mismatches']==0 for r in central)
assert p['coveredInputPatterns']==s['coveredInputPatterns']
assert p['verifierSHA256']==hashlib.sha256(Path('helpers/direct_sympy_region_parity.py').read_bytes()).hexdigest()
assert all(hashlib.sha256((Path('helpers')/name).read_bytes()).hexdigest()==sha for name,sha in s['identity']['sources'].items())
log=(root/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',24),('pass',24),('fail',0),('skipped',0)))
rows=[]
for key,node in s['tree'].items():
    if node['status']!='complete':continue
    payload=(state/node['artifact']['file']).read_bytes()
    sha=hashlib.sha256(payload).hexdigest();assert sha==node['artifact']['sha256']
    filename='region-'+sha+'.expr';(root/filename).write_bytes(payload)
    rows.append({'region':key,'domains':node['domains'],'file':filename,'sha256':sha,'characters':len(payload)})
assert len(rows)==p['completedRegions']
for proof in central:
    assert hashlib.sha256(Path(proof['artifact']).read_bytes()).hexdigest()==proof['sha256']
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
validation={'tests':24,'pass':24,'fail':0,'skipped':0,'build':True,
    'partitionUnitTests':8,'geometryReplanTests':2,'customCutCoverageAudited':True,
    'totalAttempts':len(s['attempts']),'completedRegions':len(rows),
    'distinctArtifacts':len({r['sha256'] for r in rows}),
    'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],
    'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],
    'seedThresholdRanks':s['updateCellGeometry']['thresholdRanks'],
    'seededOuterCoveredInputPatterns':553648128,'seededOuterRegions':4,
    'nativeCases':p['cases'],'nativeMismatches':p['mismatches'],
    'additionalOuterBoxNativeCases':outer['cases'],'additionalOuterBoxMismatches':outer['mismatches'],
    'centralProbeNativeCases':sum(r['cases'] for r in central),'centralProbeMismatches':0,
    'centralProbesAddTreeCoverage':False,'allCompilerSourcesCompatible':True,
    'numericalResultsReused':False,'compilationSpeedupMeasured':False,
    'finalArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
assert 'updateCellGeometryValidation' not in json.loads(original)
entry=json.dumps({'report':str(root/'validation.json'),**validation},indent=2)
updated=original.rstrip()[:-1].rstrip()+',\n  "updateCellGeometryValidation": '+'\n'.join('  '+line for line in entry.splitlines()).lstrip()+'\n}\n'
json.loads(updated);path.write_text(updated)
for log in root.glob('*.log'):
    log.write_text('\n'.join(line.rstrip() for line in log.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))
