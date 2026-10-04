"""Preserve emitted expressions and fresh parity without adopting old proofs."""
from pathlib import Path
import hashlib,json,re

root=Path(__file__).resolve().parent
state=Path('artifacts/direct-sympy-input-partitions/partial-fractions-state')
s=json.loads((state/'frontier.json').read_text())
p=json.loads((root/'partition-parity.json').read_text())
assert p['mismatches']==0 and p['compilerIdentity']==s['identity']
assert p['coveredInputPatterns']==s['coveredInputPatterns']
assert p['inputCorpusVersion']==2
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
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
matched=json.loads((root/'matched-region-sizes.json').read_text())
before=sum(r['before'] for r in matched);after=sum(r['after'] for r in matched)
coordinate=json.loads((root/'coordinate-run.json').read_text())
validation={'tests':24,'pass':24,'fail':0,'skipped':0,'build':True,
    'totalAttempts':len(s['attempts']),'completedRegions':len(rows),
    'distinctArtifacts':len({r['sha256'] for r in rows}),
    'coveredInputPatterns':s['coveredInputPatterns'],'unfinishedInputPatterns':s['unfinishedInputPatterns'],
    'coveragePercent':100*s['coveredInputPatterns']/s['totalInputPatterns'],
    'nativeCases':p['cases'],'nativeMismatches':p['mismatches'],
    'inputCorpusVersion':2,'verifierSHA256':p['verifierSHA256'],
    'allCompilerSourcesCompatible':True,'numericalResultsReused':False,
    'matchedRegions':len(matched),'matchedBeforeCharacters':before,'matchedAfterCharacters':after,
    'matchedSizeReductionPercent':100*(1-after/before),
    'logicalCharactersBefore':3248163585496,'logicalCharactersAfter':coordinate['logicalCharacters'],
    'logicalReductionPercent':100*(1-coordinate['logicalCharacters']/3248163585496),
    'firstSelectedBodyBeforeCharacters':8615711603,
    'firstSelectedBodyAfterCharacters':coordinate['paths']['lastArm']['bodyCharacters'],
    'sqrtValidation':json.loads((root/'sqrt-validation.json').read_text()),
    'compilationSpeedupMeasured':False,'gpuValidationThisVersion':False,
    'finalArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
assert 'partialFractionsValidation' not in json.loads(original)
entry=json.dumps({'report':str(root/'validation.json'),**validation},indent=2)
updated=original.rstrip()[:-1].rstrip()+',\n  "partialFractionsValidation": '+'\n'.join('  '+line for line in entry.splitlines()).lstrip()+'\n}\n'
json.loads(updated);path.write_text(updated)
for log in root.glob('*.log'):log.write_text(log.read_text().rstrip()+'\n')
print(json.dumps(validation,indent=2))
