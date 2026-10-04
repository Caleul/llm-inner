from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_partition_run import cardinality

root=Path(__file__).resolve().parent
large=json.loads((root/'large-regions-parity.json').read_text())
small=json.loads((root/'coordinate-region-parity.json').read_text())
assert small['mismatches']==large['mismatches']==0
assert small['sourceSHA256']==large['sourceSHA256']
assert all(hashlib.sha256((Path('helpers')/name).read_bytes()).hexdigest()==sha for name,sha in large['sourceSHA256'].items())
log=(root/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',24),('pass',24),('fail',0),('skipped',0)))
unit=(root/'unit-test.log').read_text()
assert 'Ran 6 tests' in unit and 'Linear signed-zero factoring: cases=122888 mismatches=0' in unit
runs=[json.loads((root/'coordinate-region-run.json').read_text()),
    *json.loads((root/'large-regions-run.json').read_text())]
proofs=[small,*large['regions']];rows=[]
for run,proof in zip(runs,proofs):
    path=root/Path(run['artifact']['path']).name
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert run['complete'] and sha==run['artifact']['sha256']==proof['sha256']
    assert len(path.read_text())==proof['characters']
    rows.append({'file':path.name,'sha256':sha,'characters':proof['characters'],
        'domains':run['inputDomains'],'patterns':cardinality(run['inputDomains'])})
for i,a in enumerate(rows):
    for b in rows[i+1:]:
        assert any(a['domains'][key][1]<b['domains'][key][0] or b['domains'][key][1]<a['domains'][key][0] for key in a['domains'])
coverage=sum(row['patterns'] for row in rows)
validation={'tests':24,'pass':24,'fail':0,'skipped':0,'build':True,
    'exactFactorUnitTests':6,'linearZeroNativeCases':122888,'linearZeroNativeMismatches':0,
    'logicalCharactersBefore':2296124090440,'logicalCharactersAfter':2296124090440,
    'globalSizeImprovement':False,'regions':len(rows),'distinctArtifacts':len({row['sha256'] for row in rows}),
    'coveredInputPatterns':coverage,'totalInputPatterns':4030726144,
    'unfinishedInputPatterns':4030726144-coverage,'coveragePercent':100*coverage/4030726144,
    'nativeCases':sum(proof['cases'] for proof in proofs),'nativeMismatches':0,
    'addsToPreviousTreeCoverage':False,'oldSavedNumericStateReused':False,
    'sourceSHA256':large['sourceSHA256'],
    'checkpointSHA256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path('docs/evidence/direct-sympy-test-checkpoint/config.json'),*sorted(Path('docs/evidence/direct-sympy-test-checkpoint').glob('*.safetensors'))]},
    'finalArtifactEmitted':False,'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'artifact-map.json').write_text(json.dumps(rows,indent=2)+'\n')
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
assert 'linearZeroFactoringValidation' not in json.loads(original)
entry=json.dumps({'report':str(root/'validation.json'),**validation},indent=2)
updated=original.rstrip()[:-1].rstrip()+',\n  "linearZeroFactoringValidation": '+'\n'.join('  '+line for line in entry.splitlines()).lstrip()+'\n}\n'
json.loads(updated);path.write_text(updated)
for path in root.glob('*.log'):
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))
