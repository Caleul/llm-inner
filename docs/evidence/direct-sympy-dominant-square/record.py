"""Audit the numerical change, effective artifacts and compatible new state."""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_partition_run import audit_tree
root=Path(__file__).resolve().parent
new=json.loads((root/'dominated.json').read_text());baseline=json.loads((root/'dominated-baseline.json').read_text())
p=json.loads((root/'baseline-parity.json').read_text())
assert new['complete'] and baseline['complete']
assert new['parity']['mismatches']==p['mismatches']==0
for record,path in ((new,root/'dominated.expr'),(baseline,root/'baseline.expr')):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['artifact']['sha256']
assert new['inputDomains']==baseline['inputDomains']
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
state=Path('artifacts/direct-sympy-input-partitions/dominant-square-state')
s=json.loads((state/'frontier.json').read_text());assert s['identity']==identity
covered,pending=audit_tree(s['tree'],s['root'])
assert covered==s['coveredInputPatterns'] and pending==s['unfinishedInputPatterns']
for node in s['tree'].values():
    if node['status']=='complete':
        payload=(state/node['artifact']['file']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==node['artifact']['sha256']
        (root/node['artifact']['file']).write_bytes(payload)
old=json.loads(Path('artifacts/direct-sympy-input-partitions/update-cell-state/frontier.json').read_text())
assert old['identity']!=identity
log=(root/'test.log').read_text()
assert all(re.search(r'ℹ '+name+' '+str(number)+r'\b',log) for name,number in (('tests',25),('pass',25),('fail',0),('skipped',0)))
unit=(root/'unit.log').read_text();assert 'Ran 5 tests' in unit and '\nOK\n' in unit
assert 'kernels=126972 normCases=90112 mismatches=0' in unit
(root/'compiler-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
(root/'frontier.json').write_text(json.dumps(s,indent=2)+'\n')
validation={'build':True,'integrationTests':25,'pass':25,'fail':0,'skipped':0,'componentTests':5,
 'exhaustiveHalfSquareRootCases':126972,'compiledNormCases':90112,'nativeMismatches':0,
 'oldArtifactCharacters':baseline['artifact']['characters'],'newArtifactCharacters':new['artifact']['characters'],
 'artifactReductionPercent':100*(1-new['artifact']['characters']/baseline['artifact']['characters']),
 'completeRegionPaths':new['artifact']['paths'],'newRegionNativeCases':new['parity']['cases'],
 'baselineRegionNativeCases':p['cases'],'regionMismatches':0,
 'newSourceCompatibleCoveredPatterns':covered,'newSourceCompatiblePendingPatterns':pending,
 'previousSourceCoverageNotAdopted':True,'compilationSpeedupMeasured':False,
 'fullCoordinateParity':False,'multipleTokenParity':False}
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
path=Path('docs/direct-string-validation.json');original=path.read_text()
entry={'report':'docs/evidence/direct-sympy-dominant-square/validation.json',**validation}
if 'dominantSquareValidation' not in json.loads(original):
    formatted=json.dumps(entry,indent=2)
    updated=original.rstrip()[:-1].rstrip()+',\n  "dominantSquareValidation": '+'\n'.join('  '+line for line in formatted.splitlines()).lstrip()+'\n}\n'
    json.loads(updated);path.write_text(updated)
else:assert json.loads(original)['dominantSquareValidation']==entry
for logfile in root.glob('*.log'):
    logfile.write_text('\n'.join(line.rstrip() for line in logfile.read_text().splitlines()).rstrip()+'\n')
print(json.dumps(validation,indent=2))
