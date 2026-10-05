"""Verify live sources, regional expressions and unchanged failure locations."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
previous=json.loads((root.parent/'direct-sympy-bidirectional-recipes/validation.json').read_text())
old=previous['compilerIdentity']
assert identity['checkpoint']==old['checkpoint'] and identity['referenceBackend']==old['referenceBackend']
assert set(identity['sources'])==set(old['sources'])
assert {name for name in old['sources']if identity['sources'][name]!=old['sources'][name]}=={'direct_sympy_conversions.py'}
regions=[]
for label,cases in (('near-zero',16),('current',66049),('mixed',252)):
 r=load(label);assert r['compilerIdentity']==identity and r['complete']
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==cases and r['parity']['mismatches']==0
 assert r['artifact']['complete'] and r['artifact']['compilerAliases']==0
 assert r['artifact']['characters']==len((root/(label+'.expr')).read_text())
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
 regions.append({'inputDomains':r['inputDomains'],'artifact':r['artifact'],'parity':r['parity'],
  'seconds':r['seconds'],'workerPeakRSSBytes':r['workerPeakRSSBytes']})
baseline,candidate=load('baseline-candidate'),load('candidate')
assert baseline['compilerIdentity']==old and candidate['compilerIdentity']==identity
assert baseline['candidate']==candidate['candidate']==8
assert baseline['bodyCharacters']==3577798761 and candidate['bodyCharacters']==3571931313
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert not (root/'full.expr').exists()
def counts(name):
 text=(root/name).read_text()
 return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
small,whole=counts('focused-tests.log'),counts('general-tests.log')
failures=re.findall(r'^test at (\S+)',(root/'general-tests.log').read_text(),re.M)
assert small=={'tests':36,'pass':36,'fail':0,'skipped':0},small
expected=dict(previous['wholeProjectSuite']['current']);expected['tests']+=1;expected['skipped']+=1
assert whole==expected,whole
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
gap=(root/'gap-proof.log').read_text();assert 'Ran 1 test' in gap and '\nOK\n' in gap
assert 'Residual magnitude exclusion: cases=147456 violations=0' in gap
bounds=(root/'bounds-proof.log').read_text();assert 'Ran 4 tests' in bounds and '\nOK\n' in bounds
proof=(root/'coherent-proof.log').read_text();assert 'Ran 19 tests' in proof and '\nOK\n' in proof
v={'compilerIdentity':identity,'baselineCommit':'ff84828','expressionRepresentation':'SymPy mathematical strings',
 'regionalArtifacts':regions,'regionalCases':66317,'regionalBitMismatches':0,
 'nativeResidualMagnitudeCases':147456,'nativeResidualMagnitudeViolations':0,
 'baselineCandidateBodyCharacters':baseline['bodyCharacters'],'currentCandidateBodyCharacters':candidate['bodyCharacters'],
 'candidateNumber':8,'candidateIsFinalArtifact':False,'fullDomainAttempt':run,
 'integrationTests':small,'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'oldNumericalSavepointsCompatible':False,'fullCoordinateArtifactEmitted':False,
 'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();oldmap=json.loads(raw);key='residualMagnitudeGapValidation'
if key in oldmap:assert oldmap[key]==v
else:
 result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(result).items()if k!=key}==oldmap
 p.write_text(result)
print(json.dumps({'regionalCases':66317,'mismatches':0,'integration':small,'generalSuite':whole,
 'candidateBodyCharacters':candidate['bodyCharacters'],'fullCoordinateParity':False}))
