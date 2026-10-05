"""Check fresh artifact identity, exact regional parity and regression evidence."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
previous=json.loads((root.parent/'direct-sympy-logical-prefix/validation.json').read_text())
old=previous['compilerIdentity']
assert identity['checkpoint']==old['checkpoint'] and identity['referenceBackend']==old['referenceBackend']
assert set(identity['sources'])==set(old['sources'])
assert {name for name in old['sources']if identity['sources'][name]!=old['sources'][name]}=={'direct_sympy_coherent_paths.py'}
regions=[]
for label,cases in (('near-zero',16),('current',66049),('mixed',252)):
 r=load(label);assert r['compilerIdentity']==identity and r['complete']
 assert r['parity']['exhaustiveRegion'] and r['parity']['cases']==cases and r['parity']['mismatches']==0
 assert r['artifact']['complete'] and r['artifact']['compilerAliases']==0
 assert r['artifact']['characters']==len((root/(label+'.expr')).read_text())
 assert r['artifact']['sha256']==r['parity']['sha256']==digest_file(root/(label+'.expr'))
 regions.append({'inputDomains':r['inputDomains'],'artifact':r['artifact'],'parity':r['parity'],
  'seconds':r['seconds'],'workerPeakRSSBytes':r['workerPeakRSSBytes']})
diagnostics=[]
for mib in (32,96):
 d=load('diagnostic-'+str(mib));assert d['compilerIdentity']==identity
 assert d['stats']['completedPaths']==14 and 'budget' in d['stop'].lower()
 assert d['stats']['lastArm']['bodyCharacters']==172480895
 assert d['stats']['lastArm']['guardCharacters']==86741265
 assert d['stats']['lastArm']['decisions']==6
 assert d['stats']['prospectiveGuardEliminations']==26
 assert not (root/('full-'+str(mib)+'.expr')).exists();diagnostics.append(d)
def counts(name):
 text=(root/name).read_text()
 return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
small,whole=counts('focused-tests.log'),counts('general-tests.log')
assert small==previous['integrationTests'] and small['fail']==0
failures=re.findall(r'^test at (\S+)',(root/'general-tests.log').read_text(),re.M)
assert whole==previous['wholeProjectSuite']['current'],whole
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
proof=(root/'coherent-tests.log').read_text();assert 'Ran 22 tests' in proof and '\nOK\n' in proof
assert 'Flat checkpoint normalization: cases=888832 mismatches=0' in proof
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert not (root/'full.expr').exists()
v={'compilerIdentity':identity,'baselineCommit':'cc7f113','expressionRepresentation':'SymPy mathematical strings',
 'regionalArtifacts':regions,'regionalCases':66317,'regionalBitMismatches':0,
 'logicalDispatchHalfPatterns':30722,'coherentTests':22,'nativeNormalizationCases':888832,
 'prospectiveGuardEliminations':26,'candidateNumber':14,'candidateBodyCharacters':172480895,'candidateGuardCharacters':86741265,
 'matchingCandidateComparison':False,'fullDomainDiagnostics':diagnostics,'fullDomainAttempt':run,
 'integrationTests':small,'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'oldNumericalSavepointsCompatible':False,'fullCoordinateArtifactEmitted':False,
 'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();oldmap=json.loads(raw);key='prospectiveGuardFeasibilityValidation'
if key in oldmap:assert oldmap[key]==v
else:
 result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(result).items()if k!=key}==oldmap
 p.write_text(result)
print(json.dumps({'regionalCases':66317,'mismatches':0,'integration':small,'generalSuite':whole,'fullCoordinateParity':False}))
