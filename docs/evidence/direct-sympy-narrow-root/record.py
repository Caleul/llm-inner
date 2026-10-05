"""Check fresh artifact identity, exact regional parity and regression evidence."""
import json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path('helpers').resolve()))
from direct_sympy_cover_regions import live_identity
from direct_sympy_savepoints import digest_file
root=Path(__file__).resolve().parent
load=lambda name:json.loads((root/(name+'.json')).read_text())
identity=live_identity('docs/evidence/direct-sympy-test-checkpoint',2)
previous=json.loads((root.parent/'direct-sympy-prospective-guards/validation.json').read_text())
old=previous['compilerIdentity']
assert identity['checkpoint']==old['checkpoint'] and identity['referenceBackend']==old['referenceBackend']
assert set(identity['sources'])==set(old['sources'])
assert {name for name in old['sources']if identity['sources'][name]!=old['sources'][name]}=={'direct_sympy_projection_constraints.py','direct_sympy_sqrt.py'}
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
 assert d['stats']['completedPaths']==(14 if mib==32 else 22) and 'budget' in d['stop'].lower()
 assert d['stats']['lastArm']['bodyCharacters']==(20426239 if mib==32 else 19448599)
 assert d['stats']['lastArm']['guardCharacters']==(10378893 if mib==32 else 10152959)
 assert not (root/('full-'+str(mib)+'.expr')).exists();diagnostics.append(d)
def counts(name):
 text=(root/name).read_text()
 return {key:int(re.search(r'ℹ '+key+r' (\d+)',text)[1])for key in ('tests','pass','fail','skipped')}
small,whole=counts('focused-tests.log'),counts('general-tests.log')
assert small==previous['integrationTests'] and small['fail']==0
failures=re.findall(r'^test at (\S+)',(root/'general-tests.log').read_text(),re.M)
assert whole==previous['wholeProjectSuite']['current'],whole
assert sorted(failures)==sorted(previous['wholeProjectSuite']['sameFailingTestLocations'])
proof=(root/'sqrt-proof.log').read_text();assert 'Ran 3 tests' in proof and '\nOK\n' in proof
for message in ('cases=1835032 mismatches=0','cases=524290 mismatches=0 midpoints=0','cases=25167601 mismatches=0 oddTies=0 evenTies=0','cases=58720257 mismatches=0'):
 assert message in proof
rms=(root/'rms-proof.log').read_text();assert 'Ran 2 tests' in rms and '\nOK\n' in rms
assert 'cases=37773316 selected=552300 violations=0' in rms
assert 'cases=319488 selected=9 violations=0' in rms
run=load('full-run');assert run['compilerIdentity']==identity and not run['complete']
assert not (root/'full.expr').exists()
slab=load('slab');assert slab['compilerIdentity']==identity and not slab['complete']
assert not (root/'slab.expr').exists()
captured=load('candidate');assert captured['compilerIdentity']==identity and captured['candidate']==24
assert captured['bodyCharacters']==168126471
old_mixed=previous['regionalArtifacts'][2]
assert regions[2]['inputDomains']==old_mixed['inputDomains']
v={'compilerIdentity':identity,'baselineCommit':'55f144a','expressionRepresentation':'SymPy mathematical strings',
 'regionalArtifacts':regions,'regionalCases':66317,'regionalBitMismatches':0,
 'sqrtKernelInputOccurrences':{'universal':5,'narrowSingleBinade':2},
 'narrowNormalizedCertificateCases':524290,'narrowEmittedScaleCases':1835032,
 'narrowSqrtBitMismatches':0,'narrowSqrtMidpoints':0,
 'universalSqrtMantissaAndExponentCases':25167601,'universalFixedScaleCases':58720257,
 'nativeRMSPreimageCases':37773316,'nativeRMSTinyOutputCases':319488,'nativeRMSBoundViolations':0,
 'regionalMixedArtifactComparison':{'identicalInputDomains':True,'previousCharacters':old_mixed['artifact']['characters'],'currentCharacters':regions[2]['artifact']['characters']},
 'baseline96MiBCandidateNumber':14,'candidate96MiBNumber':22,'capturedOversizedCandidateNumber':24,
 'matchingCandidateComparison':False,'fullDomainDiagnostics':diagnostics,'fullDomainAttempt':run,'broaderSlabAttempt':slab,
 'integrationTests':small,'wholeProjectSuite':{'current':whole,'sameFailingTestLocations':failures,'newFailingTestLocations':0},
 'oldNumericalSavepointsCompatible':False,'fullCoordinateArtifactEmitted':False,
 'fullCoordinateParity':False,'multipleTokenParity':False,'fullVectorParity':False,
 'historicalNumericalResultsReused':False,'CUDAUsed':False,'parallelSpeedupMeasured':False}
(root/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
p=Path('docs/direct-string-validation.json');raw=p.read_text();oldmap=json.loads(raw);key='narrowSqrtAndRelativeRMSValidation'
if key in oldmap:assert oldmap[key]==v
else:
 result=raw.rstrip()[:-1].rstrip()+',\n  '+json.dumps(key)+': '+json.dumps(v,indent=2).replace('\n','\n  ')+'\n}\n'
 assert {k:value for k,value in json.loads(result).items()if k!=key}==oldmap
 p.write_text(result)
print(json.dumps({'regionalCases':66317,'mismatches':0,'integration':small,'generalSuite':whole,'fullCoordinateParity':False}))
